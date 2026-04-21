import asyncio
import subprocess
import sys
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# Model 1: Ollama for parsing raw scan output (simple extraction, no need for 120B)
scanner_llm = ChatOllama(
    base_url="http://192.168.178.70:11434",
    model="qwen3.5:27b",
    temperature=0.1,
)

# Model 2: vLLM Qwen3-235B for pentest analysis - best tool calling capability
pentest_llm = ChatOpenAI(
    base_url="http://192.168.178.13:8003/v1",
    api_key="EMPTY",
    model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    temperature=0.3,
)

# Model 3: Ollama for the final report (qwen3.5 follows constraints better than gemma4)
report_llm = ChatOllama(
    base_url="http://192.168.178.70:11434",
    model="qwen3.5:27b",
    temperature=0.3,
)


def run_nmap(ip: str) -> str:
    # -T4: aggressive timing (safe on LAN/same-rack)
    # --min-rate 1000: don't drop below 1000 pkt/s
    # -n: skip reverse DNS (saves per-host lookup)
    # --version-intensity 5: slightly less thorough banner grabbing (default 7)
    cmd = [
        "nmap", "-sV", "-Pn", "-n",
        "-T4", "--min-rate", "1000",
        "--version-intensity", "5",
        ip,
    ]
    print(f"[*] Running {' '.join(cmd)} ...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"[!] nmap stderr: {result.stderr}", file=sys.stderr)
    return result.stdout


async def main(target_ip: str):
    print("=" * 60)
    print(f"AI-Assisted Security Assessment — Target: {target_ip}")
    print("=" * 60)
    print("\n[!] Only run this against systems you are authorized to test.\n")

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
    print("\n[+] Stage 1 — Parsed services:\n")
    print(parsed.content)

    # Stage 2: pentest analysis using MetasploitMCP tools
    mcp_client = MultiServerMCPClient({
        "metasploit": {
            "command": "/home/user/metasploit-mcp.sh",
            "args": [],
            "transport": "stdio",
        }
    })
    msf_tools = await mcp_client.get_tools()
    pentest_llm_with_tools = pentest_llm.bind_tools(msf_tools, tool_choice="auto")

    tools_by_name = {t.name: t for t in msf_tools}
    print(f"[*] Loaded {len(msf_tools)} Metasploit tools: "
          f"{', '.join(tools_by_name.keys())}\n")

    pentest_msgs = [
        SystemMessage(content=(
            f"You are a penetration testing assistant supporting an authorized "
            f"assessment against {target_ip}. You have access to these Metasploit tools:\n"
            f"- list_exploits(search_term): Search exploits by service name (e.g., 'proftpd', 'apache', 'samba')\n"
            f"- run_auxiliary_module(module_name, options): Run scanner modules\n"
            f"- run_exploit(module_name, options, payload): Run exploits\n"
            f"- list_active_sessions(): Check if you got shells\n\n"
            f"CRITICAL: Use 'module_name' parameter, NOT 'module_path'!\n"
            f"Example: run_auxiliary_module(module_name='scanner/ftp/ftp_version', options={{'RHOSTS': '{target_ip}'}})\n\n"
            f"Process:\n"
            f"1. Search exploits for each service individually\n"
            f"2. Try auxiliary scanners like 'scanner/ftp/ftp_version', 'scanner/ssh/ssh_version'\n"
            f"3. If scanners find vulnerabilities, try relevant exploits from your search results\n"
            f"4. Check for active sessions after each exploit\n"
            f"5. Summarize what worked\n\n"
            f"You found these exploits earlier - try scanning first, then exploit if promising:\n"
            f"ProFTPD: proftpd_133c_backdoor, proftpd_modcopy_exec\n"
            f"CUPS: cups_bash_env_exec, cups_ipp_remote_code_execution"
        )),
        HumanMessage(content=(
            f"Target {target_ip} has these services:\n{parsed.content}\n\n"
            f"Start by searching for exploits against each service individually."
        )),
    ]

    # Agent loop: keep running tools until the LLM stops calling them
    # High limit since this is local infrastructure - electricity isn't a concern
    max_iterations = 50
    for iteration in range(max_iterations):
        print(f"\n[*] Agent iteration {iteration + 1}/{max_iterations}")
        response = await pentest_llm_with_tools.ainvoke(pentest_msgs)
        print(f"[DEBUG] Raw response content: '{response.content}'")
        print(f"[DEBUG] Tool calls: {response.tool_calls}")
        pentest_msgs.append(response)

        if response.content:
            print(f"[LLM] {response.content}")

        if not response.tool_calls:
            if not response.content:
                print("[!] Model returned empty content AND no tool calls — "
                      "likely a tool-binding or server config issue.")
                print(f"[!] additional_kwargs={response.additional_kwargs}")
                print(f"[!] response_metadata={response.response_metadata}")
            print("[*] No more tool calls — agent done.")
            pentest_plan = response
            break

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            print(f"\n[TOOL] -> {name}({args})")

            if name not in tools_by_name:
                result = f"ERROR: unknown tool {name}"
            else:
                try:
                    result = await tools_by_name[name].ainvoke(tool_call)
                except Exception as e:
                    result = f"ERROR: {e}"

            result_str = str(result)
            preview = result_str[:500] + ("..." if len(result_str) > 500 else "")
            print(f"[TOOL] <- {preview}")

            pentest_msgs.append(ToolMessage(
                content=result_str,
                tool_call_id=tool_call["id"],
            ))
    else:
        print(f"[!] Hit max iterations ({max_iterations}) — stopping.")
        pentest_plan = response

    print("\n[+] Stage 2 — Pentest analysis:\n")
    print(pentest_plan.content)

    # Stage 3: final report
    report_msgs = [
        SystemMessage(content=(
            "You are a security report writer. Produce a short, professional "
            "command-line report (under 300 words). STRICT RULES:\n"
            "- Only report findings that appear verbatim in the tool outputs below.\n"
            "- If no exploits were successfully verified, say so explicitly.\n"
            "- Do not invent CVEs, dates, compromise indicators, or severity ratings "
            "that are not present in the provided data.\n"
            "- Clearly distinguish between 'confirmed exploitable' and 'potentially "
            "vulnerable (not verified)'."
        )),
        HumanMessage(content=(
            f"Target: {target_ip}\n\n"
            f"Discovered services:\n{parsed.content}\n\n"
            f"Pentest analysis:\n{pentest_plan.content}\n\n"
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
        print("Usage: python main2.py <target_ip>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
