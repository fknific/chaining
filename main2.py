import asyncio
import subprocess
import sys
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# Model 1: vLLM for parsing raw scan output
scanner_llm = ChatOpenAI(
    base_url="http://your-vllm-server:8000/v1",
    api_key="EMPTY",
    model="gpt-oss:120b",
)

# Model 2: Ollama for pentest analysis (tools bound later)
pentest_llm = ChatOllama(
    base_url="http://192.168.178.70:11434",
    model="gpt-oss:20b",
    temperature=0.3,
)

# Model 3: Ollama for the final report
report_llm = ChatOllama(
    base_url="http://192.168.178.70:11434",
    model="gemma4:31b",
    temperature=0.5,
)


def run_nmap(ip: str) -> str:
    print(f"[*] Running nmap -sV -Pn against {ip} ...")
    result = subprocess.run(
        ["nmap", "-sV", "-Pn", ip],
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
    async with MultiServerMCPClient({
        "metasploit": {
            "url": "http://127.0.0.1:8080/sse",  # MetasploitMCP SSE endpoint
            "transport": "sse",
        }
    }) as mcp_client:
        msf_tools = mcp_client.get_tools()
        pentest_llm_with_tools = pentest_llm.bind_tools(msf_tools)

        tools_by_name = {t.name: t for t in msf_tools}
        print(f"[*] Loaded {len(msf_tools)} Metasploit tools: "
              f"{', '.join(tools_by_name.keys())}\n")

        pentest_msgs = [
            SystemMessage(content=(
                f"You are a penetration testing assistant supporting an authorized "
                f"assessment against {target_ip}. You have full access to a Metasploit "
                f"RPC backend through the provided tools. Your job:\n"
                f"1. Search for relevant exploit and auxiliary modules for the services.\n"
                f"2. Run auxiliary scanner modules to confirm vulnerabilities.\n"
                f"3. If a check confirms vulnerability, run the matching exploit module "
                f"with RHOSTS={target_ip} and an appropriate payload.\n"
                f"4. After each exploit, list active sessions to verify success.\n"
                f"5. When done, write a concise summary of confirmed vulnerabilities "
                f"and any sessions obtained.\n\n"
                f"Be methodical: one tool call at a time, observe each result, "
                f"then decide the next step. Stop calling tools when you have "
                f"enough evidence to write the summary."
            )),
            HumanMessage(content=(
                f"Discovered services on {target_ip}:\n\n{parsed.content}\n\n"
                f"Find applicable Metasploit modules, verify which vulnerabilities "
                f"are real, and exploit them to confirm. Report what works."
            )),
        ]

        # Agent loop: keep running tools until the LLM stops calling them
        max_iterations = 15
        for iteration in range(max_iterations):
            print(f"\n[*] Agent iteration {iteration + 1}/{max_iterations}")
            response = await pentest_llm_with_tools.ainvoke(pentest_msgs)
            pentest_msgs.append(response)

            if response.content:
                print(f"[LLM] {response.content}")

            if not response.tool_calls:
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
            "command-line report (under 300 words) summarizing the assessment "
            "process, key findings, and recommended next steps."
        )),
        HumanMessage(content=(
            f"Target: {target_ip}\n\n"
            f"Discovered services:\n{parsed.content}\n\n"
            f"Pentest analysis:\n{pentest_plan.content}\n\n"
            "Write the final report."
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
