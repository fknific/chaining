import subprocess
import sys
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


# Model 1: fast vLLM for parsing raw scan output
scanner_llm = ChatOpenAI(
    base_url="http://your-vllm-server:8000/v1",
    api_key="EMPTY",
    model="gpt-oss:120b",
)

# Model 2: Ollama for pentest analysis
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
    """Run an nmap service-version scan against the target."""
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


def main(target_ip: str):
    print("=" * 60)
    print(f"AI-Assisted Security Assessment — Target: {target_ip}")
    print("=" * 60)
    print("\n[!] Only run this against systems you are authorized to test.\n")

    # Stage 1: scan + parse
    raw_scan = run_nmap(target_ip)

    parse_msgs = [
        (
            "system",
            "You are a security scanner assistant. Given raw nmap output, "
            "extract a clean structured list of open ports, the service "
            "running on each, and the detected version. Be concise.",
        ),
        ("human", f"Parse this nmap output:\n\n{raw_scan}"),
    ]
    parsed = scanner_llm.invoke(parse_msgs)
    print("\n[+] Stage 1 — Parsed services:\n")
    print(parsed.content)

    # Stage 2: pentest analysis
    pentest_msgs = [
        (
            "system",
            "You are a penetration testing assistant supporting an "
            "authorized security assessment. Given a list of services and "
            "versions, identify likely attack surface, known CVEs worth "
            "investigating, and relevant Metasploit modules (give exact "
            "module paths like exploit/windows/smb/ms17_010_eternalblue). "
            "Be specific and actionable.",
        ),
        (
            "human",
            f"Target services:\n\n{parsed.content}\n\n"
            "List pentest avenues and matching Metasploit modules.",
        ),
    ]
    pentest_plan = pentest_llm.invoke(pentest_msgs)
    print("\n[+] Stage 2 — Pentest analysis:\n")
    print(pentest_plan.content)

    # Stage 3: final report
    report_msgs = [
        (
            "system",
            "You are a security report writer. Produce a short, professional "
            "command-line report (under 300 words) summarizing the assessment "
            "process, key findings, and recommended next steps.",
        ),
        (
            "human",
            f"Target: {target_ip}\n\n"
            f"Discovered services:\n{parsed.content}\n\n"
            f"Pentest analysis:\n{pentest_plan.content}\n\n"
            "Write the final report.",
        ),
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
    main(sys.argv[1])
