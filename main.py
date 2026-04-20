from langchain_ollama import ChatOllama
from datetime import  datetime
from langchain.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://your-vllm-server:8000/v1",
    api_key="EMPTY",
    model="gpt-oss:120b",
)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Fake implementation for testing
    return f"It's 18°C and sunny in {city}."



llm = ChatOllama(base_url="http://192.168.178.70:11434",
    model="gpt-oss:20b",
    temperature=0.5
)

llm2 = ChatOllama(base_url="http://192.168.178.70:11434",
    model="gemma4:31b",
    temperature=0.5
)
llm2 = llm2.bind_tools([get_weather])


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence. and add one sentence",
    ),
    ("human", "I love programming."),
]


ai_msg = llm.invoke(messages)


messages2 = [
    (
        "system",
        "You are a helpful assistant that translates English to German. Translate the user sentence.",
    ),
    ("human", ai_msg.content + "what is the weather like in berlin."),
]


ai_msg2 = llm2.invoke(messages2)

if ai_msg2.tool_calls:
    messages2.append(ai_msg2)
    for tool_call in ai_msg2.tool_calls:
        result = get_weather.invoke(tool_call)
        messages2.append(result)
    ai_msg2 = llm2.invoke(messages2)

print(ai_msg.content)
print(ai_msg2.content)

