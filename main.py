from langchain_ollama import ChatOllama
from datetime import  datetime


llm = ChatOllama(
    model="qwen3.5:4b",
    temperature=0.5
)




messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)