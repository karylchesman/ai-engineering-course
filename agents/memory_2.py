import os

from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "openai_base_url": "https://api.groq.com/openai/v1",
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0,
            "api_key": os.environ["GROQ_API_KEY"],
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "memory",
            "url": QDRANT_URL,
            "api_key": QDRANT_API_KEY,
            "embedding_model_dims": 384,
        },
    },
    "embedder": {
        "provider": "fastembed",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
}

openai_client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"]
)
memory = Memory.from_config(config)


def chat_with_memories(message: str, user_id: str = "default_user"):
    relevant_memories = memory.search(message, user_id=user_id, limit=3)
    memories_str = "\n".join(
        f"- {entry['memory']}" for entry in relevant_memories["results"]
    )
    input_prompt = f"""
    Você é um assistente pessoal. Responda à pergunta considerando as memórias do usuário.
    Memórias do usuário: {memories_str}
    Pergunta: {message}
    """

    response = openai_client.responses.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        input=input_prompt,
    )

    assistant_response = response.output_text
    messages = [
        {
            "role": "user",
            "content": message,
        },
        {
            "role": "assistant",
            "content": assistant_response,
        },
    ]
    memory.add(messages, user_id=user_id)
    return assistant_response


def main():
    print("Chat com IA (digite sair para encerrar)")
    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() == "sair":
            print("Encerrando o chat. Até a próxima!")
            break
        response = chat_with_memories(user_input)
        print(f"IA: {response}")


if __name__ == "__main__":
    main()
