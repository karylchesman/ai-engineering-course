import json

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"]
)


def search_kb(query: str):
    response = requests.post(
        "http://localhost:8000/search",
        json={"query": query, "limit": 3},
    )
    return response.json()


tools = [
    {
        "type": "function",
        "name": "search_kb",
        "description": "Busca informações na base de conhecimento para responder perguntas.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A pergunta do usuário"},
            },
            "required": ["query"],
        },
    },
]

input_list = [{"role": "user", "content": "What are the AAPL main financial risks?"}]

response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    tools=tools,
    input=input_list,
)

input_list += response.output

for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        result = search_kb(**args)
        texts = [r["text"] for r in result["results"]]
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({"results": texts}, ensure_ascii=False),
            }
        )

final_response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    instructions="Responsa à pergunta do usuário usando as informações retornadas na busca.",
    tools=tools,
    input=input_list,
)
print(final_response.output_text)
