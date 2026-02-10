import json
import os

import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"]
)


def get_stock(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    output = {
        "ticker": ticker,
        "company_name": info.get("shortName", ticker),
        "current_price": info.get("currentPrice", 0),
    }
    return json.dumps(output)


tools = [
    {
        "type": "function",
        "name": "get_stock",
        "description": "Retorna informações básicas sobre uma ação.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Símbolo da ação, por exemplo, AAPL para Apple.",
                },
            },
            "required": ["ticker"],
        },
    },
]

input_list = [
    {
        "role": "user",
        "content": "Qual é o preço atual da ação da Apple?",
    }
]

response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    tools=tools,
    input=input_list,
)

response.output

for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        result = get_stock(**args)
        # For openai compatibility, it requires the call object to be in the input list
        # input_list.append(item)
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": result,
            }
        )

final_response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    instructions="Responsa com uma análise baseada nos dados retornados pela função.",
    tools=tools,
    input=input_list,
)

final_response.output_text
