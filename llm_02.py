from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

model_key = "llama-3.1-8b-instant"

client = OpenAI(base_url="https://api.groq.com/openai/v1")

response = client.responses.create(
    model=model_key,
    instructions="Responda de forma simples em apenas 1 parágrafo curto.",
    input="O que é Machine Learning?",
    temperature=0,
)

print(response.output)
print(response.output_text)
