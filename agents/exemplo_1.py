import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"]
)

response = client.responses.parse(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    input="""Daniel e Alberto vão gravar uma aula na terça-feira.""",
    instructions="Extraia informações do evento.",
    text_format=CalendarEvent,
)

print(response.output_parsed)
