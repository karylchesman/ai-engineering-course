from dotenv import load_dotenv
from mem0 import MemoryClient

load_dotenv()

memory_client = MemoryClient()

messages = [
    {
        "role": "user",
        "content": "Meu nome é Karyl e eu gosto de estudar inteligência artificial.",
    },
    {
        "role": "assistant",
        "content": "Olá Karyl! Anotei que você gosta de estudar inteligência artificial! Vou manter isso em mente para futuras conversas.",
    },
]


# memory_client.add(messages, user_id="karyl")

# You can also pass the message directly as a string
memory_client.add("Meu nome é Karyl e eu gosto de jogos", user_id="karyl")


query = "Qual meu nome?"

response = memory_client.search(query, filters={"user_id": "karyl"})
response
response["results"][0]["memory"]
