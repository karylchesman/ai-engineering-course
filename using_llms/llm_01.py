from groq import Groq
from dotenv import load_dotenv

load_dotenv()

model_key = "llama-3.1-8b-instant"

client = Groq()

response = client.chat.completions.create(
    model=model_key,
    messages=[
        {"role": "system", "content": "Atue como um especialista em Machine Learning."},
        {
            "role": "user",
            "content": "De forma resumida, explique o que é Machine Learning?",
        },
    ],
    temperature=0.5,
    top_p=1,
)

print(response.choices[0].message.content)
