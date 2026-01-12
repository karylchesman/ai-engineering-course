import os
import uuid

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from sympy import content

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "financial"
FILE_PATH = "./AAPL_10-K_1A_temp.md"

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

with open(FILE_PATH, "r", encoding="utf-8") as file:
    content = file.read()

paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

chunks[0]

model = TextEmbedding(model_name=MODEL_NAME)

points = []
for idx, chunk in enumerate(chunks):
    embedding = list(model.passage_embed([chunk]))[0].tolist()
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"text": chunk, "source": FILE_PATH},
    )
    points.append(point)

qdrant_client.upload_points(collection_name=COLLECTION_NAME, points=points)

query_text = "What are the main financial risks?"
query_embedding = list(model.query_embed([query_text]))[0].tolist()

search_result = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=3,
)

for item in search_result.points:
    print(f"Score: {item.score}")
    print(f"Text: {item.payload.get('text') if item.payload else 'N/A'[:100]}...")
    print("-----")
