import os
import uuid

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

load_dotenv()

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "financial"
SPARSE_MODEL = "Qdrant/bm25"
FILE_PATH = "./AAPL_10-K_1A_temp.md"

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
)

with open(FILE_PATH, "r", encoding="utf-8") as file:
    content = file.read()

paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

chunks[0]

dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

points = []
for idx, chunk in enumerate(chunks):
    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_embedding = list(sparse_model.passage_embed([chunk]))[0].as_object()
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            # type: ignore Pylance isn't recognizing the dict structure type,
            # expects Dict[str, float] but numpy is returning Dict[str, NumpyArray]
            "dense": dense_embedding,
            "sparse": sparse_embedding,
        },
        payload={"text": chunk, "source": FILE_PATH},
    )
    points.append(point)

qdrant_client.upload_points(collection_name=COLLECTION_NAME, points=points)

query_text = "What are the main financial risks?"
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()

search_result = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        # type: ignore Pylance isn't recognizing the dict structure type,
        # expects Dict[str, float] but numpy is returning Dict[str, NumpyArray]
        {
            "query": query_dense,
            "using": "dense",
            "limit": 10,
        },
        {
            "query": query_sparse,
            "using": "sparse",
            "limit": 10,
        },
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=3,
)

for item in search_result.points:
    print(f"Score: {item.score}")
    print(f"Text: {item.payload.get('text') if item.payload else 'N/A'[:100]}...")
    print("-----")
