import os

from dotenv import load_dotenv
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION_NAME = "financial"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)


query_text = "What are the main financial risks?"
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()
query_colbert = list(colbert_model.query_embed([query_text]))[0].tolist()

search_result = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        # type: ignore Pylance isn't recognizing the dict structure type,
        # expects Dict[str, float] but numpy is returning Dict[str, NumpyArray]
        {
            "prefetch": [
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
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit": 20,
        }
    ],
    query=query_colbert,
    using="colbert",
    limit=3,
)

max_score = max(p.score for p in search_result.points)
for item in search_result.points:
    normalized_score = item.score / max_score if max_score > 0 else 0
    print(f"Score: {normalized_score}")
    print(f"Text: {item.payload.get('text') if item.payload else 'N/A'[:100]}...")
    print("-----")
