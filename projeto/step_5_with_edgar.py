import os
import uuid

from dotenv import load_dotenv
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from utils.edgar_client import EdgarClient
from utils.semantic_chunker import SemanticChunker

load_dotenv()

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "financial"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
MAX_TOKENS = 300
EMAIL_ADDRESS = "karylbps@gmail.com"

edgar_client = EdgarClient(email=EMAIL_ADDRESS)

data_10k = edgar_client.fetch_filling_date(ticker="AAPL", form_type="10-K")
text_10k = edgar_client.get_combined_text(data_10k)

data_10q = edgar_client.fetch_filling_date(ticker="AAPL", form_type="10-Q")
text_10q = edgar_client.get_combined_text(data_10q)

chunker = SemanticChunker(max_tokens=MAX_TOKENS)

all_chunks = []
for data, text in [(data_10k, text_10k), (data_10q, text_10q)]:
    chunks = chunker.create_chunks(text)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "metadata": data["metadata"]})

dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)

points = []
for chunk_data in all_chunks:
    chunk = chunk_data["text"]
    metadata = chunk_data["metadata"]

    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_embedding = list(sparse_model.passage_embed([chunk]))[0].as_object()
    colbert_embedding = list(colbert_model.passage_embed([chunk]))[0].tolist()
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            # type: ignore Pylance isn't recognizing the dict structure type,
            # expects Dict[str, float] but numpy is returning Dict[str, NumpyArray]
            "dense": dense_embedding,
            "sparse": sparse_embedding,
            "colbert": colbert_embedding,
        },
        payload={"text": chunk, "metadata": metadata},
    )
    points.append(point)

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

qdrant_client.upload_points(
    collection_name=COLLECTION_NAME, points=points, batch_size=5
)
