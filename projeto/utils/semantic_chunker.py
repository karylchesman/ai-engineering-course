import warnings
from collections import defaultdict

import hdbscan
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

warnings.simplefilter(action="ignore", category=FutureWarning)


class SemanticChunker:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_cluster_size: int = 3,
        orphan_cluster_size: int = 2,
        max_tokens: int = 300,
    ):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.min_cluster_size = min_cluster_size
        self.orphan_cluster_size = orphan_cluster_size
        self.max_tokens = max_tokens

    def create_chunks(self, text_content: str):
        paragraphs = [
            p.strip() for p in text_content.split("\n") if len(p.strip()) > 10
        ]
        if not paragraphs:
            return []

        embeddings = self.model.encode(paragraphs, show_progress_bar=False)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=self.orphan_cluster_size, metric="euclidean"
        ).fit_predict(embeddings)

        clusters = defaultdict(list)
        orphans = []

        for idx, label in enumerate(labels):
            if label != -1:
                clusters[label].append(paragraphs[idx])
            else:
                orphans.append(paragraphs[idx])

        final_chunks = []
        for cluster_paragraphs in clusters.values():
            current_chunk = []
            current_tokens = 0
            for paragraph in cluster_paragraphs:
                paragraph_tokens = len(
                    self.tokenizer.encode(paragraph, add_special_tokens=False)
                )
                if (
                    current_tokens + paragraph_tokens > self.max_tokens
                    and current_chunk
                ):
                    final_chunks.append("\n\n".join(current_chunk))
                    current_chunk = [paragraph]
                    current_tokens = paragraph_tokens
                else:
                    current_chunk.append(paragraph)
                    current_tokens += paragraph_tokens
            if current_chunk:
                final_chunks.append("\n\n".join(current_chunk))

        if len(orphans) > 1:
            orphan_embeddings = self.model.encode(orphans, show_progress_bar=False)
            orphan_labels = hdbscan.HDBSCAN(
                min_cluster_size=self.orphan_cluster_size, metric="euclidean"
            ).fit_predict(orphan_embeddings)

            orphan_clusters = defaultdict(list)
            single_orphans = []

            for idx, label in enumerate(orphan_labels):
                if label != -1:
                    orphan_clusters[label].append(orphans[idx])
                else:
                    single_orphans.append(orphans[idx])

            for orphan_cluster_paragraphs in orphan_clusters.values():
                current_chunk = []
                current_tokens = 0
                for paragraph in orphan_cluster_paragraphs:
                    paragraph_tokens = len(
                        self.tokenizer.encode(paragraph, add_special_tokens=False)
                    )
                    if (
                        current_tokens + paragraph_tokens > self.max_tokens
                        and current_chunk
                    ):
                        final_chunks.append("\n\n".join(current_chunk))
                        current_chunk = [paragraph]
                        current_tokens = paragraph_tokens
                    else:
                        current_chunk.append(paragraph)
                        current_tokens += paragraph_tokens
                if current_chunk:
                    final_chunks.append("\n\n".join(current_chunk))

            final_chunks.extend(single_orphans)
        elif len(orphans) == 1:
            final_chunks.extend(orphans[0])
        return final_chunks
