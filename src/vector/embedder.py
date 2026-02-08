"""Local embedding generation using sentence-transformers."""

import logging
from typing import Iterator

from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# Options:
# - BAAI/bge-m3: Best quality, 2.2GB, needs 6GB+ VRAM
# - BAAI/bge-base-en-v1.5: Good quality, 440MB, works on 4GB VRAM
# - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: Multilingual, 470MB
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DIMENSIONS = 384


class Embedder:
    """Generate embeddings using local sentence-transformers model."""

    def __init__(self, model: str = DEFAULT_MODEL):
        log.info(f"Loading embedding model: {model}")
        self.model = SentenceTransformer(model)
        self.dimensions = self.model.get_sentence_embedding_dimension()
        log.info(f"Model loaded. Dimensions: {self.dimensions}")

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding
        """
        # BGE-M3 can handle long texts, but truncate for safety
        if len(text) > 8000:
            text = text[:8000]

        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> Iterator[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch

        Yields:
            Embeddings for each text
        """
        # Truncate texts
        texts = [t[:8000] if len(t) > 8000 else t for t in texts]

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode(
                batch, normalize_embeddings=True, show_progress_bar=False
            )

            for emb in embeddings:
                yield emb.tolist()
