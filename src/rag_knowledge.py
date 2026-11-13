import hashlib
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.embeddings import Embeddings


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / "knowledge_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
MIN_RELEVANCE_SCORE = 0.82


class E5Embeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Для RAG установи зависимости из requirements.txt") from exc
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            [f"passage: {text}" for text in texts],
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(
            f"query: {text}",
            normalize_embeddings=True,
        )
        return vector.tolist()


class RAGKnowledgeBase:
    def __init__(
        self,
        knowledge_dir: Path = KNOWLEDGE_DIR,
        vector_db_dir: Path = VECTOR_DB_DIR,
        top_k: int = 2,
        min_similarity: float = MIN_RELEVANCE_SCORE,
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Для RAG установи зависимости из requirements.txt") from exc

        self.knowledge_dir = Path(knowledge_dir)
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.embeddings = E5Embeddings()
        self.client = chromadb.PersistentClient(path=str(vector_db_dir))
        self.collection = self.client.get_or_create_collection(
            name="personal_assistant_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        self.sync_documents()

    @staticmethod
    def _split_text(text: str, chunk_size: int = 900) -> List[str]:
        units = []
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = " ".join(paragraph.split())
            if paragraph:
                units.extend(re.split(r"(?<=[.!?])\s+", paragraph))

        chunks = []
        current = []
        current_size = 0

        for unit in units:
            words = unit.split()
            pieces = []
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if len(candidate) > chunk_size and piece:
                    pieces.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                pieces.append(piece)

            for piece in pieces:
                if current and current_size + len(piece) + 1 > chunk_size:
                    chunks.append(" ".join(current))
                    current = []
                    current_size = 0
                current.append(piece)
                current_size += len(piece) + 1

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _read_documents(self) -> List[dict]:
        documents = []
        if not self.knowledge_dir.exists():
            return documents

        for path in sorted(self.knowledge_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8-sig").strip()
            source = path.relative_to(self.knowledge_dir).as_posix()
            for chunk_index, content in enumerate(self._split_text(text)):
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                document_id = hashlib.sha256(f"{source}:{chunk_index}".encode("utf-8")).hexdigest()
                documents.append({
                    "id": document_id,
                    "content": content,
                    "source": source,
                    "chunk_index": chunk_index,
                    "content_hash": content_hash,
                })
        return documents

    def sync_documents(self) -> None:
        documents = self._read_documents()
        existing = self.collection.get(include=["metadatas"])
        existing_metadata = dict(zip(existing.get("ids", []), existing.get("metadatas", [])))
        current_ids = {document["id"] for document in documents}
        stale_ids = [document_id for document_id in existing_metadata if document_id not in current_ids]
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        changed = [
            document
            for document in documents
            if existing_metadata.get(document["id"], {}).get("content_hash") != document["content_hash"]
        ]
        if not changed:
            return

        vectors = self.embeddings.embed_documents([document["content"] for document in changed])
        self.collection.upsert(
            ids=[document["id"] for document in changed],
            documents=[document["content"] for document in changed],
            embeddings=vectors,
            metadatas=[
                {
                    "source": document["source"],
                    "chunk_index": document["chunk_index"],
                    "content_hash": document["content_hash"],
                }
                for document in changed
            ],
        )

    def search(self, question: str, top_k: Optional[int] = None) -> List[dict]:
        count = self.collection.count()
        if not count:
            return []

        result = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(question)],
            n_results=min(top_k or self.top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            hits.append({
                "document": document,
                "source": metadata.get("source", "неизвестный источник"),
                "similarity": 1 - distance,
            })
        return hits

    def format_context(self, question: str) -> str:
        hits = self.search(question)
        if not hits:
            return "В базе знаний пока нет документов."
        fragments = [
            f"[Источник: {hit['source']}; сходство: {hit['similarity']:.3f}]\n{hit['document']}"
            for hit in hits
            if hit["similarity"] >= self.min_similarity
        ]
        return "\n\n".join(fragments) if fragments else "В базе знаний нет релевантных фрагментов."
