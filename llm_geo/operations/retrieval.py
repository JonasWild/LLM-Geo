"""Local hybrid retrieval over trusted operation docstrings."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

from llm_geo.operations.registry import RegisteredOperation


class OperationRetriever:
    """Select planner-visible operations using FAISS and lexical retrieval."""

    def __init__(
        self,
        embeddings: Embeddings,
        index_dir: Path,
        *,
        embedding_identity: str,
    ) -> None:
        self.embeddings = embeddings
        self.index_dir = index_dir
        self.embedding_identity = embedding_identity

    def select(
        self,
        task: str,
        operations: Sequence[RegisteredOperation],
        limit: int = 50,
    ) -> tuple[RegisteredOperation, ...]:
        """Return hybrid-ranked operations, bypassing retrieval for small catalogs."""
        available = tuple(operations)
        if len(available) <= limit:
            return available

        documents = [_operation_document(operation) for operation in available]
        index = self._load_or_build_index(available, documents)
        query_vector = np.asarray(
            [self.embeddings.embed_query(task)], dtype=np.float32
        )
        faiss.normalize_L2(query_vector)

        candidate_count = min(len(available), max(limit * 2, limit))
        _, semantic_indices = index.search(query_vector, candidate_count)
        semantic_ranking = [int(index) for index in semantic_indices[0] if index >= 0]

        vectorizer = TfidfVectorizer(sublinear_tf=True)
        document_matrix = vectorizer.fit_transform(documents)
        lexical_scores = (document_matrix @ vectorizer.transform([task]).T).toarray()[:, 0]
        lexical_ranking = np.argsort(-lexical_scores, kind="stable")[:candidate_count]

        fused_scores: dict[int, float] = {}
        for ranking in (semantic_ranking, lexical_ranking.tolist()):
            for rank, operation_index in enumerate(ranking, start=1):
                fused_scores[operation_index] = fused_scores.get(operation_index, 0.0) + (
                    1.0 / (60 + rank)
                )
        selected_indices = sorted(
            fused_scores,
            key=lambda index: (-fused_scores[index], index),
        )[:limit]
        return tuple(available[index] for index in selected_indices)

    def _load_or_build_index(
        self,
        operations: Sequence[RegisteredOperation],
        documents: Sequence[str],
    ) -> faiss.Index:
        fingerprint = _catalog_fingerprint(
            operations, documents, self.embedding_identity
        )
        index_path = self.index_dir / "operations.faiss"
        metadata_path = self.index_dir / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata == {"version": 1, "fingerprint": fingerprint}:
                index = faiss.read_index(str(index_path))
                if index.ntotal == len(operations):
                    return index
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            pass

        vectors = np.asarray(
            self.embeddings.embed_documents(list(documents)), dtype=np.float32
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(operations):
            raise ValueError("Embedding provider returned an unexpected vector shape")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        temporary_index = index_path.with_suffix(".faiss.tmp")
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        faiss.write_index(index, str(temporary_index))
        temporary_metadata.write_text(
            json.dumps({"version": 1, "fingerprint": fingerprint}, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_index, index_path)
        os.replace(temporary_metadata, metadata_path)
        return index


def _operation_document(operation: RegisteredOperation) -> str:
    docstring = inspect.getdoc(operation.function) or operation.description
    return f"Operation: {operation.name}\nID: {operation.id}\n\n{docstring}"


def _catalog_fingerprint(
    operations: Sequence[RegisteredOperation],
    documents: Sequence[str],
    embedding_identity: str,
) -> str:
    payload = {
        "embedding": embedding_identity,
        "operations": [
            {"id": operation.id, "document": document}
            for operation, document in zip(operations, documents, strict=True)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
