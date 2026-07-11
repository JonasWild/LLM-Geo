from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import Embeddings

from llm_geo.operations.registry import RegisteredOperation
from llm_geo.operations.retrieval import OperationRetriever


def _placeholder() -> str:
    return "ok"


class FakeEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float("urban" in lowered or "city" in lowered),
            float("raster" in lowered),
            0.1,
        ]


def operation(name: str, docstring: str) -> RegisteredOperation:
    function = lambda: "ok"
    function.__name__ = name
    function.__doc__ = docstring
    return RegisteredOperation(
        id=f"example.{name}",
        function=function,
        module="example",
        name=name,
        description=docstring.splitlines()[0],
        inputs=(),
        defaults={},
        output_type="str",
        output_description="Result.",
    )


class OperationRetrieverTests(unittest.TestCase):
    def test_small_catalog_bypasses_embeddings(self) -> None:
        embeddings = FakeEmbeddings()
        with tempfile.TemporaryDirectory() as directory:
            retriever = OperationRetriever(
                embeddings, Path(directory), embedding_identity="fake"
            )
            operations = (operation("urban", "Find urban places."),)
            self.assertEqual(retriever.select("urban", operations, limit=1), operations)
        self.assertEqual(embeddings.document_calls, 0)

    def test_hybrid_retrieval_maps_docstring_to_operation(self) -> None:
        embeddings = FakeEmbeddings()
        with tempfile.TemporaryDirectory() as directory:
            retriever = OperationRetriever(
                embeddings, Path(directory), embedding_identity="fake"
            )
            operations = (
                operation("raster", "Render a raster image."),
                operation("cities", "Retrieve cities and urban areas."),
                operation("identity", "Return input unchanged."),
            )
            selected = retriever.select("find urban areas", operations, limit=1)
        self.assertEqual([item.id for item in selected], ["example.cities"])

    def test_unchanged_catalog_reuses_faiss_index(self) -> None:
        embeddings = FakeEmbeddings()
        with tempfile.TemporaryDirectory() as directory:
            retriever = OperationRetriever(
                embeddings, Path(directory), embedding_identity="fake"
            )
            operations = (
                operation("raster", "Render a raster image."),
                operation("cities", "Retrieve cities and urban areas."),
            )
            retriever.select("urban", operations, limit=1)
            retriever.select("raster", operations, limit=1)
        self.assertEqual(embeddings.document_calls, 1)


if __name__ == "__main__":
    unittest.main()
