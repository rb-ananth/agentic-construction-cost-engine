from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_RULE_CODE = re.compile(r"\[Code:\s*([A-Z]{2,}-\d+)\]")
_SECTION = re.compile(r"SECTION\s+(\d+)", re.IGNORECASE)
_CATEGORY_BY_PREFIX = {
    "MAT": "materials",
    "EQ": "equipment",
    "LAB": "labor",
}


class CostRuleEngine:
    """Persistent semantic index over the construction cost specification rules."""

    def __init__(
        self,
        rules_path: Path | None = None,
        vector_store_path: Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        distance_cutoff: float = 0.76,
    ) -> None:
        project_root = Path(__file__).resolve().parent
        self.rules_path = rules_path or project_root / "data" / "spec_rules.txt"
        self.vector_store_path = vector_store_path or project_root / "data" / "vector_store"
        self.distance_cutoff = distance_cutoff

        if not self.rules_path.is_file():
            raise FileNotFoundError(f"Specification rules not found: {self.rules_path}")
        if not math.isfinite(distance_cutoff) or distance_cutoff <= 0:
            raise ValueError("distance_cutoff must be finite and greater than zero")

        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=str(self.vector_store_path))
        self.collection = self.client.get_or_create_collection(
            name="construction_spec_rules",
            metadata={"hnsw:space": "cosine"},
        )
        self._index_rules()

    def _index_rules(self) -> None:
        rules = [
            line.strip()
            for line in self.rules_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rules:
            raise ValueError(f"No specification rules found in {self.rules_path}")

        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for index, rule in enumerate(rules):
            code_match = _RULE_CODE.search(rule)
            if code_match is None:
                raise ValueError(f"Rule is missing a [Code: ...] value: {rule}")

            cost_code = code_match.group(1)
            prefix = cost_code.split("-", maxsplit=1)[0]
            section_match = _SECTION.search(rule)
            documents.append(rule)
            ids.append(f"{cost_code}-{index}")
            metadatas.append(
                {
                    "cost_code": cost_code,
                    "category": _CATEGORY_BY_PREFIX.get(prefix, "unclassified"),
                    "section": section_match.group(1) if section_match else "",
                    "source": self.rules_path.name,
                }
            )

        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()
        existing_ids = set(self.collection.get()["ids"])
        stale_ids = existing_ids.difference(ids)
        if stale_ids:
            self.collection.delete(ids=sorted(stale_ids))
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Indexed %d cost specification rules", len(documents))

    def lookup(self, description: str) -> dict[str, Any]:
        """Return the nearest rule and whether it passes the distance cutoff."""
        normalized_description = description.strip()
        if not normalized_description:
            raise ValueError("Invoice description must not be empty")

        query_embedding = self.model.encode(
            normalized_description,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        if not documents or not metadatas or not distances:
            raise RuntimeError("Vector search returned no cost specification results")

        distance = float(distances[0])
        if not math.isfinite(distance):
            raise RuntimeError(f"Vector search returned a non-finite distance: {distance}")
        metadata = dict(metadatas[0])
        in_distribution = distance <= self.distance_cutoff
        result = {
            "in_distribution": in_distribution,
            "cost_code": metadata["cost_code"] if in_distribution else None,
            "matched_rule": documents[0],
            "metadata": metadata,
            "distance": distance,
            "distance_cutoff": self.distance_cutoff,
        }
        logger.info(
            "RAG lookup completed (in_distribution=%s, distance=%.4f, cost_code=%s)",
            in_distribution,
            distance,
            result["cost_code"],
        )
        return result
