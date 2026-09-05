from __future__ import annotations

import chromadb
from pathlib import Path


DB_DIR = Path(__file__).parent / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="edugrade_assessments"
)


def add_assessment(
    assessment_id: int,
    assignment_prompt: str,
    assignment_type: str,
    rubric: str,
    submission: str,
    feedback: str,
) -> None:
    document = f"""
Assignment Type: {assignment_type}

Assignment Prompt:
{assignment_prompt}

Rubric:
{rubric}

Student Submission:
{submission}

Feedback:
{feedback}
""".strip()

    collection.upsert(
        ids=[str(assessment_id)],
        documents=[document],
        metadatas=[
            {
                "assessment_id": str(assessment_id),
                "assignment_type": assignment_type,
            }
        ],
    )

def search_similar(
    query: str,
    n_results: int = 5,
    max_distance: float = 1.5,
) -> list[dict]:

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    matches = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        distance = float(distance)

        if distance <= max_distance:

            matches.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

    return matches
