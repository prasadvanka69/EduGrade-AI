import database
import vector_db


def index_all_assessments():

    assessments = database.get_all_assessments_for_vector_db()

    print(f"Found {len(assessments)} assessments in SQLite.")

    for assessment in assessments:

        vector_db.add_assessment(
            assessment_id=assessment["assessment_id"],
            assignment_prompt=assessment["assignment_prompt"] or "",
            assignment_type=assessment["assignment_type"] or "",
            rubric=assessment["rubric"] or "",
            submission=assessment["submission"] or "",
            feedback=assessment["feedback"] or "",
        )

        print(
            f"Indexed Assessment #{assessment['assessment_id']}"
        )

    print("\nIndexing completed.")

    print(
        "Total documents in ChromaDB:",
        vector_db.collection.count()
    )


if __name__ == "__main__":
    index_all_assessments()