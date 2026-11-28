import sys
import logging
from pathlib import Path

# Add MedicRAG directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
MEDICRAG_ROOT = PROJECT_ROOT / "MedicRAG"

sys.path.insert(0, str(MEDICRAG_ROOT))

from app.vectorStore.chroma_store import ChromaStore

logging.basicConfig(level=logging.INFO)


def main():
    store = ChromaStore()
    patient_id = "P001"

    print("\nChromaDB cleanup")
    print("-------------------------")
    print(f"Collection count before: {store.count()}")

    existing_chunks = store.get_patient_records(patient_id=patient_id)
    print(f"Chunks belonging to {patient_id}: {len(existing_chunks)}")

    if not existing_chunks:
        print("No records found. Nothing to delete.")
        return

    confirmation = input(
        f"\nDelete ALL indexed records for {patient_id}? (yes/no): "
    )

    if confirmation.strip().lower() != "yes":
        print("Deletion cancelled.")
        return

    store.delete_patient_records(patient_id)

    remaining = store.get_patient_records(patient_id=patient_id)

    print("\nCleanup completed.")
    print(f"Remaining chunks for {patient_id}: {len(remaining)}")
    print(f"Collection count after: {store.count()}")


if __name__ == "__main__":
    main()