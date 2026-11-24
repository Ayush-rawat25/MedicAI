
from app.retrieval.intent_classifier import IntentClassifier

classifier = IntentClassifier()

queries = [
    "Give me a complete overview of this patient's medical history",
    "Summarize the patient's medical history",
    "Show all medications prescribed to the patient",
    "What allergies does the patient have?",
    "What happened during the last consultation?",
    "Has the patient's stomach pain gotten worse?",
    "Tell me about the patient's previous conditions",
    "Give me a brief overview of the patient"
]

for query in queries:
    result = classifier.classify(query)

    print("\nQuery:", query)
    print("Strategy:", result.strategy)
    print("Method:", result.method)
    print("Category:", result.record_category)
