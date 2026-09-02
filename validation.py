from .models import ValidationResult, Answer
from .numerical import extract_numbers

def validate_answer(answer: str, retrieved_chunks: list) -> ValidationResult:
    # 1. Check refusal
    if "I don't know" in answer:
        return ValidationResult(status="refused", confidence=1.0, claims=[])

    # 2. Extract numbers in answer and check they appear in context
    answer_numbers = extract_numbers(answer)
    context_text = " ".join([c.chunk.text for c in retrieved_chunks])
    context_numbers = extract_numbers(context_text)
    # Simplified: check each number value appears in context numbers
    for anum in answer_numbers:
        if not any(abs(anum['value'] - c['value']) < 1e-6 for c in context_numbers):
            # Unsupported number found
            return ValidationResult(status="unsupported", confidence=0.0, claims=[])
    return ValidationResult(status="grounded", confidence=0.9, claims=[])