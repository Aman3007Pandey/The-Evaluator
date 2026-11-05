import math


def calculate_mrr(keyword:str, retrieved_docs:list) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for a single keyword."""
    keyword_lower = keyword.lower()
    for idx,doc in enumerate(retrieved_docs):
        if keyword_lower in doc.page_content.lower():
            return 1 / (idx + 1)  # Reciprocal rank is 1/(rank)
    return 0.0  # Return 0 if keyword is not found


def calculate_dcg(relevances: list[int]) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(len(relevances)):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()

    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]

    # DCG
    dcg = calculate_dcg(relevances)

    # Ideal DCG (best case: keyword in first position)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances)

    return dcg / idcg if idcg > 0 else 0.0