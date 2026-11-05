import sys
import os
from dotenv import load_dotenv
from pydantic import BaseModel ,Field
from evaluations.retrieval_methods import calculate_mrr,calculate_ndcg
from src.llm import answer_question,answer_question_v2,answer_question_v3
from evaluations.test_file import TestQuestion,load_tests
from langchain_ollama import ChatOllama
from utility.rrf import reciprocal_rank_fusion_v2
from langchain_core.documents import Document
from utility.cross_Encoder import cross_encoder_v1
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from langchain_openai import ChatOpenAI
load_dotenv(override=True)


MODEL_NAME = os.getenv("MODEL_NAME")
VECTOR_DB = os.getenv("VECTOR_DB")
MAX_WORKERS= int(os.getenv("MAX_WORKERS"))

class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr : float = Field(description="Mean Reciprocal Rank (MRR) for retrieval performance")
    ncdg : float = Field(description="Normalized Discounted Cumulative Gain (NDCG) for retrieval performance")
    keywords_found_K : int = Field(description="Number of keywords found in top K retrieved context")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")

class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


def evaluate_retrieval(test_question:TestQuestion,bm25_index) -> RetrievalEval:
    """
    Evaluate retrieval performance for a test question.

    Args:
        test: TestQuestion object containing question and keywords
        k: Number of top documents to retrieve (default 10)

    Returns:
        RetrievalEval object with MRR, nDCG, and keyword coverage metrics
    """
    print(f"Evaluating retrieval for question: {test_question.question}")
    # retrieved_docs=fetch_context(test_question.question)
    retrieved_docs_old : list[Document]=reciprocal_rank_fusion_v2(test_question.question,bm25_index)
    print(f"Retrieved {len(retrieved_docs_old)} documents for question: {test_question.question}")

    ## Before Calculating MRR and nDCG -- > Pass them through ReRanker
    retrieved_docs = cross_encoder_v1(retrieved_docs_old,test_question.question)
    ## Calucate MRR and nDCG for each keyword and average them

    mrr_scores=[calculate_mrr(keyword,retrieved_docs) for keyword in test_question.keywords]
    ndcg_scores=[calculate_ndcg(keyword,retrieved_docs) for keyword in test_question.keywords]

    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test_question.keywords)
    keyword_coverage = (keywords_found / total_keywords) * 100 if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ncdg=avg_ndcg,
        keywords_found_K=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage
    )

def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """
    Evaluate answer quality using LLM-as-a-judge (async).

    Args:
        test: TestQuestion object containing question and reference answer

    Returns:
        Tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """
    # Get RAG response using shared answer module
    generated_answer, retrieved_docs = answer_question(test.question)

    # LLM judge prompt
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]

    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(AnswerEval)
    judge_response = structured_llm.invoke(judge_messages)
    return judge_response, generated_answer, retrieved_docs

def evaluate_answer_v2(test: TestQuestion,bm25_index) -> tuple[AnswerEval, str, list]:
    generated_answer, retrieved_docs = answer_question_v3(test.question,bm25_index)

    print("Answer generated , now LLM as a judge")
     # LLM judge prompt
    """
    Evaluate answer quality using LLM-as-a-judge (async).

    Args:
        test: TestQuestion object containing question and reference answer

    Returns:
        Tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """

    # LLM judge prompt
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(AnswerEval)
    judge_response = structured_llm.invoke(judge_messages)
    return judge_response, generated_answer, retrieved_docs

def evaluate_all_retrieval(bm25_index):
    """Evaluate all retrieval tests."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test,bm25_index)
        progress = (index + 1) / total_tests
        yield test, result, progress

def evaluate_all_retrieval_v2(bm25_index):
    """Evaluate all retrieval tests."""
    tests = load_tests()
    total_tests = len(tests)
    completed=0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor :
        futures=[
            executor.submit(evaluate_retrieval,test,bm25_index) for test in tests
        ]

        for test,future in zip (tests,futures):
            result = future.result()
            completed+=1
            progress=completed/total_tests    
            yield test, result, progress        

def evaluate_all_answers():
    """Evaluate all answers to tests using batched async execution."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress

def evaluate_all_answers_v2(bm25_index):
    """Evaluate all answers to tests using batched async execution."""
    tests = load_tests()
    total_tests = len(tests)
    completed=0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_test = {
        executor.submit(evaluate_answer_v2, test, bm25_index): test
        for test in tests
    }

        for future in as_completed(future_to_test):

            test = future_to_test[future]
            result = future.result()[0]

            completed += 1
            progress = completed / total_tests

            yield test, result, progress

def run_cli_evaluation(test_number: int):
    """Run evaluation for a specific test (async helper for CLI)."""
    # Load tests
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    # Get the test
    test = tests[test_number]

    # Print test info
    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    # Retrieval Evaluation
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ncdg:.4f}")
    print(f"Keywords Found: {retrieval_result.keywords_found_K}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

      # Answer Evaluation
    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")

    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


# run_cli_evaluation(0)        