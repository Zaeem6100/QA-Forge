import asyncio
import json
import os
import re
import zipfile
from typing import Literal

import pandas as pd
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from pydantic import BaseModel, Field
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
MCQ_FILE = "data/test_dataset_mcq.csv"
SAQ_FILE = "data/test_dataset_saq.csv"
TRAIN_DATA = "data/processed_dataset_cleaned.csv"
OUTPUT_DIR = "evaluation_results_ollama_rag"

# RAG Config
OLLAMA_LLM = "llama3:8b"
EMBED_MODEL = "nomic-embed-text"
USE_WEB_SEARCH = False  # Set to True to enable web search
RAG_TOP_K = 5  # Number of documents to retrieve

# Generation parameters
TEMPERATURE = 0.0  # 0 = deterministic, higher = more creative
MAX_TOKENS = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# PYDANTIC MODELS
# ----------------------------
class MCQAnswer(BaseModel):
    """Structured MCQ answer"""
    answer_choice: Literal["A", "B", "C", "D"] = Field(
        description="The correct answer: A, B, C, or D"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation"
    )


class SAQAnswer(BaseModel):
    """Structured SAQ answer"""
    answer: str = Field(
        description="Short answer (1-5 words)"
    )


# ----------------------------
# RAG RETRIEVER
# ----------------------------
class HybridRetriever(BaseRetriever):
    """Combines local vector store with optional web search"""
    local_retriever: any
    search_tool: any = None
    use_web: bool = False

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        # Get documents from local vector store
        local_docs = self.local_retriever.invoke(query)

        # Optionally add web search
        web_docs = []
        if self.use_web and self.search_tool:
            try:
                web_results = self.search_tool.run(query)
                if web_results:
                    web_docs = [Document(
                        page_content=web_results,
                        metadata={"source": "web"}
                    )]
            except Exception as e:
                print(f"Web search error: {e}")

        return local_docs + web_docs


def setup_rag(use_web=False):
    """Initialize RAG system with Ollama"""
    print("Setting up RAG system...")

    # Initialize embeddings
    print(f"Loading embeddings model: {EMBED_MODEL}")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Load knowledge base
    print(f"Loading knowledge base: {TRAIN_DATA}")
    loader = CSVLoader(file_path=TRAIN_DATA, encoding="utf-8")
    docs = loader.load()
    print(f"   → Loaded {len(docs)} documents")

    # Create vector store
    print("Building vector database...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    # Initialize Ollama LLM
    print(f"Loading LLM: {OLLAMA_LLM}")
    llm = ChatOllama(
        model=OLLAMA_LLM,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS
    )

    # Create retriever
    search_tool = DuckDuckGoSearchRun() if use_web else None
    retriever = HybridRetriever(
        local_retriever=vectorstore.as_retriever(search_kwargs={"k": RAG_TOP_K}),
        search_tool=search_tool,
        use_web=use_web
    )

    # Create RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    print("RAG system ready!\n")
    return rag_chain, llm


# ----------------------------
# PARSING FUNCTIONS
# ----------------------------
def extract_json(text):
    """Extract JSON from text"""
    # Try to find JSON object
    matches = re.findall(r'\{[^}]+\}', text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except:
            continue
    return None


def parse_mcq_response(text, fallback="C"):
    """Parse MCQ response with fallback"""
    # Try JSON parsing
    json_obj = extract_json(text)
    if json_obj:
        answer = json_obj.get("answer_choice", json_obj.get("answer", "")).upper()
        if answer in ["A", "B", "C", "D"]:
            return MCQAnswer(
                answer_choice=answer,
                reasoning=json_obj.get("reasoning", "")
            )

    # Fallback: search for letter in text
    text_upper = text.upper()
    for letter in ["A", "B", "C", "D"]:
        if f'"{letter}"' in text or f"'{letter}'" in text or f"answer is {letter}" in text_upper:
            return MCQAnswer(answer_choice=letter, reasoning="extracted")

    # Search for any occurrence
    for letter in ["A", "B", "C", "D"]:
        if letter in text_upper:
            return MCQAnswer(answer_choice=letter, reasoning="found in text")

    # Ultimate fallback
    return MCQAnswer(answer_choice=fallback, reasoning="fallback")


def parse_saq_response(text, fallback="unknown"):
    """Parse SAQ response with fallback"""
    # Try JSON parsing
    json_obj = extract_json(text)
    if json_obj:
        answer = json_obj.get("answer", "").strip().lower()
        if answer:
            # Limit to 5 words
            words = answer.split()[:5]
            return SAQAnswer(answer=" ".join(words))

    # Fallback: extract answer from text
    # Remove common phrases
    clean = text.lower()
    for phrase in ["the answer is", "answer:", "it is", "response:", "in conclusion"]:
        clean = clean.replace(phrase, "")

    # Get first meaningful word(s)
    words = clean.split()
    if words:
        # Take first word, clean it
        answer = words[0].strip('.,!?:;"()[]{}').lower()
        if answer and len(answer) > 0:
            return SAQAnswer(answer=answer)

    return SAQAnswer(answer=fallback)


# ----------------------------
# EVALUATION FUNCTIONS
# ----------------------------
async def evaluate_mcq(mcq_file, rag_chain, llm):
    """Evaluate MCQ using Ollama RAG"""
    print("=" * 70)
    print("MCQ EVALUATION")
    print("=" * 70)

    df = pd.read_csv(mcq_file)
    results = []

    print(f"Processing {len(df)} questions...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="MCQ"):
        mcqid = row['MCQID']
        question = row['prompt']

        # Build prompt for Ollama
        prompt = f"""Based on the retrieved context, answer this multiple choice question.

Question: {question}

Provide your answer in JSON format:
{{"answer_choice": "A/B/C/D", "reasoning": "brief explanation"}}

Answer (JSON only):"""

        # Get answer from RAG
        try:
            result = await rag_chain.ainvoke({"query": prompt})
            response = result.get('result', '')
        except Exception as e:
            print(f"\n  Error for {mcqid}: {e}")
            response = ""

        # Parse response
        parsed = parse_mcq_response(response, fallback="C")

        # Store result
        results.append({
            'MCQID': mcqid,
            'A': parsed.answer_choice == 'A',
            'B': parsed.answer_choice == 'B',
            'C': parsed.answer_choice == 'C',
            'D': parsed.answer_choice == 'D',
            'chosen_answer': parsed.answer_choice,
            'reasoning': parsed.reasoning,
            'raw_response': response[:300]
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save submission file
    submission_path = os.path.join(OUTPUT_DIR, "mcq_prediction.tsv")
    results_df[['MCQID', 'A', 'B', 'C', 'D']].to_csv(
        submission_path,
        sep='\t',
        index=False
    )
    print(f"\nSaved: {submission_path}")

    # Save detailed file
    detailed_path = os.path.join(OUTPUT_DIR, "mcq_detailed.tsv")
    results_df.to_csv(detailed_path, sep='\t', index=False)
    print(f"Saved: {detailed_path}")

    # Statistics
    print(f"\nStatistics:")
    print(f"   Total: {len(results_df)}")
    for letter in ['A', 'B', 'C', 'D']:
        count = (results_df['chosen_answer'] == letter).sum()
        print(f"   {letter}: {count} ({count / len(results_df) * 100:.1f}%)")

    return results_df


async def evaluate_saq(saq_file, rag_chain, llm):
    """Evaluate SAQ using Ollama RAG"""
    print("\n" + "=" * 70)
    print("SAQ EVALUATION")
    print("=" * 70)

    df = pd.read_csv(saq_file)
    results = []

    print(f"Processing {len(df)} questions...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="SAQ"):
        question_id = row['ID']
        question = row['en_question'] if pd.notna(row.get('en_question')) else row['question']

        # Build prompt for Ollama
        prompt = f"""Based on the retrieved context, answer this question with a SHORT answer (1-5 words maximum).

Question: {question}

Provide your answer in JSON format:
{{"answer": "your short answer"}}

Answer (JSON only):"""

        # Get answer from RAG
        try:
            result = await rag_chain.ainvoke({"query": prompt})
            response = result.get('result', '')
        except Exception as e:
            print(f"Error for {question_id}: {e}")
            response = ""

        # Parse response
        parsed = parse_saq_response(response, fallback="unknown")

        # Store result
        results.append({
            'ID': question_id,
            'answer': parsed.answer,
            'raw_response': response[:300]
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save submission file
    submission_path = os.path.join(OUTPUT_DIR, "saq_prediction.tsv")
    results_df[['ID', 'answer']].to_csv(
        submission_path,
        sep='\t',
        index=False
    )
    print(f"Saved: {submission_path}")

    # Save detailed file
    detailed_path = os.path.join(OUTPUT_DIR, "saq_detailed.tsv")
    results_df.to_csv(detailed_path, sep='\t', index=False)
    print(f"Saved: {detailed_path}")

    # Statistics
    print(f"Statistics:")
    print(f"   Total: {len(results_df)}")
    print(f"   Unique answers: {results_df['answer'].nunique()}")
    avg_words = results_df['answer'].str.split().str.len().mean()
    print(f" Avg answer length: {avg_words:.1f} words")

    return results_df


# ----------------------------
# MAIN
# ----------------------------
async def main():
    print("\n" + "=" * 70)
    print("OLLAMA RAG EVALUATION SYSTEM")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  LLM Model: {OLLAMA_LLM}")
    print(f"  Embedding Model: {EMBED_MODEL}")
    print(f"  Web Search: {'Enabled' if USE_WEB_SEARCH else 'Disabled'}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Retrieval Top-K: {RAG_TOP_K}")
    print("=" * 70)

    # Setup RAG
    rag_chain, llm = setup_rag(use_web=USE_WEB_SEARCH)

    # Run evaluations
    mcq_results = await evaluate_mcq(MCQ_FILE, rag_chain, llm)
    saq_results = await evaluate_saq(SAQ_FILE, rag_chain, llm)

    # Create submission package
    print("\n" + "=" * 70)
    print("Creating submission package...")
    zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(
            os.path.join(OUTPUT_DIR, "mcq_prediction.tsv"),
            arcname="mcq_prediction.tsv"
        )
        zipf.write(
            os.path.join(OUTPUT_DIR, "saq_prediction.tsv"),
            arcname="saq_prediction.tsv"
        )

    print(f"Created: {zip_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
