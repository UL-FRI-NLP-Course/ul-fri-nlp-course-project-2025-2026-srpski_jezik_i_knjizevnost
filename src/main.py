import warnings
warnings.filterwarnings("ignore")

import pymupdf
from pathlib import Path
import os
import re
import numpy as np
from functools import partial
import json
import sys
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, MarianMTModel, MarianTokenizer
from langchain_core.documents import Document

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

import pymupdf4llm 

input_path = [
    Path("../Računalništvo_in_informatika_UNI(2026-2027).pdf"),
    Path("../Računalništvo_in_informatika_VSS(2026-2027).pdf")
]

output_path = Path("../documents")

SHORTCUTS = {
    "uni":  "univerzitetni program",
    "vss":  "visokošolski strokovni program",
    "ects": "kreditne točke",
    "let":  "letnik",
}


def _rect_overlap_area(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    return float(inter.width * inter.height)


def _is_mostly_inside(inner: pymupdf.Rect, outer: pymupdf.Rect, threshold: float = 0.6) -> bool:
    overlap = _rect_overlap_area(inner, outer)
    area = float(inner.width * inner.height)
    if area <= 0:
        return False
    return (overlap / area) >= threshold


def _to_markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    col_count = max(len(r) for r in rows)

    def norm_cell(v: str) -> str:
        if v is None:
            return ""
        # Keep cell text readable and safe for markdown tables.
        return str(v).replace("\n", " ").replace("|", "\\|").strip()

    normalized = []
    for row in rows:
        padded = [norm_cell(cell) for cell in row] + [""] * (col_count - len(row))
        normalized.append(padded)

    header = normalized[0]
    separators = ["---"] * col_count
    body_rows = normalized[1:]

    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separators) + " |",
    ]
    for row in body_rows:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)


def document_to_markdown(input_path: Path, output_path: Path, name: str) -> None:
    doc = pymupdf.open(input_path)
    
    # Uporabite pymupdf4llm za ekstrakcijo v markdown
    md_content = pymupdf4llm.to_markdown(doc)
    
    with open(output_path / name, "w", encoding="utf-8") as out:
        out.write(md_content)
    
    doc.close()

def load_markdown_documents(folder: str) -> list[Document]:
    docs = []
    for path in Path(folder).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path)}
            )
        )
    return docs

def embed_documents(source_dir: str, vectorstore_dir: str):
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    all_docs = load_markdown_documents(source_dir)

    if not all_docs:
        raise ValueError(f"No PDF documents found in: {source_dir}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # Target size in characters
        chunk_overlap=150,       # 100-char overlap between consecutive chunks
        length_function=len,
        add_start_index=True,    # Store the original char offset in metadata
    )

    chunks = text_splitter.split_documents(all_docs)

    print(f"Documents → chunks: {len(all_docs)} → {len(chunks)}")
    print(f"Average chunk size : {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
    print(f"Min / Max          : {min(len(c.page_content) for c in chunks)} / {max(len(c.page_content) for c in chunks)} chars")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # L2-normalise → cosine similarity = dot product
    )

    print(f"Embedding model loaded: {EMBED_MODEL}")

    print(f"Indexing {len(chunks)} chunks into FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Done.")
    print(f"Index size: {vectorstore.index.ntotal} vectors of dimension {vectorstore.index.d}")

    vectorstore_path = Path(vectorstore_dir)
    vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(vectorstore_dir)
    print(f"Saved vectorstore to: {vectorstore_dir}")

    return vectorstore

def get_retriever(vectorstore, context_size, search_type="similarity"):
    if context_size is None:
        size_fac = 1
    else:
        size_fac = context_size

    retrieve_num = size_fac * 6

    if search_type == "similarity":
        # --- Similarity search ---
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieve_num},
        )
    elif search_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": retrieve_num, "fetch_k": 30 + retrieve_num, "lambda_mult": 0.6},
            # fetch_k: initial candidate pool size (larger = better diversity quality)
            # lambda_mult: relevance weight (0.5 = equal balance)
        )
    else:
        raise ValueError("Unknown search type.")
    
    return retriever

def load_model(model_name):
    MODEL_NAME = model_name

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — optimal for bell-curve weight distributions
        bnb_4bit_use_double_quant=True,     # Quantise the quantisation constants too (saves ~0.4 bits/param)
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for matrix multiplications
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically distribute across available GPUs
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # Left padding for inference: generated tokens follow the last real token, not padding tokens
    print("Model loaded.")

    return model, tokenizer

def make_rag_prompt(inputs: dict, tokenizer) -> str:
    system = (
        "You are an academic advisor assistant for the University of Ljubljana, "
        "Faculty of Computer Science. You have access to course syllabi.\n"
        "Answer ONLY from the provided context. Be specific about course names.\n"
        "If listing courses, always include: course name, ECTS credits, and semester.\n"
        "If the answer is not in the context, say 'I don't have enough information.'\n\n"
        f"CONTEXT:\n{inputs['context']}"
    )

    messages = [{"role": "system", "content": system}]
    
    # history massages
    for turn in inputs.get("history", []):
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    
    # current massages
    messages.append({"role": "user", "content": inputs["question"]})
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_docs(docs): # Helper function
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('title', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

def ask(question_en: str, rag_chain, history: list) -> str:
    print("-" * 60)
    answer = rag_chain.invoke({"question": question_en, "history": history})
    print(answer)
    print("=" * 60)
    return answer

def preprocess_query(text: str) -> str:
    words = text.split()
    expanded = [SHORTCUTS.get(w.lower(), w) for w in words]
    return " ".join(expanded)

def load_conversation_from_file(filepath: str) -> tuple[list[dict], str]:
    """
    Load conversation history and current query from JSON file.
    File format:
    {
        "history": [
            {"question": "...", "answer": "..."},
            {"question": "...", "answer": "..."}
        ],
    }
    Returns: history list
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("history", [])

def save_conversation_to_file(filepath: str, history: list[dict], current_query: str = None, answer: str = None) -> None:
    output = {"history": history}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

def process_query_with_history(question: str, rag_chain, history: list[dict]) -> str:
    """Process a single query with conversation history."""
    # Preprocess the question
    question = preprocess_query(question)
    print(f"[Vprašanje]: {question}")
    
    # Get answer from LLM with full history
    answer = ask(question, rag_chain, history)
    print(f"\n Odgovor: {answer}")
    
    return answer

def extract_chunk_number(model, tokenizer, input_query):
    with open("../examples/examples.md", "r", encoding="utf-8") as f:
        examples_string = f.read()

    system_message = (
        "You extract only numerical information that changes how much context a model needs. "
        "Return an integer only when the user explicitly asks for a count, quantity, or number of items to retrieve. "
        "Ignore incidental digits such as academic year, semester, age, dates, IDs, or class year. "
        "If the number is part of background information and not a request for how many items to answer with, return None. "
        "If the query is asking about a certain number of courses, then ALWAYS return that number. "
        "Do not explain your answer.\n\n"
        f"EXAMPLES:\n{examples_string}"
    )
    user_message = (
        "Query: "
        f"{input_query}\n\n"
        "Return exactly one token: an integer like 3, or None."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(prompt, return_tensors="pt")
    model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_token_ids = [tokenizer.eos_token_id, eot_token_id]

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_ids,
        )

    new_tokens = generated_ids[0][model_inputs["input_ids"].shape[-1]:]
    raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if raw_answer.lower().startswith("none"):
        return None

    match = re.search(r"-?\d+", raw_answer)
    if match is not None:
        return int(match.group(0))

    return None

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process a conversation query with RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        help="Path to input JSON file with conversation history"
    )
    parser.add_argument(
        "--documents-dir",
        default="..",
        help="Directory to scan recursively for PDF documents when building the index"
    )
    parser.add_argument(
        "--vectorstore-dir",
        default="../vectorstore/faiss_index",
        help="Directory where the FAISS index is saved and loaded from"
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Rebuild the vectorstore from --documents-dir before answering",
        default=False
    )

    return parser.parse_args()

if __name__ == "__main__":
    # To run the program all arguments must be provided in the command line first!!!
    args = parse_arguments()

    embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
    
    if args.build_index:
        print(f"Building vectorstore from documents in: {args.documents_dir}")
        vectorstore = embed_documents(args.documents_dir, args.vectorstore_dir)
    else:
        vectorstore = FAISS.load_local(
                args.vectorstore_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )

    model, tokenizer = load_model("meta-llama/Meta-Llama-3-8B-Instruct")
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        
    query = "Tell me that name of two courses that deal with embedded systems."
    context_size = extract_chunk_number(model, tokenizer, query)

    retriever = get_retriever(vectorstore, context_size, search_type="mmr")
    # results = get_relevant_chunks(vectorstore) # This is to test retrieval

    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=2048,   
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, eot_token_id]
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    RAG_PROMPT = RunnableLambda(partial(make_rag_prompt, tokenizer=tokenizer))

    rag_chain = (
    {
        "context":  RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
        "question": RunnableLambda(lambda x: x["question"]),
        "history":  RunnableLambda(lambda x: x.get("history", [])),
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

    input_file = args.input_file
    output_file = args.input_file
    
    print(f"Loading conversation from: {input_file}")
    chat_history = load_conversation_from_file(input_file)

    answer = process_query_with_history(query, rag_chain, chat_history)
    
    chat_history.append({"question": query, "answer": answer})
    
    save_conversation_to_file(output_file, chat_history, query, answer)
    print(f"\nConversation saved to: {output_file}")