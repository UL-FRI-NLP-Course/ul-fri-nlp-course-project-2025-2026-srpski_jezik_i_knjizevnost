import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from functools import partial
import json
import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, MarianMTModel, MarianTokenizer
from pydantic import BaseModel, field_validator
from langchain_core.documents import Document

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fri_schedule_agent import build_agent

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

def load_markdown_documents(folder: str) -> list[Document]:
    docs = []
    for path in Path(folder).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs


def embed_documents(source_dir: str, vectorstore_dir: str):
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    all_docs = load_markdown_documents(source_dir)

    if not all_docs:
        raise ValueError(f"No PDF documents found in: {source_dir}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(all_docs)
    print(f"Documents → chunks: {len(all_docs)} → {len(chunks)}")
    print(f"Average chunk size : {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
    print(f"Min / Max          : {min(len(c.page_content) for c in chunks)} / {max(len(c.page_content) for c in chunks)} chars")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
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
    size_fac = context_size if context_size is not None else 1
    retrieve_num = size_fac * 6

    print(f"Retrieving {retrieve_num} chunks")

    if search_type == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieve_num},
        )
    elif search_type == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": retrieve_num, "fetch_k": 30 + retrieve_num, "lambda_mult": 0.6},
        )
    else:
        raise ValueError("Unknown search type.")


def load_model(model_name, cache_dir):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Model loaded.")
    return model, tokenizer

def make_rag_prompt(inputs: dict) -> str:
    system = (
        "You are an academic advisor assistant for the University of Ljubljana, "
        "Faculty of Computer Science. You have access to course syllabi.\n"
        "Answer ONLY from the provided context. Be specific about course names.\n"
        "If listing courses, always include: course name, ECTS credits, and semester.\n"
        "If the answer is not in the context, say 'I don't have enough information.'\n\n"
        f"CONTEXT:\n{inputs['context']}"
    )
    return f"{system}\n\nQuestion: {inputs['question']}"


def format_docs(docs):
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


def load_conversation_from_file(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("history", [])


def save_conversation_to_file(filepath: str, history: list[dict], current_query: str = None, answer: str = None) -> None:
    output = {"history": history}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def extract_chunk_number(agent, input_query):
    class ChunkNumberResponse(BaseModel):
        """Validated response for chunk number extraction."""
        chunk_count: int | None = None
        
        @field_validator('chunk_count')
        @classmethod
        def validate_chunk_count(cls, v):
            if v is not None and v <= 0:
                raise ValueError("chunk_count must be positive")
            return v
    
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
    
    prompt = f"{system_message}\n\n{user_message}"
    result = agent.invoke(prompt)
    print(f"Agent response: {result}")
    
    # Extract integer from the response using regex
    match = re.search(r'\d+', str(result).strip())
    chunk_count = int(match.group()) if match else None
    
    # Validate using Pydantic
    try:
        response = ChunkNumberResponse(chunk_count=chunk_count)
        print(f"Extracted number: {response.chunk_count}")
        return response.chunk_count if response.chunk_count is not None else 1
    except ValueError as e:
        print(f"Validation error: {e}. Using default value of 1.")
        return 1


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process a conversation query with RAG + Agent pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", help="Path to input JSON file with conversation history")
    parser.add_argument("--models-dir", default="/d/hpc/projects/onj_fri/srpski_jezik_i_knjizevnost", help="Directory where models are stored")
    parser.add_argument("--documents-dir", default="..", help="Directory to scan for documents when building the index")
    parser.add_argument("--vectorstore-dir", default="../vectorstore/faiss_index", help="FAISS index directory")
    parser.add_argument("--build-index", action="store_true", default=False, help="Rebuild vectorstore before answering")
    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_arguments()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )

    if args.build_index:
        print(f"Building vectorstore from: {args.documents_dir}")
        vectorstore = embed_documents(args.documents_dir, args.vectorstore_dir)
    else:
        vectorstore = FAISS.load_local(
            args.vectorstore_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    # model, tokenizer = load_model(
    #     "Qwen/Qwen2.5-7B-Instruct",
    #     args.models_dir
    # )

    llm = build_agent(args.models_dir)

    query = "Tell me some courses where I can learn about embebbed-systems."

    context_size = extract_chunk_number(llm, query)
    retriever = get_retriever(vectorstore, context_size, search_type="mmr")

    RAG_PROMPT = RunnableLambda(partial(make_rag_prompt))
    rag_chain = (
        {
            "context":  RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
            "question": RunnableLambda(lambda x: x["question"]),
            "history":  RunnableLambda(lambda x: x.get("history", [])),
        }
        | RAG_PROMPT
        | llm
        # | StrOutputParser()
    )

    input_file = args.input_file
    print(f"Loading conversation from: {input_file}")
    chat_history = load_conversation_from_file(input_file)

    print(f"[Vprašanje]:\n {query}")

    answer = rag_chain.invoke({
        "question": query,
        "history": chat_history
    })

    print(f"[Odgovor]:\n {answer}")

    chat_history.append({"question": query, "answer": answer})
    save_conversation_to_file(input_file, chat_history, query, answer)
    print(f"\nConversation saved to: {input_file}")