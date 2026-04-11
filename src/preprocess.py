import warnings
warnings.filterwarnings("ignore")

import pymupdf
from pathlib import Path
import os
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from transformers import GenerationConfig



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


def document_to_markdown(input_path: Path, output_path: Path) -> None:
    doc = pymupdf.open(input_path)
    with open(output_path, "w", encoding="utf-8", newline="\n") as out:  # create markdown output
        for page in doc:  # iterate the document pages
            blocks = page.get_text("blocks")
            table_finder = page.find_tables()
            tables = table_finder.tables if table_finder else []

            elements = []

            # Add text blocks that are not inside tables.
            for block in blocks:
                x0, y0, x1, y1, text, *_ = block
                block_rect = pymupdf.Rect(x0, y0, x1, y1)
                if any(_is_mostly_inside(block_rect, t.bbox) for t in tables):
                    continue
                if text and text.strip():
                    if text.startswith("UČNI NAČRT PREDMETA / COURSE SYLLABUS"):
                        # Add indicator of new course.
                        elements.append((y0, x0, "text", "\n\n\n" + f"{70 * "-"}" + "\n" + text.rstrip()))
                    else:
                        elements.append((y0, x0, "text", text.rstrip()))

            # Add tables as markdown at their page position.
            for t in tables:
                rows = t.extract() or []
                md_table = _to_markdown_table(rows)
                if md_table:
                    x0, y0, *_ = t.bbox
                    elements.append((y0, x0, "table", md_table))

            # Keep natural reading order on page.
            elements.sort(key=lambda e: (e[0], e[1]))

            for _, _, kind, content in elements:
                out.write(content)
                out.write("\n\n" if kind == "table" else "\n")

            # page delimiter (form feed 0x0C)
            out.write("\f")

    doc.close()

    print("saved to markdown.")

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

def embed_documents():
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    all_docs = load_markdown_documents(os.environ.get("DOCUMENTS"))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          # Target size in characters
        chunk_overlap=100,       # 100-char overlap between consecutive chunks
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

    return vectorstore

def get_retriever(vectorstore, search_type="similarity"):
    if search_type == "similarity":
        # --- Similarity search ---
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
    elif search_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
            # fetch_k: initial candidate pool size (larger = better diversity quality)
            # lambda_mult: relevance weight (0.5 = equal balance)
        )
    else:
        raise ValueError("Unknown search type.")
    
    return retriever

def get_relevant_chunks(vectorstore, search_type="similarity"):
    query = "What subject has a lot to do with embedded systems?"
    retriever = get_retriever(search_type)
    sim_results = retriever.invoke(query)

    print("=== Similarity Search Results ===")
    for i, doc in enumerate(sim_results):
        title = doc.metadata.get("title", "unknown")
        print(f"[{i+1}] Source: {title}")
        print(doc.page_content[:200])
        print()

    return sim_results

def load_model():
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

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
        "You are a precise, helpful assistant. Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say \"I don't have enough information to answer that.\"\n"
        "Do not make up facts. Cite which source(s) you used at the end of your answer.\n\n"
        f"CONTEXT:\n{inputs['context']}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": inputs["question"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_docs(docs): # Helper function
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('title', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

def ask(question: str, rag_chain):
    print(f"Question: {question}")
    print("-" * 60)
    answer = rag_chain.invoke(question)
    print(answer)
    print("=" * 60)
    return answer


if __name__ == "__main__":
    input_path = Path(os.environ.get("INPUT_PDF"))
    output_path = Path(os.environ.get("OUTPUT_MD"))
    # document_to_markdown(input_path, output_path)

    vectorstore = embed_documents()
    retriever = get_retriever(vectorstore)
    # results = get_relevant_chunks(vectorstore) # This is to test retrieval
    model, tokenizer = load_model()
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

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

    RAG_PROMPT = RunnableLambda(make_rag_prompt)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    ask(rag_chain)

    
    