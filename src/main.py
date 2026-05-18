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

from fri_schedule_agent import build_timetable_agent, ask_timetable_agent

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

# ─────────────────────────────────────────────
# TOOL REGISTRY
# Each tool is a dict with:
#   name        – identifier the LLM uses in its decision
#   description – natural language explanation for the LLM router
#   fn          – callable(query, **kwargs) → str
# ─────────────────────────────────────────────
TOOL_REGISTRY: list[dict] = []


def register_tool(name: str, description: str):
    """Decorator that registers a function as an agent tool."""
    def decorator(fn):
        TOOL_REGISTRY.append({"name": name, "description": description, "fn": fn})
        return fn
    return decorator


# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────

@register_tool(
    name="rag_search",
    description=(
        "Search the course syllabi knowledge base. Use this for questions about "
        "courses, ECTS credits, semesters, professors, course content, prerequisites, "
        "or any academic programme information."
    ),
)
def rag_search_tool(query: str, rag_chain=None, history: list = None, **kwargs) -> str:
    """Run a RAG lookup against the FAISS vectorstore."""
    if rag_chain is None:
        return "RAG chain not initialised."
    history = history or []
    return ask(query, rag_chain, history)


@register_tool(
    name="timetable",
    description=(
        "Query the live timetable / schedule system. Use this for questions about "
        "lecture times, room assignments, overlapping classes, day-of-week schedules, "
        "teachers in a specific slot, or any time-table related query."
    ),
)
def timetable_tool(query: str, timetable_agent=None, **kwargs) -> str:
    """Forward the query to the schedule scraping agent."""
    if timetable_agent is None:
        return "Timetable agent not initialised."
    return ask_timetable_agent(query, timetable_agent)


@register_tool(
    name="direct_answer",
    description=(
        "Answer directly from the LLM's own knowledge. Use ONLY for greetings, "
        "simple factual questions that don't require the syllabus or timetable, "
        "or meta questions about the assistant itself."
    ),
)
def direct_answer_tool(query: str, llm=None, tokenizer=None, **kwargs) -> str:
    """Let the LLM answer without retrieval."""
    if llm is None:
        return "LLM not initialised."
    messages = [
        {"role": "system", "content": "You are a helpful academic assistant for FRI, University of Ljubljana."},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return llm.invoke(prompt)


# ─────────────────────────────────────────────
# AGENT ROUTER  (Concept 1 – Autonomous Decision Making)
# ─────────────────────────────────────────────

def build_tool_description_block() -> str:
    """Format the tool registry into a prompt block for the router."""
    lines = ["Available tools:"]
    for i, tool in enumerate(TOOL_REGISTRY, 1):
        lines.append(f"  {i}. {tool['name']}: {tool['description']}")
    return "\n".join(lines)


def route_query(query: str, model, tokenizer) -> str:
    """
    Ask the LLM which tool to use for the given query.
    Returns the tool name string (e.g. 'rag_search').
    Concept 1: The LLM autonomously decides which tool is appropriate.
    """
    tool_block = build_tool_description_block()
    tool_names = [t["name"] for t in TOOL_REGISTRY]

    system = (
        "You are a query router. Given a user question, decide which tool should handle it.\n\n"
        f"{tool_block}\n\n"
        f"Respond with ONLY the tool name, one of: {tool_names}. "
        "No explanation, no punctuation — just the tool name."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, eot_token_id],
        )

    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    decision = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

    # Match decision to a known tool name (fallback: rag_search)
    for name in tool_names:
        if name in decision:
            return name
    return "rag_search"


# ─────────────────────────────────────────────
# MULTI-STEP AGENT  (Concept 2 – Multi-Step Queries)
# ─────────────────────────────────────────────

def run_agent(
    query: str,
    model,
    tokenizer,
    rag_chain,
    timetable_agent,
    llm,
    history: list,
    max_steps: int = 3,
) -> str:
    tool_kwargs = {
        "rag_chain": rag_chain,
        "timetable_agent": timetable_agent,
        "llm": llm,
        "tokenizer": tokenizer,
        "history": history,
    }

    tool_map = {t["name"]: t["fn"] for t in TOOL_REGISTRY}

    gathered_context: list[str] = []
    steps_taken: list[str] = []

    for step in range(1, max_steps + 1):
        routing_query = query
        if gathered_context:
            prior = "\n\n".join(gathered_context)
            routing_query = (
                f"Original question: {query}\n\n"
                f"Information gathered so far:\n{prior}\n\n"
                "Is this sufficient? If not, which tool should be called next?"
            )

        chosen_tool = route_query(routing_query, model, tokenizer)
        print(f"[Agent step {step}] → tool: {chosen_tool}")
        steps_taken.append(chosen_tool)

        fn = tool_map[chosen_tool]
        result = fn(query, **tool_kwargs)
        gathered_context.append(f"[{chosen_tool}]: {result}")

        # Early-exit: if direct_answer was chosen or we have enough context
        if chosen_tool == "direct_answer" or _result_is_sufficient(result, model, tokenizer, query):
            break

    # Synthesise final answer from all gathered context
    if len(gathered_context) == 1:
        return gathered_context[0].split("]:", 1)[-1].strip()

    return _synthesise(query, gathered_context, model, tokenizer, history)


def _result_is_sufficient(result: str, model, tokenizer, query: str) -> bool:
    """
    Quick LLM check: is the retrieved result enough to answer the query?
    Returns True → stop the loop.  False → keep going.
    """
    system = (
        "You are a quality checker. Given a user question and a retrieved result, "
        "decide if the result fully answers the question.\n"
        "Reply with exactly one word: YES or NO."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {query}\n\nResult: {result[:800]}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, eot_token_id],
        )

    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    verdict = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()
    return verdict.startswith("YES")


def _synthesise(query: str, context_parts: list[str], model, tokenizer, history: list) -> str:
    """Merge results from multiple tools into one coherent answer."""
    combined = "\n\n".join(context_parts)
    system = (
        "You are an academic assistant. Using the information retrieved by multiple tools, "
        "write a clear, concise answer to the user's question.\n\n"
        f"RETRIEVED INFORMATION:\n{combined}"
    )
    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": query})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, eot_token_id],
        )

    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
    md_content = pymupdf4llm.to_markdown(doc)
    with open(output_path / name, "w", encoding="utf-8") as out:
        out.write(md_content)
    doc.close()


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


# def load_model(model_name):
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )
#     print("Loading model...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "left"
#     print("Model loaded.")
#     return model, tokenizer

def load_model(model_name, cache_dir):
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
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
    for turn in inputs.get("history", []):
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": inputs["question"]})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
        description="Process a conversation query with RAG + Agent pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", help="Path to input JSON file with conversation history")
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

    model, tokenizer = load_model(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "/d/hpc/projects/onj_fri/srpski_jezik_i_knjizevnost"
    )
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    timetable_agent = build_timetable_agent(model, tokenizer)

    query = "Tell me some courses where I can learn about embebbed-systems."

    context_size = extract_chunk_number(model, tokenizer, query)
    retriever = get_retriever(vectorstore, context_size, search_type="mmr")

    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=2048,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, eot_token_id],
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

    # Load conversation history
    input_file = args.input_file
    print(f"Loading conversation from: {input_file}")
    chat_history = load_conversation_from_file(input_file)

    # Preprocess query
    query = preprocess_query(query)
    print(f"[Vprašanje]: {query}")

    answer = run_agent(
        query=query,
        model=model,
        tokenizer=tokenizer,
        rag_chain=rag_chain,
        timetable_agent=timetable_agent,
        llm=llm,
        history=chat_history,
    )

    print(f"\nOdgovor: {answer}")

    # Save updated history
    chat_history.append({"question": query, "answer": answer})
    save_conversation_to_file(input_file, chat_history, query, answer)
    print(f"\nConversation saved to: {input_file}")