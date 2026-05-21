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
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

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
    # EMBED_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    # EMBED_MODEL = "BAAI/bge-large-en-v1.5"
    # EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    all_docs = load_markdown_documents(source_dir)

    if not all_docs:
        raise ValueError(f"No PDF documents found in: {source_dir}")

    # Manually split on course headers (######) and extract course names
    chunks = []
    for doc in all_docs:
        # Split on lines starting with ######
        lines = doc.page_content.split('\n')
        current_course_name = None
        current_chunk_lines = []
        
        for line in lines:
            if line.startswith('Predmet: '):
                # Save previous chunk if it exists
                if current_chunk_lines:
                    chunk_content = '\n'.join(current_chunk_lines)
                    if chunk_content.strip():
                        chunk_doc = Document(
                            page_content=chunk_content,
                            metadata={
                                "source": doc.metadata["source"],
                                "course_name": current_course_name or "Unknown Course"
                            }
                        )
                        chunks.append(chunk_doc)
                
                # Extract new course name
                current_course_name = line.replace('Predmet: ', '').strip()
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)
        
        # Save final chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            if chunk_content.strip():
                chunk_doc = Document(
                    page_content=chunk_content,
                    metadata={
                        "source": doc.metadata["source"],
                        "course_name": current_course_name or "Unknown Course"
                    }
                )
                chunks.append(chunk_doc)
    
    print(f"After manual header splitting: {len(chunks)} chunks")
    print(f"Sample course names: {[c.metadata.get('course_name') for c in chunks[:5]]}")
    print(f"Average chunk size : {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
    print(f"Min / Max          : {min(len(c.page_content) for c in chunks)} / {max(len(c.page_content) for c in chunks)} chars")
    
    # Stage 2: Further split large chunks with RecursiveCharacterTextSplitter
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1500,
        chunk_overlap=200,
    )
    
    final_chunks = []
    for chunk in chunks:
        # Only split chunks that are larger than 4000 chars
        if len(chunk.page_content) > 4000:
            sub_chunks = recursive_splitter.split_text(chunk.page_content)
            for sub_chunk in sub_chunks:
                # Preserve course_name metadata when splitting
                doc_obj = Document(page_content=sub_chunk, metadata=chunk.metadata)
                final_chunks.append(doc_obj)
        else:
            # Keep small chunks as-is (already under 4000 chars, no need to split further)
            final_chunks.append(chunk)
    
    print(f"After recursive splitting: {len(final_chunks)} chunks")
    print(f"Average chunk size : {sum(len(c.page_content) for c in final_chunks) / len(final_chunks):.0f} chars")
    print(f"Min / Max          : {min(len(c.page_content) for c in final_chunks)} / {max(len(c.page_content) for c in final_chunks)} chars")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"Embedding model loaded: {EMBED_MODEL}")
    print(f"Indexing {len(final_chunks)} chunks into FAISS...")
    vectorstore = FAISS.from_documents(final_chunks, embeddings)
    print("Done.")
    print(f"Index size: {vectorstore.index.ntotal} vectors of dimension {vectorstore.index.d}")

    vectorstore_path = Path(vectorstore_dir)
    vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(vectorstore_dir)
    print(f"Saved vectorstore to: {vectorstore_dir}")
    return vectorstore


def get_retriever(vectorstore, context_size, search_type="similarity"):
    size_fac = context_size if context_size is not None else 1
    retrieve_num = size_fac * 3

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


def make_rag_prompt(inputs: dict) -> str:
    # system = (
    #     "You are an academic advisor assistant for the University of Ljubljana, "
    #     "Faculty of Computer Science. You have access to course syllabi.\n\n"
    #     "RESPONSE RULES:\n"
    #     "- Answer ONLY from the provided context\n"
    #     "- Be concise and match the detail level of the question:\n"
    #     "  * If asked for course information/details → always include: course name, ECTS credits, semester\n"
    #     "  * If asked for descriptions or recommendations → provide relevant details from syllabus\n"
    #     "- If information is not in the context, say 'I don't have enough information'\n"
    #     "- When asked about scheduling or conflicts, use the available tools\n"
    #     "- Do not use any tool unless it is needed for timetable conflicts.\n"
    #     "- Do not repeat the same courses in your answer when listing them.\n\n"
    #     f"CONTEXT:\n{inputs['context']}"
    # )
    # return f"{system}\n\nQuestion: {inputs['question']}"
    system = (
        "Ste akademski svetovalni asistent za Univerzo v Ljubljani, "
        "Fakulteto za računalništvo in informatiko. Imate dostop do učnih načrtov predmetov.\n\n"
        "PRAVILA ODGOVARJANJA:\n"
        "- Odgovarjajte SAMO na podlagi podanega konteksta\n"
        "- Bodite jedrnati in prilagodite podrobnost odgovora vprašanju:\n"
        "  * Če vas vprašajo po informacijah/podrobnostih o predmetu → vedno vključite: ime predmeta, število ECTS kreditnih točk, semester\n"
        "  * Če vas vprašajo po opisu ali priporočilih → navedite ustrezne podrobnosti iz učnega načrta\n"
        "- Če informacije ni v kontekstu, recite 'Nimam dovolj informacij'\n"
        "- Ko vas vprašajo o urniku ali konfliktih, uporabite razpoložljiva orodja\n"
        "- Ne ponavljajte istih predmetov v svojem odgovoru, ko jih naštevate.\n\n"
        f"KONTEKST:\n{inputs['context']}"
    )
    return f"{system}\n\nVprašanje: {inputs['question']}"


def format_docs(docs):
    """Format documents with course name metadata visible to the model."""
    formatted = []
    for d in docs:
        course_name = d.metadata.get('course_name', 'Unknown Course')
        section = d.metadata.get('section', '')
        source = d.metadata.get('source', 'unknown')
        
        # Build header with all relevant metadata
        header = f"[Course: {course_name}"
        if section:
            header += f" | Section: {section}"
        header += f" | Source: {source}]"
        
        formatted.append(f"{header}\n{d.page_content}")
    
    return "\n\n---\n\n".join(formatted)

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

    # system_message = (
    #     "You extract only numerical information that changes how much context a model needs. "
    #     "Return an integer only when the user explicitly asks for a count, quantity, or number of items to retrieve. "
    #     "Ignore incidental digits such as academic year, semester, age, dates, IDs, or class year. "
    #     "If the number is part of background information and not a request for how many items to answer with, return None. "
    #     "If the query is asking about a certain number of courses, then ALWAYS return that number. "
    #     "Do not explain your answer.\n\n"
    #     f"EXAMPLES:\n{examples_string}"
    # )
    # user_message = (
    #     "Query: "
    #     f"{input_query}\n\n"
    #     "Return exactly one token: an integer like 3, or None."
    # )

    system_message = (
        "Izluščite samo numerične informacije, ki spremenijo, koliko konteksta model potrebuje. "
        "Vrnite celo število samo, ko uporabnik izrecno vpraša po številu, količini ali koliko elementov naj pridobi. "
        "Prezrite naključne številke, kot so študijsko leto, semester, starost, datumi, ID-ji ali letnik. "
        "Če je številka del ozadja in ne zahteva po tem, s koliko elementi naj odgovorim, vrnite None. "
        "Če poizvedba sprašuje po določenem številu predmetov, VEDNO vrnite to število. "
        "Ne razlagajte svojega odgovora.\n\n"
        f"PRIMERI:\n{examples_string}"
    )
    user_message = (
        "Poizvedba: "
        f"{input_query}\n\n"
        "Vrnite natanko en žeton: celo število, kot je 3, ali None."
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
    parser.add_argument("--documents-dir", default="../documents", help="Directory to scan for documents when building the index")
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
        # exit(0)
    else:
        vectorstore = FAISS.load_local(
            args.vectorstore_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )


    llm = build_agent(args.models_dir)

    query = "Ali mi lahko poves par predmetov vezanih na vgrajene sisteme?"

    context_size = extract_chunk_number(llm, query)
    # retriever = get_retriever(vectorstore, context_size, search_type="mmr")
    retriever = get_retriever(vectorstore, context_size)

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