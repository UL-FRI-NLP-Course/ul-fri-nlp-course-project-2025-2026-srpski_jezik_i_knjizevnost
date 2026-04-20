import warnings
warnings.filterwarnings("ignore")

import pymupdf
from pathlib import Path
import os
import numpy as np
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, MarianMTModel, MarianTokenizer
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from transformers import GenerationConfig

import pymupdf.layout
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


# def document_to_markdown(input_path: Path, output_path: Path, name: str, translator=None) -> None:
#     doc = pymupdf.open(input_path)
#     with open(output_path / name, "w", encoding="utf-8", newline="\n") as out:
#         for page in doc:
#             blocks = page.get_text("blocks")
#             table_finder = page.pymupdf_layout()
#             tables = table_finder.tables if table_finder else []

#             elements = []

#             for block in blocks:
#                 x0, y0, x1, y1, text, *_ = block
#                 block_rect = pymupdf.Rect(x0, y0, x1, y1)
#                 if any(_is_mostly_inside(block_rect, t.bbox) for t in tables):
#                     continue
#                 if text and text.strip():
#                     # Prevedi besedilo če je prevajalnik na voljo
#                     if translator:
#                         text = translator(text.strip())

#                     if text.startswith("UČNI NAČRT PREDMETA / COURSE SYLLABUS") or \
#                        text.startswith("Course syllabus"):  # po prevodu
#                         elements.append((y0, x0, "text", "\n\n\n" + f"{70 * '-'}" + "\n" + text.rstrip()))
#                     else:
#                         elements.append((y0, x0, "text", text.rstrip()))

#             for t in tables:
#                 rows = t.extract() or []
#                 # Prevedi celice v tabeli
#                 if translator:
#                     rows = [
#                         [translator(cell) if cell and cell.strip() else cell for cell in row]
#                         for row in rows
#                     ]
#                 md_table = _to_markdown_table(rows)
#                 if md_table:
#                     x0, y0, *_ = t.bbox
#                     elements.append((y0, x0, "table", md_table))

#             elements.sort(key=lambda e: (e[0], e[1]))

#             for _, _, kind, content in elements:
#                 out.write(content)
#                 out.write("\n\n" if kind == "table" else "\n")

#             out.write("\f")

#     doc.close()
#     print("saved to markdown.")


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

def embed_documents():
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    all_docs = load_markdown_documents(output_path) #load_markdown_documents(os.environ.get("DOCUMENTS"))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          # Target size in characters
        chunk_overlap=100,       # 100-char overlap between consecutive chunks
        length_function=len,
        add_start_index=True,    # Store the original char offset in metadata
    )

    # chunks = text_splitter.split_documents(all_docs)

    chunks = split_by_course(all_docs)

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
            search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.6},
            # fetch_k: initial candidate pool size (larger = better diversity quality)
            # lambda_mult: relevance weight (0.5 = equal balance)
        )
    else:
        raise ValueError("Unknown search type.")
    
    return retriever

def split_by_course(docs: list[Document]) -> list[Document]:
    """Razdeli dokument na manjše smiselne kose."""
    chunks = []
    
    for doc in docs:
        # Najprej razdeli po straneh (form feed \f)
        pages = doc.page_content.split('\f')
        
        for page_num, page in enumerate(pages):
            if not page.strip():
                continue
                
            # Vsako stran razdeli na odstavke/sekcije
            sections = page.split('\n\n')
            
            current_chunk = ""
            for section in sections:
                # Če dodajanje sekcije preseže mejo, shrani trenutni chunk
                if len(current_chunk) + len(section) > 2000:  # 2000 znakov ≈ 500 tokenov
                    if current_chunk:
                        chunks.append(Document(
                            page_content=current_chunk.strip(),
                            metadata={
                                "source": doc.metadata["source"],
                                "page": page_num,
                                "chunk_id": len(chunks)
                            }
                        ))
                    current_chunk = section
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + section
                    else:
                        current_chunk = section
            
            # Zadnji chunk
            if current_chunk:
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={
                        "source": doc.metadata["source"],
                        "page": page_num,
                        "chunk_id": len(chunks)
                    }
                ))
    
    return chunks

def get_relevant_chunks(vectorstore, search_type="similarity"):
    # query = "What subject has a lot to do with embedded systems?"
    query = "Kateri predmeti obravnavajo vgrajene sisteme?"
    retriever = get_retriever(vectorstore, search_type)
    sim_results = retriever.invoke(query)

    print("=== Similarity Search Results ===")
    for i, doc in enumerate(sim_results):
        title = doc.metadata.get("title", "unknown")
        print(f"[{i+1}] Source: {title}")
        print(doc.page_content[:200])
        print()

    return sim_results

### THIS FUNCTION IS USED WHEN ACTUALLY RUNNING ###
# def load_model():
#     MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",          # NormalFloat4 — optimal for bell-curve weight distributions
#         bnb_4bit_use_double_quant=True,     # Quantise the quantisation constants too (saves ~0.4 bits/param)
#         bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for matrix multiplications
#     )

#     print("Loading model...")
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=bnb_config,
#         device_map="auto",          # Automatically distribute across available GPUs
#         trust_remote_code=True,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "left"   # Left padding for inference: generated tokens follow the last real token, not padding tokens
#     print("Model loaded.")

#     return model, tokenizer

### THIS FUNCTION IS FOR TESTING ###
def load_model():
    # Use the smallest instruct model that works with your chat template
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

    print("Loading tiny model on CPU (just for pipeline test)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,   # CPU only works with float32
        device_map="cpu",            # Force CPU – no GPU needed
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer

### THIS FUNCTION IS FOR EXSPERIMENTIG ###
def load_real_model():
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    
    print("Loading Phi-3-mini...")
    
    # Dodajte BitsAndBytes konfiguracijo TUKAJ, preden jo uporabite
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,  # Zdaj je definiran
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,  # Zdaj je definiran
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer

def make_rag_prompt(inputs: dict, tokenizer) -> str:

    # context = inputs['context']
    # if len(context) > 3000:  # Če je predolg
    #     context = context[:3000] + "... [context truncated]"

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

if __name__ == "__main__":

    # for pdf in input_path:
    #     name = pdf.stem + ".md"
    #     document_to_markdown(pdf, output_path, name)

    embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
    vectorstore = FAISS.load_local(
            "../vectorstore/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    # vectorstore = embed_documents()
    # vectorstore.save_local("../vectorstore/faiss_index")


    retriever = get_retriever(vectorstore, search_type="mmr")
    # results = get_relevant_chunks(vectorstore) # This is to test retrieval
    model, tokenizer = load_model()
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    ### THIS IS THE ACTUAL PIPELINE ###
    # hf_pipeline = pipeline(
    #     task="text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     return_full_text=False,
    #     max_new_tokens=2048,   
    #     do_sample=False,
    #     repetition_penalty=1.1,
    #     pad_token_id=tokenizer.eos_token_id,
    #     eos_token_id=[tokenizer.eos_token_id, eot_token_id]
    # )
    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        # Ne mešajte generation_config s parametri
        # Uporabite samo parametre ali pa generation_config, ne obojega
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
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

    #ask("Vgrajeni sistemi", rag_chain)


    chat_history: list[dict] = []   
    MAX_HISTORY = 10

    while True:
        try:
            question = input("\n Vprašanje: ").strip()

            if not question:
                continue

            # 1. Prevedi vprašanje SL -> EN
            question = preprocess_query(question)
            print(f"[Vprašanje]: {question}")

            # 2. Pošlji angleško vprašanje modelu
            answer = ask(question, rag_chain, chat_history)

            # 3. Prevedi odgovor EN -> SL
            print(f"\n💬 Odgovor: {answer}")

            # 4. V zgodovino shrani angleške verzije (model pričakuje angleščino)
            chat_history.append({"question": question, "answer": answer})
            if len(chat_history) > MAX_HISTORY:
                chat_history = chat_history[-MAX_HISTORY:]

        except KeyboardInterrupt:
            print("\n\n👋 Nasvidenje!")
            break
        except Exception as e:
            print(f"\n❌ Prišlo je do napake: {e}")
            print("Poskusite znova z drugim vprašanjem.")