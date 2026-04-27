from flask import Flask, request, jsonify
from flask_cors import CORS

# Uvozi vse iz main.py
from main import (
    load_model,
    embed_documents,
    get_retriever,
    make_rag_prompt,
    format_docs,
    preprocess_query,
    SHORTCUTS
)

import torch
from functools import partial
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Dovoli klice iz frontend-a (localhost:3000 itd.)

# --- Globalne spremenljivke za modele ---
rag_chain = None
sl_en_tok = None
sl_en_mod = None
chat_history = []
MAX_HISTORY = 10

def initialize():
    """Naloži vse modele ob zagonu strežnika (enkrat)."""
    global rag_chain

    print("🔄 Nalagam embeddings in vectorstore...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    vectorstore = FAISS.load_local(
        "../vectorstore/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = get_retriever(vectorstore, search_type="mmr")

    print("🔄 Nalagam jezikovni model...")
    model, tokenizer = load_model()
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
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

    print("✅ Vsi modeli naloženi. Strežnik je pripravljen.")


@app.route("/api/model", methods=["POST"])
def model_endpoint():
    global chat_history

    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "Manjka polje 'messages'"}), 400

    # Vzemi zadnje sporočilo uporabnika iz zgodovine pogovora
    messages = data["messages"]
    user_messages = [m for m in messages if m["role"] == "user"]
    if not user_messages:
        return jsonify({"error": "Ni uporabniškega sporočila"}), 400

    question = user_messages[-1]["content"]

    try:
        # 1. Predobdelava in prevod SL -> EN
        question = preprocess_query(question)
        print(f"[Vprašanje]: {question}")

        # 2. RAG pipeline
        answer = rag_chain.invoke({
            "question": question,
            "history": chat_history
        })

        # 3. Shrani v zgodovino
        chat_history.append({"question": question, "answer": answer})
        if len(chat_history) > MAX_HISTORY:
            chat_history = chat_history[-MAX_HISTORY:]
        
        print(f"[Odgovor]: {answer}")
        return jsonify({"content": answer})

    except Exception as e:
        print(f"❌ Napaka: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_history():
    """Ponastavi zgodovino pogovora (opcijsko)."""
    global chat_history
    chat_history = []
    return jsonify({"status": "ok", "message": "Zgodovina počiščena."})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": rag_chain is not None})


if __name__ == "__main__":
    initialize()
    app.run(host="0.0.0.0", port=4000, debug=False)