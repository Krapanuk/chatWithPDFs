import os
import json
import requests
import hashlib
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from deep_translator import GoogleTranslator
import streamlit as st
from readPDFs import process_pdfs_in_folder
import pickle

INDEX_FILE = "faiss_index.pkl"
MODEL_FILE = "faiss_model.pkl"
CHUNKS_FILE = "faiss_chunks.pkl"

# Funktion zum Speichern von Index, Modell und Chunks
def save_index_model_and_chunks(index, model, chunks):
    with open(INDEX_FILE, "wb") as f_index:
        pickle.dump(index, f_index)
    with open(MODEL_FILE, "wb") as f_model:
        pickle.dump(model, f_model)
    with open(CHUNKS_FILE, "wb") as f_chunks:
        pickle.dump(chunks, f_chunks)

# Funktion zum Laden von Index, Modell und Chunks
def load_index_model_and_chunks():
    if os.path.exists(INDEX_FILE) and os.path.exists(MODEL_FILE) and os.path.exists(CHUNKS_FILE):
        with open(INDEX_FILE, "rb") as f_index:
            index = pickle.load(f_index)
        with open(MODEL_FILE, "rb") as f_model:
            model = pickle.load(f_model)
        with open(CHUNKS_FILE, "rb") as f_chunks:
            chunks = pickle.load(f_chunks)
        return index, model, chunks
    return None, None, None

HASH_FILE = "pdf_hash.txt"

# Funktion zum Berechnen eines Hash-Werts für den aktuellen Zustand der PDFs
def calculate_folder_hash(folder_path):
    hash_md5 = hashlib.md5()
    if not os.path.exists(folder_path):
        return None
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Funktion zum Laden des gespeicherten Hash-Werts
def load_last_hash():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return f.read().strip()
    return None

# Funktion zum Speichern des aktuellen Hash-Werts
def save_current_hash(current_hash):
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)

# Übersetzung von Text
def translate_text(text, target_lang="en"):
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        st.error(f"Fehler bei der Übersetzung: {e}")
        return text

# FAISS-Index erstellen
def create_faiss_index(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode([chunk[3] for chunk in chunks], convert_to_tensor=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, model

# Anfrage an Ollama API
def query_ollama(model: str, prompt: str):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "system": "Du bist ein Dokumentenanalyse-Experte. Antworte in Deutscher Sprache präzise mit klaren Quellenangaben."
    }

    response_text = ""
    try:
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            response_text += data["response"]
                        if data.get("done"):
                            break
                # Entferne <think>-Inhalte
                response_text = remove_think_tags(response_text)
                return response_text.strip()
            else:
                st.error(f"Fehler bei der API: {response.status_code}, {response.text}")
    except Exception as e:
        st.error(f"Fehler bei der Anfrage: {e}")
        return None

# Funktion zum Entfernen von <think>-Inhalten
def remove_think_tags(response_text):
    while "<think>" in response_text and "</think>" in response_text:
        start_idx = response_text.index("<think>")
        end_idx = response_text.index("</think>") + len("</think>")
        response_text = response_text[:start_idx] + response_text[end_idx:]
    return response_text

# Streamlit-App
def main():
    st.title("PDF Chat mit Ollama")

    # PDF-Verzeichnis
    pdf_folder = "pdf"

    # Berechne den aktuellen Hash des PDF-Ordners
    current_hash = calculate_folder_hash(pdf_folder)
    last_hash = load_last_hash()

    # Session-State initialisieren, falls nicht vorhanden
    if "chunks" not in st.session_state or "index" not in st.session_state or "model" not in st.session_state:
        index, model, chunks = load_index_model_and_chunks()
        st.session_state["index"] = index
        st.session_state["model"] = model
        st.session_state["chunks"] = chunks
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Debugging: Zustand von Session-State anzeigen
    st.write("**Debugging-Status:**")
    st.write(f"Letzter Hash: {last_hash}")
    st.write(f"Aktueller Hash: {current_hash}")
    st.write(f"Index vorhanden: {st.session_state['index'] is not None}")
    st.write(f"Modell vorhanden: {st.session_state['model'] is not None}")
    st.write(f"Chunks vorhanden: {st.session_state['chunks'] is not None}")

    # Überprüfen, ob die PDFs geändert wurden oder der FAISS-Index fehlt
    index_missing = st.session_state["index"] is None or st.session_state["model"] is None
    if last_hash != current_hash or index_missing:
        if last_hash != current_hash:
            st.write("**Änderungen erkannt: Verarbeite PDFs und erstelle Index...**")
        else:
            st.write("**Index oder Modell fehlen: Erstelle Index...**")

        # PDFs verarbeiten und Index erstellen
        chunks = process_pdfs_in_folder(pdf_folder) if os.path.exists(pdf_folder) else []
        if chunks:
            st.session_state["chunks"] = chunks

            st.write("**FAISS-Ansatz ausgewählt**")
            index, model = create_faiss_index(chunks)
            st.session_state["index"] = index
            st.session_state["model"] = model

            save_current_hash(current_hash)
            save_index_model_and_chunks(st.session_state["index"], st.session_state["model"], chunks)
            st.write("**Index, Modell und Chunks erfolgreich erstellt und gespeichert.**")
        else:
            st.warning("Keine validen PDFs oder Module verfügbar. Der Chat ist trotzdem aktiv.")
            st.session_state["chunks"] = []
            st.session_state["index"] = None
            st.session_state["model"] = SentenceTransformer('all-MiniLM-L6-v2')  # Fallback-Modell

    else:
        st.write("**Keine Änderungen an den PDFs erkannt. Verwende bestehenden Index.**")

    # Chat
    st.header("Chat mit deinen Dokumenten")

    user_question = st.text_input("Frage stellen:")
    relevant_chunks = []  # Default-Wert für den Fall, dass keine Frage gestellt wird

    # Prüfen, ob Chunks existieren
    if not st.session_state.get("chunks"):
        st.warning("Es sind keine verarbeiteten Chunks vorhanden. Der Chat funktioniert trotzdem.")

    if user_question:
        # Chatverlauf hinzufügen
        st.session_state["chat_history"].append({"role": "user", "content": user_question})

        # Übersetze die Frage
        translated_question = translate_text(user_question, target_lang="en")

        if st.session_state["index"] and st.session_state["chunks"]:
            # Nutze FAISS für die Suche nach relevanten Chunks
            question_embedding = st.session_state["model"].encode([translated_question], convert_to_tensor=False)
            distances, indices = st.session_state["index"].search(np.array(question_embedding), k=5)
            relevant_chunks = [st.session_state["chunks"][idx] for idx in indices[0].tolist()]

            # Kontext für die Anfrage erstellen
            context = "\n\n".join(
                [f"(Dokument: {chunk[0]}, Seite: {chunk[1]}, Chunk: {chunk[2]}) {chunk[3]}" for chunk in relevant_chunks]
            )
        else:
            context = "Keine relevanten Daten verfügbar."

        ollama_prompt = f"Hier sind relevante Textausschnitte:\n{context}\n\nFrage: {translated_question}"

        with st.spinner("Frage an Ollama..."):
            response = query_ollama("deepseek-r1:14b", ollama_prompt)

        # Speichere und zeige die Antwort
        if response:
            st.session_state["chat_history"].append({"role": "assistant", "content": response})

    # Chatverlauf anzeigen
    st.write("### Chatverlauf")
    for message in st.session_state["chat_history"]:
        if message["role"] == "user":
            st.write(f"**Du:** {message['content']}")
        else:
            st.write(f"**Ollama:** {message['content']}")

    # Quellenangaben anzeigen
    if relevant_chunks:
        st.write("### Verwendete Quellen")
        for chunk in relevant_chunks:
            st.write(f"Dokument: {chunk[0]}, Seite: {chunk[1]}, Chunk: {chunk[2]}")

if __name__ == "__main__":
    main()