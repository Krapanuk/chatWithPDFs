import os
from PyPDF2 import PdfReader
import streamlit as st

# Funktion zur Textextraktion aus PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text_per_page = [page.extract_text() for page in reader.pages]
        return text_per_page
    except Exception as e:
        st.error(f"Fehler beim Lesen der Datei {pdf_path}: {e}")
        return None

# Funktion zum Ermitteln der optimalen Chunk-Größe
def determine_chunk_size_per_page(text_per_page, min_chunk_size=50):
    chunk_sizes = []
    for page_text in text_per_page:
        word_count = len(page_text.split())
        chunk_size = max(min_chunk_size, word_count // 2)  # Default: Zwei Chunks pro Seite
        chunk_sizes.append(chunk_size)
    return chunk_sizes

# Funktion zum Chunken des Texts
def chunk_text(text_per_page, chunk_sizes):
    chunks = []
    for page_number, (page_text, chunk_size) in enumerate(zip(text_per_page, chunk_sizes), start=1):
        words = page_text.split()
        page_chunks = [
            ' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)
        ]
        for chunk_index, chunk in enumerate(page_chunks, start=1):
            chunks.append((page_number, chunk_index, chunk))
    return chunks

# PDFs verarbeiten und in Chunks aufteilen
def process_pdfs_in_folder(folder_path, min_chunk_size=50):
    all_chunks = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            st.write(f"**Verarbeite Datei:** {file_name}")
            text_per_page = extract_text_from_pdf(pdf_path)
            if text_per_page:
                chunk_sizes = determine_chunk_size_per_page(text_per_page, min_chunk_size)
                chunks = chunk_text(text_per_page, chunk_sizes)
                for page_number, chunk_index, chunk in chunks:
                    all_chunks.append((file_name, page_number, chunk_index, chunk))
    return all_chunks
