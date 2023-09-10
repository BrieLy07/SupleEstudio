import streamlit as st
import transformers
import pdfplumber
from bs4 import BeautifulSoup
import tempfile
import os

# Cargar modelo de Transformers
model = transformers.pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")


# Función para cargar y procesar documentos PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


# Función para cargar y procesar documentos HTML
def extract_text_from_html(html_file):
    with open(html_file, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text()
    return text


# Función para dividir el texto en partes
def split_text(text, max_length=3000):
    parts = []
    while len(text) > max_length:
        part, text = text[:max_length], text[max_length:]
        parts.append(part)
    parts.append(text)
    return parts


# Interfaz de usuario con Streamlit
st.title("Sistema de Preguntas y Respuestas")

# Subir archivo PDF o HTML
file = st.file_uploader("Sube un documento PDF o HTML", type=["pdf", "html"])

if file:
    # Guardar temporalmente el archivo en el sistema de archivos
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    # Procesar el documento
    if file.type == "application/pdf":
        text = extract_text_from_pdf(temp_file_path)
    elif file.type == "text/html":
        text = extract_text_from_html(temp_file_path)

    # Eliminar el archivo temporal
    os.remove(temp_file_path)

    # Dividir el texto en partes
    text_parts = split_text(text)

    # Pregunta al usuario
    question = st.text_input("Haz una pregunta sobre el documento:")

    if st.button("Obtener Respuesta"):
        # Inicializar variables para la respuesta final
        final_answer = ""

        # Obtener respuesta usando el modelo de Transformers para cada parte del texto
        for part in text_parts:
            answer = model(question=question, context=part)
            final_answer += answer["answer"] + " "

        st.write("Respuesta:", final_answer)
