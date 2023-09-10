#CLASIFICADOR DE TEXTOS

import streamlit as st
from transformers import pipeline
from pdf2image import convert_from_bytes  # Cambio en la importación
from bs4 import BeautifulSoup

st.title("Sistema de Clasificación de Documentos")

# Carga el modelo preentrenado de Zero-shot
model_name = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=model_name)

# Función para procesar archivos PDF
def process_pdf(file):
    pdf_bytes = file.read()  # Lee los bytes del archivo
    pages = convert_from_bytes(pdf_bytes)
    text = ""
    for page in pages:
        text += page.get_text()
    return text

# Función para procesar archivos HTML
def process_html(file):
    html_content = file.read()  # Lee el contenido del archivo
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text()
    return text

# Interfaz de usuario
uploaded_file = st.file_uploader("Cargar archivo PDF o HTML", type=["pdf", "html"])
if uploaded_file:
    st.write("Archivo cargado con éxito.")

    # Procesar el archivo
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        text = process_pdf(uploaded_file)
    elif file_extension == "html":
        text = process_html(uploaded_file)
    else:
        st.error("Formato de archivo no compatible. Debe ser PDF o HTML.")
        st.stop()

    # Clasificar el texto
    labels = ["Blog", "Noticias", "Articulo"]
    result = classifier(text, labels)

    # Mostrar la clase predicha
    st.subheader("Clase del documento:")
    st.write(result["labels"][0])
