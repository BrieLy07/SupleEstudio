import streamlit as st
from transformers import pipeline
from pdf2image import convert_from_bytes
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # Utilizaremos LSA para resumir

st.title("Sistema de Clasificación de Documentos con Resumen")

# Carga el modelo preentrenado de Zero-shot
model_name = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=model_name)


# Función para procesar archivos PDF
def process_pdf(file):
    pdf_bytes = file.read()
    pages = convert_from_bytes(pdf_bytes)
    text = ""
    for page in pages:
        text += page.get_text()
    return text


# Función para procesar archivos HTML
def process_html(file):
    html_content = file.read()
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text()
    return text


# Función para dividir el texto en secciones
def split_text(text):
    # Dividir el texto en secciones (aquí se usa un criterio simple)
    sections = text.split("\n\n")  # Puedes ajustar el criterio de división según tus documentos
    return sections


# Función para generar un resumen de una sección
def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=3)  # Puedes ajustar el número de oraciones en el resumen
    return " ".join([str(sentence) for sentence in summary])


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

    # Dividir el texto en secciones
    sections = split_text(text)

    st.subheader("Generación de Resúmenes:")
    for section in sections:
        if section.strip():  # Ignorar secciones vacías
            #st.write(
               # f"Sección original: {section[:100]}...")  # Muestra los primeros 100 caracteres de la sección original
            # Genera un resumen de la sección
            summary = generate_summary(section)
            st.write(f"Resumen de la sección: {summary}")
