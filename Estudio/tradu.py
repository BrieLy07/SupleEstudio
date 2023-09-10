import streamlit as st
from transformers import pipeline
from pdf2image import convert_from_bytes
from bs4 import BeautifulSoup
from translate import Translator  # Importa la biblioteca translate

st.title("Sistema de Clasificación de Documentos con Traducción")

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


# Función para traducir texto de inglés a español
def translate_to_spanish(text):
    translator = Translator(to_lang="es")
    translation = translator.translate(text)
    return translation


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

    # Clasificar cada sección y traducirla a español si está en inglés
    labels = ["Blog", "Noticias", "Articulo"]
    st.subheader("Clasificación de Secciones:")
    for section in sections:
        if section.strip():  # Ignorar secciones vacías
            # Verifica si la sección está en inglés
            if "english" in classifier(section, labels)["labels"][0].lower():
                translated_section = translate_to_spanish(section)
            else:
                translated_section = section

            st.write(
                f"Sección original: {section[:100]}...")  # Muestra los primeros 100 caracteres de la sección original
            st.write(
                f"Sección traducida: {translated_section[:100]}...")  # Muestra los primeros 100 caracteres de la sección traducida
