import streamlit as st # aplicaciones web interactivas
import transformers # biblioteca desarrollada por Hugging Face
import pdfplumber #  biblioteca de Python que se utiliza para extraer texto y datos de archivos PDF
from bs4 import BeautifulSoup # biblioteca para analizar y manipular documentos HTML y XML.
import tempfile # crear y gestionar archivos temporales
import os # proporciona una interfaz para interactuar con el SO para manipular archivos
from sklearn.feature_extraction.text import TfidfVectorizer # se utiliza para calcular representaciones vectoriales TF-IDF de documentos de texto.
from sklearn.metrics.pairwise import cosine_similarity # calcular la similitud coseno entre vectores.

# Cargar modelo de la biblioteca Transformers
# transformers se utiliza para implementar y trabajar con modelos de lenguaje basados en la arquitectura de Transformer.
# Usamos el modelo BERT que es creado por Google para tareas de respuestas a preguntas
model = transformers.pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")


# Función para cargar y procesar documentos PDF
# Extraemos el texto del PDF y lo devolvemos como cadena de texto.
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


# Función para cargar y procesar documentos HTML
# Extraemos el texto del html y lo devuelve como una cadena de texto sin las etiquetas html
def extract_text_from_html(html_file):
    with open(html_file, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text()
    return text


# Función para dividir el texto en partes para mejorar su procesamiento
# Tiene una longitud maximo de 3000 caracteres
def split_text(text, max_length=3000):
    parts = []
    while len(text) > max_length:
        part, text = text[:max_length], text[max_length:]
        parts.append(part)
    parts.append(text)
    return parts


# Función para calcular embeddings de texto
# Tomamos una lista de textos y un vectorizador como entrada para generar las representaciones numericas
def calculate_text_embeddings(texts, vectorizer):
    embeddings = vectorizer.transform(texts)
    return embeddings


# Función para obtener la respuesta más similar utilizando embeddings

def get_most_similar_answer(question, document_parts, vectorizer):
    #almacenamos la funcion de arriba en question_embedding
    question_embedding = calculate_text_embeddings([question], vectorizer)
    #creamos una matriz de similitud para la pregunta y el fragmento del documento
    similarities = cosine_similarity(question_embedding, document_parts)
    #encontramos el indice del fragmento con mayor similutd a la pregunta
    #se almacena en most_similar_index
    most_similar_index = similarities.argmax()
    #me devuelve el index del fragmento mas similar a la pregunta.
    return most_similar_index


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
        # Inicializar un objeto TfidfVectorizer
        # se utiliza para calcular embeddings TF-IDF para las partes del documento.
        # TF-IDF evalua la importancia de una palabra en un documento en relación con una colección de documentos.
        vectorizer = TfidfVectorizer()

        #text_parts es una lista de fragmentos de texto.
        # Calcular embeddings para las partes del documento
        document_embeddings = vectorizer.fit_transform(text_parts)

        # Encontrar la parte del documento más similar a la pregunta
        most_similar_index = get_most_similar_answer(question, document_embeddings, vectorizer)

        # Obtener respuesta usando el modelo de Transformers para la parte más similar
        answer = model(question=question, context=text_parts[most_similar_index])
        st.write("Respuesta:", answer["answer"])
