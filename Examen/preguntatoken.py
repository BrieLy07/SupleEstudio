import os
import streamlit as st
import transformers
from bs4 import BeautifulSoup

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

doc_name = 'doc.html'


def load_document(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()


def split_html_into_sections(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    sections = [section.get_text() for section in soup.find_all('section')]
    return sections


def process_doc(
        path: str = 'Documento-de-examen-Grupo1.html',
        is_local: bool = False,
        question: str = 'Cuáles son los autores del HTML?',
):
    if not is_local:
        os.system(f'curl -o {doc_name} {path}')
        document = load_document(doc_name)
    else:
        document = load_document(path)

    if document is None:
        return "Error: El documento no se pudo cargar."

    # Divide el documento en secciones
    sections = split_html_into_sections(document)

    best_answer = None
    best_score = 0

    qa_model = transformers.pipeline("question-answering",
                                     model="bert-large-uncased-whole-word-masking-finetuned-squad")

    for section in sections:
        result = qa_model(question=question, context=section)

        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]

    return best_answer


def client():
    st.title('Gestionar Respuestas desde HTML con Hugging Face')
    uploader = st.file_uploader('Subir HTML', type='html')

    if uploader:
        with open(f'./{doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('HTML guardado.')

    question = st.text_input('Generar Respuesta',
                             placeholder='Haz una pregunta sobre el HTML', disabled=not uploader)

    if st.button('Enviar Pregunta'):
        if uploader:
            best_answer = process_doc(
                path=doc_name,
                is_local=True,
                question=question,
            )
            st.write("Respuesta:", best_answer)
        else:
            st.info('Cargando HTML')
            best_answer = process_doc(
                question=question,
            )
            st.write("Respuesta:", best_answer)


if __name__ == '__main__':
    client()
