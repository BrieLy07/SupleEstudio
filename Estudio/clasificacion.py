import os
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Annoy
from transformers import pipeline
import streamlit as st

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

doc_name = 'doc.html'  # Change the document name to 'doc.html' for HTML loading


def process_doc(
        path: str = 'Documento-de-examen-Grupo1.html',
        is_local: bool = False,
        question: str = 'Cuáles son los autores del HTML?',
        zero_shot_labels: list = None
):
    if not is_local:
        os.system(f'curl -o {doc_name} {path}')
        loader = UnstructuredHTMLLoader(doc_name)  # Use HTMLLoader for remote HTML files
    else:
        loader = UnstructuredHTMLLoader(path)  # Use HTMLLoader for local HTML files

    doc = loader.load()  # Load the HTML document

    db = Annoy.from_documents(doc, embedding=HuggingFaceEmbeddings())

    repo_id = "google/flan-t5-xxl"
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 100}
    )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='refine', retriever=db.as_retriever())

    result = qa.run(question)

    if zero_shot_labels:
        # Realizar la clasificación zero-shot
        classifier = pipeline("zero-shot-classification")
        classification_result = classifier(result, zero_shot_labels)

        # Imprimir la respuesta y los porcentajes de clasificación
        st.write('Answer:', result)
        st.write('Zero-Shot Classification:')
        for label, score in zip(classification_result['labels'], classification_result['scores']):
            st.write(f'{label}: {score}')
    else:
        st.write('Answer:', result)


def client():
    st.title('CLASIFICACION DE DOCUMENTOS HTML')
    uploader = st.file_uploader('Upload HTML', type='html')  # Update the file uploader label

    if uploader:
        with open(f'./{doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('HTML saved!!')

    zero_shot_labels = st.text_input('Zero-Shot Labels (comma-separated)',
                                     placeholder='Etiqueta 1, Etiqueta 2, Etiqueta 3')

    if st.button('Send Question'):
        if uploader:
            process_doc(
                path=doc_name,
                is_local=True,
                zero_shot_labels=[label.strip() for label in zero_shot_labels.split(',')]
            )
        else:
            st.info('Cargando HTML')
            process_doc()


if __name__ == '__main__':
    client()
