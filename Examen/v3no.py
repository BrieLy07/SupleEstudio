import os
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Annoy
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
        loader = UnstructuredHTMLLoader(doc_name)
    else:
        loader = UnstructuredHTMLLoader(path)

    doc = loader.load()  # Cargar el documento

    db = Annoy.from_documents(doc, embedding=HuggingFaceEmbeddings())

    repo_id = "google/flan-t5-xxl"
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 100}
    )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='refine', retriever=db.as_retriever())

    result = qa.run(question)


def client():
    st.title('Gestionar Respuestas desde HTML con Hugging Face')
    uploader = st.file_uploader('Upload HTML', type='html')

    if uploader:
        with open(f'./{doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('HTML saved!!')

    question = st.text_input('Genera Respuesta',
                             placeholder='Da respuesta sobre tu HTML', disabled=not uploader)

    if st.button('Send Question'):
        if uploader:
            process_doc(
                path=doc_name,
                is_local=True,
                question=question,
            )
        else:
            st.info('Cargando HTML')
            process_doc()


if __name__ == '__main__':
    client()
