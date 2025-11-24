import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="PDF Chat", layout="wide")

# Sidebar contents
with st.sidebar:
    st.title("🤗💬 LLM Chat App")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Google Generative AI](https://ai.google.dev/) LLM model
    
    Upload a PDF and ask questions about its content!
    """
    )


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def load_embeddings():
    """Load Google Generative AI embeddings."""
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def create_vector_store(chunks, store_name):
    """Create and save FAISS vector store using save_local method."""
    embeddings = load_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    # Use FAISS's built-in save_local method instead of pickle
    vector_store.save_local(store_name)
    return vector_store


def load_vector_store(store_name):
    """Load FAISS vector store using load_local method."""
    embeddings = load_embeddings()
    # Use FAISS's built-in load_local method
    return FAISS.load_local(
        store_name, embeddings, allow_dangerous_deserialization=True
    )


def process_pdf(pdf_file):
    """Process PDF file and return text chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pdf_docs = loader.load()

        # Combine all pages
        text = "\n".join(doc.page_content for doc in pdf_docs)

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    finally:
        os.unlink(tmp_path)


def create_rag_chain(docs):
    """Create RAG chain for question answering."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    prompt = PromptTemplate(
        template="""Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    chain = (
        {
            "context": lambda x: format_docs(docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    st.header("Chat with PDF 📄")

    # File uploader
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        store_name = pdf.name[:-4]

        # Initialize session state for vector store
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None

        # Check if embeddings exist on disk
        if os.path.exists(store_name):
            if st.session_state.vector_store is None:
                with st.spinner("Loading embeddings from disk..."):
                    st.session_state.vector_store = load_vector_store(store_name)
                st.success("✅ Embeddings loaded from disk")
        else:
            # Create new embeddings
            with st.spinner("Processing PDF and creating embeddings..."):
                chunks = process_pdf(pdf)
                st.session_state.vector_store = create_vector_store(chunks, store_name)
            st.success("✅ Embeddings created and saved")

        # Question input
        query = st.text_input("Ask a question about your PDF:")

        if query:
            with st.spinner("Searching and generating answer..."):
                docs = st.session_state.vector_store.similarity_search(query=query, k=3)
                chain = create_rag_chain(docs)
                response = chain.invoke(query)

            st.subheader("Answer:")
            st.write(response)

            # Optional: Show source documents
            with st.expander("📎 View source documents"):
                for i, doc in enumerate(docs, 1):
                    st.write(f"**Source {i}:**")
                    st.write(
                        doc.page_content[:500] + "..."
                        if len(doc.page_content) > 500
                        else doc.page_content
                    )


if __name__ == "__main__":
    main()
