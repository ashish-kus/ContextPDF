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
import hashlib

load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="Multi-PDF Chat", layout="wide")

# Initialize session state for API key
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False

# Sidebar contents
with st.sidebar:
    st.title("🤗💬 Multi-PDF Chat App")

    # API Key Configuration Section
    st.subheader("🔑 API Configuration")
    api_key_input = st.text_input(
        "Enter your Gemini API Key:",
        type="password",
        help="Get your API key from https://ai.google.dev/",
        key="api_key_input",
    )

    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.session_state.api_key_configured = True
        st.success("✅ API Key configured!")
    elif os.getenv("GOOGLE_API_KEY"):
        st.session_state.api_key_configured = True
        st.info("ℹ️ Using API key from environment")
    else:
        st.warning("⚠️ Please enter your Gemini API Key to proceed")

    st.markdown("---")

    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Google Generative AI](https://ai.google.dev/) LLM model
    
    Upload multiple PDFs and have a conversation about their content!
    """
    )
    add_vertical_space(5)
    st.write("Made by [Ashish Kushwaha](https://ashishkus.com)")


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def load_embeddings():
    """Load Google Generative AI embeddings."""
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def get_file_hash(file_content):
    """Generate a hash for file content to detect duplicates."""
    return hashlib.md5(file_content).hexdigest()


def create_vector_store(chunks, store_name):
    """Create and save FAISS vector store."""
    embeddings = load_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(store_name)
    return vector_store


def load_vector_store(store_name):
    """Load FAISS vector store."""
    embeddings = load_embeddings()
    return FAISS.load_local(
        store_name, embeddings, allow_dangerous_deserialization=True
    )


def merge_vector_stores(existing_store, new_store):
    """Merge two FAISS vector stores."""
    existing_store.merge_from(new_store)
    return existing_store


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


def create_rag_chain(vector_store):
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
If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    chain = (
        {
            "context": retriever | (lambda docs: format_docs(docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    # Check if API key is configured
    if not st.session_state.api_key_configured:
        st.header("🔑 API Key Required")
        st.warning(
            "Please enter your Gemini API Key in the sidebar to continue.\n\n"
            "**How to get your API Key:**\n"
            "1. Visit https://ai.google.dev/\n"
            "2. Click 'Get API Key'\n"
            "3. Create a new API key or use an existing one\n"
            "4. Copy and paste it in the sidebar"
        )
        return

    st.header("Chat with Multiple PDFs 📄")

    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    # Main store name
    main_store_name = "multi_pdf_store"

    # File uploader for multiple PDFs
    uploaded_files = st.file_uploader(
        "Upload your PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        # Process new files
        new_files_added = False
        for pdf_file in uploaded_files:
            file_hash = get_file_hash(pdf_file.read())
            pdf_file.seek(0)  # Reset file pointer

            if file_hash not in st.session_state.processed_files:
                st.session_state.processed_files.add(file_hash)
                new_files_added = True

                with st.spinner(f"Processing {pdf_file.name}..."):
                    chunks = process_pdf(pdf_file)

                    if st.session_state.vector_store is None:
                        # Create new vector store
                        st.session_state.vector_store = create_vector_store(
                            chunks, main_store_name
                        )
                    else:
                        # Merge with existing vector store
                        new_store = create_vector_store(chunks, f"temp_{file_hash}")
                        st.session_state.vector_store = merge_vector_stores(
                            st.session_state.vector_store, new_store
                        )
                        st.session_state.vector_store.save_local(main_store_name)

                st.success(f"✅ {pdf_file.name} processed and added")

        # Load existing vector store if not in memory
        if new_files_added is False and st.session_state.vector_store is None:
            if os.path.exists(main_store_name):
                with st.spinner("Loading embeddings from disk..."):
                    st.session_state.vector_store = load_vector_store(main_store_name)
                st.success("✅ Embeddings loaded from disk")

        # Display processed files
        st.subheader("📋 Processed PDFs:")
        st.write(f"Total files processed: {len(st.session_state.processed_files)}")

        # Chat interface
        st.subheader("💬 Chat")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        user_input = st.chat_input("Ask a question about your PDFs:")

        if user_input:
            # Add user message to history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            with st.chat_message("user"):
                st.write(user_input)

            # Generate response
            with st.spinner("Searching and generating answer..."):
                chain = create_rag_chain(st.session_state.vector_store)
                response = chain.invoke(user_input)

            # Add assistant message to history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )

            with st.chat_message("assistant"):
                st.write(response)

        # Sidebar controls
        with st.sidebar:
            add_vertical_space(5)
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

            if st.button("🔄 Reset All (Clear PDFs & Chat)"):
                st.session_state.vector_store = None
                st.session_state.chat_history = []
                st.session_state.processed_files = set()
                if os.path.exists(main_store_name):
                    import shutil

                    shutil.rmtree(main_store_name)
                st.rerun()


if __name__ == "__main__":
    main()
