# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
import asyncio
import tempfile
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import time
from langchain_community.document_loaders import PyPDFLoader

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
RETRIEVER_K = 4
DEFAULT_SYSTEM_MESSAGE = """
You are Document RAG Assistant 📄🤖. 
Your role is to help users understand and explore the content of uploaded documents.

Follow these rules:
1. Always prioritize the document context when answering questions.
2. If the answer is not in the document, clearly say you don't know.
3. Keep responses friendly, clear, and concise.
"""

# Load environment variables
load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # if "current_video_id" not in st.session_state:
    # st.session_state.current_video_id = None


def configure_page():
    st.set_page_config(
        page_title="Document RAG Assistant",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for clean modern design
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        min-height: 100vh;
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #666;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .chat-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .stChatMessage {
        background: #f8f9fa !important;
        border-radius: 15px !important;
        margin: 1rem 0 !important;
        padding: 1.5rem !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        border: 1px solid #e9ecef !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 60, 114, 0.4);
    }
    
    .upload-section {
        background: #ffffff;
        border-radius: 15px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        border: 2px dashed #1e3c72;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #2a5298;
        background: #f8f9fa;
    }
    
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 3px 15px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .stFileUploader {
        background: transparent;
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-success {
        background: #28a745;
        color: white;
    }
    
    .status-warning {
        background: #ffc107;
        color: #212529;
    }
    
    .sidebar-header {
        background: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 3px 15px rgba(0,0,0,0.1);
    }
    
    .animated-icon {
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🚀 Document RAG Assistant</div>
        <div class="hero-subtitle">Transform any document into an intelligent conversation</div>
    </div>
    """, unsafe_allow_html=True)


def create_feature_cards():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; margin-bottom: 1rem; color: #1e3c72;">📄</div>
            <h3 style="color: #333; margin-bottom: 0.5rem; font-weight: 600;">Multi-Format</h3>
            <p style="color: #666; font-size: 0.95rem;">Support for PDF and TXT files</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; margin-bottom: 1rem; color: #1e3c72;">🤖</div>
            <h3 style="color: #333; margin-bottom: 0.5rem; font-weight: 600;">AI Powered</h3>
            <p style="color: #666; font-size: 0.95rem;">Google Gemini integration</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; margin-bottom: 1rem; color: #1e3c72;">⚡</div>
            <h3 style="color: #333; margin-bottom: 0.5rem; font-weight: 600;">Fast Search</h3>
            <p style="color: #666; font-size: 0.95rem;">Vector-based similarity search</p>
        </div>
        """, unsafe_allow_html=True)


def handle_new_document_button():
    if st.sidebar.button("🔄 New Document", use_container_width=True):
        # Clear document-related session state
        if "retriever" in st.session_state:
            del st.session_state["retriever"]
        if "document_name" in st.session_state:
            del st.session_state["document_name"]

        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
        st.success("🔄 Ready for new document!")
        time.sleep(1)
        st.rerun()


def handle_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2 style="color: #1e3c72; margin: 0; font-weight: 600;">🔧 Configuration</h2>
        </div>
        """, unsafe_allow_html=True)

        api_key = st.text_input(
            "🔐 Google Gemini API Key",
            type="password",
            placeholder="Enter your API key...",
            help="Your key is kept only in your current browser session.",
            value=st.session_state.get("api_key", ""),
        )
        
        if api_key:
            st.session_state.api_key = api_key
            if len(api_key) < 20:
                st.markdown('<div class="status-badge status-warning">⚠️ API key looks too short</div>', unsafe_allow_html=True)
            elif not api_key.startswith("AIza"):
                st.markdown('<div class="status-badge status-warning">⚠️ Invalid Google API key format</div>', unsafe_allow_html=True)
            else:
                os.environ["GOOGLE_API_KEY"] = api_key
                st.markdown('<div class="status-badge status-success">✅ API key validated</div>', unsafe_allow_html=True)
        else:
            st.info("💡 Enter your API key to start")

        st.markdown("---")

        selected_model = st.selectbox(
            "🤖 AI Model",
            [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ],
            index=0,
            help="Choose the Gemini model for generation",
        )

        st.session_state.model = selected_model

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <h3 style="color: #1e3c72; font-weight: 600;">💬 Chat Controls</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
                st.rerun()

        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Reset complete!")

        handle_new_document_button()

        st.markdown("---")
        
        # Session metrics with modern cards
        message_count = len(st.session_state.messages) - 1
        document_processed = "retriever" in st.session_state and st.session_state.get("retriever") is not None

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.5rem; color: #1e3c72;">💬</div>
                <div style="color: #333; font-weight: 600; font-size: 1.2rem;">{message_count}</div>
                <div style="color: #666; font-size: 0.9rem;">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status_icon = "✅" if document_processed else "❌"
            status_color = "#28a745" if document_processed else "#dc3545"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.5rem; color: {status_color};">{status_icon}</div>
                <div style="color: #333; font-weight: 600; font-size: 1.2rem;">{'Ready' if document_processed else 'None'}</div>
                <div style="color: #666; font-size: 0.9rem;">Document</div>
            </div>
            """, unsafe_allow_html=True)

        if message_count > 0:
            st.markdown("---")
            chat_text = ""
            for msg in st.session_state.messages[1:]:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                chat_text += f"{role}: {msg.content}\n\n"

            st.download_button(
                "📥 Export Chat",
                chat_text,
                f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
                use_container_width=True,
            )

    return selected_model, st.session_state.get("api_key")


def handle_document_upload():
    st.markdown("""
    <div class="upload-section">
        <div class="animated-icon" style="font-size: 3rem; margin-bottom: 1rem; color: #1e3c72;">📁</div>
        <h3 style="color: #1e3c72; margin-bottom: 1rem; font-weight: 600;">Upload Your Document</h3>
        <p style="color: #666;">Drag and drop or click to select PDF or TXT files</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        help="Upload a PDF or text file to chat with",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="background: #ffffff; border: 2px solid #28a745; border-radius: 15px; padding: 1.5rem; text-align: center; margin: 1rem 0; box-shadow: 0 5px 15px rgba(40, 167, 69, 0.2);">
                <div style="color: #28a745; font-size: 1.5rem; margin-bottom: 0.5rem;">✅ File Ready</div>
                <div style="color: #333; font-weight: 600; font-size: 1.1rem;">{uploaded_file.name}</div>
                <div style="color: #666; font-size: 0.9rem;">{uploaded_file.size:,} bytes</div>
            </div>
            """, unsafe_allow_html=True)
    
    return uploaded_file

def handle_document_processing(uploaded_file=""):
    if uploaded_file:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                user_api_key = st.session_state.get("api_key", "")
                if not user_api_key:
                    st.error("❌ Please enter your Google Gemini API key in the sidebar first!")
                    return
                
                # Modern processing UI
                st.markdown("""
                <div style="background: #ffffff; border-radius: 15px; padding: 2rem; margin: 2rem 0; text-align: center; box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
                    <div style="font-size: 2rem; margin-bottom: 1rem; color: #1e3c72;">⚡</div>
                    <h3 style="color: #1e3c72; font-weight: 600;">Processing Your Document</h3>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.markdown("<div style='text-align: center; color: #1e3c72; font-weight: 600;'>🔄 Saving document...</div>", unsafe_allow_html=True)
                    progress_bar.progress(25)

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    status_text.markdown("<div style='text-align: center; color: #1e3c72; font-weight: 600;'>📄 Loading content...</div>", unsafe_allow_html=True)
                    progress_bar.progress(50)

                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_file_path)
                    else:
                        loader = TextLoader(tmp_file_path)

                    documents = loader.load()

                    status_text.markdown("<div style='text-align: center; color: #1e3c72; font-weight: 600;'>✂️ Creating chunks...</div>", unsafe_allow_html=True)
                    progress_bar.progress(75)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )
                    chunks = splitter.split_documents(documents)

                    status_text.markdown("<div style='text-align: center; color: #1e3c72; font-weight: 600;'>🧠 Building AI index...</div>", unsafe_allow_html=True)
                    progress_bar.progress(100)
                    
                    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    retriever = vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
                    )

                    st.session_state["retriever"] = retriever
                    st.session_state["document_name"] = uploaded_file.name

                    os.unlink(tmp_file_path)
                    progress_bar.empty()
                    status_text.empty()

                    st.markdown("""
                    <div style="background: #28a745; border-radius: 15px; padding: 2rem; margin: 2rem 0; text-align: center; box-shadow: 0 5px 20px rgba(40, 167, 69, 0.3);">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">🎉</div>
                        <h3 style="color: white; margin: 0; font-weight: 600;">Document Ready!</h3>
                        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Start asking questions below</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    time.sleep(2)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


@st.cache_resource()
def get_chat_model(model_name: str, api_key_keyed_for_cache: str | None):
    # api_key_keyed_for_cache is unused except for cache key isolation across different keys
    return ChatGoogleGenerativeAI(model=model_name)


def display_chat_messages():
    if len(st.session_state.messages) > 1:
        st.markdown("""
        <div class="chat-container">
            <h3 style="text-align: center; color: #333; margin-bottom: 2rem;">💬 Conversation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for message in st.session_state.messages[1:]:
            if isinstance(message, HumanMessage):
                with st.chat_message("user", avatar="👤"):
                    st.write(message.content)

            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message.content)


def handle_user_input(chat_model, input_disabled: bool = False):
    if prompt := st.chat_input(
        "Ask a question about the document...", disabled=input_disabled
    ):
        if not prompt.strip():
            st.warning("Please type a message before sending!")
            return

        st.session_state.messages.append(HumanMessage(content=prompt))

        prompt_template = PromptTemplate(
            template="""Based on this document content:

            {context}

            Question: {question}""",
            input_variables=["context", "question"],
        )

        with st.chat_message("user"):
            st.write(prompt)

        retriever = st.session_state.get("retriever")
        if not retriever:
            with st.chat_message("assistant"):
                error_msg = (
                    "❌ Please process a document first to enable question answering."
                )
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))
            return
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing document content..."):
                try:
                    retrieved_docs = retriever.invoke(prompt)
                    if not retrieved_docs:
                        no_context_msg = "🤷‍♂️ I couldn't find relevant information in the document for your question."
                        st.warning(no_context_msg)
                        st.session_state.messages.append(
                            AIMessage(content=no_context_msg)
                        )
                        return
                    parallel_chain = RunnableParallel(
                        {
                            "context": retriever | RunnableLambda(format_docs),
                            "question": RunnablePassthrough(),
                        }
                    )
                    parser = StrOutputParser()
                    main_chain = parallel_chain | prompt_template | chat_model | parser

                    message_placeholder = st.empty()
                    full_response = ""

                    # Stream the response using stream method (synchronous)
                    for chunk in main_chain.stream(prompt):
                        if chunk and chunk.strip():
                            full_response += chunk
                            message_placeholder.markdown(
                                full_response + "▌"
                            )  # Cursor indicator

                    # Remove cursor and display final response
                    if full_response and full_response.strip():
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append(
                            AIMessage(content=full_response)
                        )
                    else:
                        error_msg = (
                            "🚫 No response received. Please try a different model."
                        )
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append(AIMessage(content=error_msg))

                    # Rerun to refresh the UI after streaming
                    st.rerun()

                except Exception as e:
                    error_message = str(e).lower()
                    if "not found" in error_message or "invalid" in error_message:
                        error_msg = "❌ This model is not available. Please select a different model."
                    elif "quota" in error_message or "limit" in error_message:
                        error_msg = "📊 API quota exceeded. Please try again later or use a different model."
                    elif "timeout" in error_message:
                        error_msg = (
                            "⏱️ Request timed out. Try a different model or try again."
                        )
                    else:
                        error_msg = f"❌ An error occurred. Try selecting different model or check your api key:("

                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))
            # st.rerun()


# Main App Flow
init_session_state()
configure_page()

# Feature cards
create_feature_cards()

# Sidebar
selected_model, user_api_key = handle_sidebar()

# Document upload section
uploaded_file = handle_document_upload()

# Document processing
handle_document_processing(uploaded_file)

# Chat model setup
chat_model = None
if user_api_key:
    os.environ["GOOGLE_API_KEY"] = user_api_key
    chat_model = get_chat_model(selected_model, user_api_key)

# Chat interface
display_chat_messages()

# Input handling
if chat_model is None:
    st.markdown("""
    <div style="background: #ffffff; border: 2px solid #ffc107; border-radius: 15px; padding: 2rem; margin: 2rem 0; text-align: center; box-shadow: 0 5px 20px rgba(255, 193, 7, 0.2);">
        <div style="font-size: 2rem; margin-bottom: 1rem; color: #ffc107;">🔑</div>
        <h3 style="color: #1e3c72; font-weight: 600;">API Key Required</h3>
        <p style="color: #666;">Please enter your Google Gemini API key in the sidebar to start chatting</p>
    </div>
    """, unsafe_allow_html=True)

handle_user_input(chat_model, input_disabled=(chat_model is None))
