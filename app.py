import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI College Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Whole page background */
.stApp{
    background: linear-gradient(135deg,#E3F2FD,#F8FBFF);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#D6EAF8;
}

/* Title */
.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#0D47A1;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:#555;
    margin-bottom:30px;
}

/* Input Box */
.stTextInput input{
    background:white !important;
    color:black !important;
    border-radius:15px !important;
    border:2px solid #42A5F5 !important;
    padding:12px !important;
}

/* Chat Box Common */
.chat-box{
    padding:18px;
    border-radius:18px;
    margin-bottom:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.15);
}

/* User Message */
.user-msg{
    background:#BBDEFB;
    color:black;
    border-left:7px solid #1565C0;
}

/* AI Message */
.bot-msg{
    background:#C8E6C9;
    color:black;
    border-left:7px solid #2E7D32;
}

/* Footer */
footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)
# ---------------- HEADER ----------------
st.markdown(
"""
<div class="title">
🎓 AI College Admission Assistant 🤖
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="subtitle">
Ask anything about Admissions • Placements • Fees • Courses • Faculty • Campus
</div>
""",
unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("📌 About Project")

    st.write("This chatbot uses:")

    st.write("✅ LangChain")
    st.write("✅ Groq API")
    st.write("✅ RAG Architecture")
    st.write("✅ FAISS Vector Database")
    st.write("✅ HuggingFace Embeddings")
    st.write("✅ Streamlit UI")

    st.divider()

    st.subheader("💡 Example Questions")

    st.write("• What courses are available?")
    st.write("• What is the admission process?")
    st.write("• Tell me about placements")
    st.write("• What are the fees?")
    st.write("• Who are the faculty members?")

# ---------------- LOAD PDF ----------------
pdf_path = "data/brochure.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

# ---------------- TEXT SPLITTER ----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(documents)

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- VECTOR DATABASE ----------------
vectorstore = FAISS.from_documents(texts, embeddings)

retriever = vectorstore.as_retriever()

# ---------------- LLM ----------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.5
)

# ---------------- QA CHAIN ----------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ---------------- CHAT HISTORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- USER INPUT ----------------
query = st.chat_input("💬 Ask Your Question")

# ---------------- GENERATE RESPONSE ----------------
if query:

    with st.spinner("🤖 Thinking..."):

        response = qa_chain.run(query)

    st.session_state.chat_history.append((query, response))

# ---------------- DISPLAY CHAT ----------------
for user_q, bot_a in reversed(st.session_state.chat_history):

    st.markdown(
        f"""
        <div class="chat-box user-msg">
        <b>👨‍🎓 You:</b><br>{user_q}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="chat-box bot-msg">
        <b>🤖 AI Assistant:</b><br>{bot_a}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------
st.divider()

st.caption("Built with ❤️ using LangChain + Groq + Streamlit")