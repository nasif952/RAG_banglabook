import streamlit as st
from st_supabase_connection import SupabaseConnection
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain

# Load secrets directly from st.secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --------- 1. Connection check functions -----------
def check_supabase():
    try:
        supabase_conn = st.connection(
            "supabase",
            type=SupabaseConnection,
            url=SUPABASE_URL,
            key=SUPABASE_KEY,
        )
        # Try a minimal call: get client info
        _ = supabase_conn.client.auth.get_user()
        return True, "✅ Supabase Connected"
    except Exception as e:
        return False, f"❌ Supabase Error: {e}"

def check_openai():
    try:
        # Minimal OpenAI call
        openai.api_key = OPENAI_API_KEY
        openai.Model.list()  # will fail if key invalid
        return True, "✅ OpenAI Connected"
    except Exception as e:
        return False, f"❌ OpenAI Error: {e}"

# --------- 2. Check and display at top of app -----------
supabase_status, supabase_msg = check_supabase()
openai_status, openai_msg = check_openai()

with st.sidebar:
    st.markdown("### Connection Status")
    st.write(supabase_msg)
    st.write(openai_msg)
    if not (supabase_status and openai_status):
        st.warning("One or more connections failed. The app may not work.")

# Only proceed if both are connected
if not (supabase_status and openai_status):
    st.stop()

# --------- 3. Continue with app as normal -----------
# (Re-create connections to use in app logic)
supabase_conn = st.connection(
    "supabase",
    type=SupabaseConnection,
    url=SUPABASE_URL,
    key=SUPABASE_KEY,
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
)
vectorstore = SupabaseVectorStore(
    client=supabase_conn.client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

system_prompt = """তুমি একজন সহায়ক বাংলা সহকারী।
তুমি শুধুমাত্র নিচের তথ্যসূত্র দেখে উত্তর দেবে—যদি তথ্য না থাকে, দয়া করে বলো 'তথ্য নেই'।
তথ্যসূত্র:
{context}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.5, api_key=OPENAI_API_KEY)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

st.title("📚 বাংলা HSC26 RAG সহায়ক")
query = st.text_area("আপনার প্রশ্ন লিখুন:", height=100)

if st.button("উত্তর পান"):
    with st.spinner("উত্তর তৈরি হচ্ছে…"):
        try:
            res = rag_chain.invoke({"input": query})
            answer = res.get("output") or res.get("result") or "তথ্য নেই"
            st.success(answer)
            st.markdown("#### তথ্যসূত্র দেখুন:")
            for idx, doc in enumerate(res.get("source_documents", []), 1):
                st.write(f"Chunk {idx}: {doc.page_content[:200]}…")
        except Exception as e:
            st.error(f"❌ Error generating answer: {e}")
