import streamlit as st
from st_supabase_connection import SupabaseConnection
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.title("📚 বাংলা HSC26 RAG সহায়ক")

# Connection checks
def check_supabase():
    try:
        sup = st.connection("supabase", type=SupabaseConnection,
                            url=SUPABASE_URL, key=SUPABASE_KEY)
        _ = sup.client.auth.get_user()
        return True, "✅ Supabase Connected"
    except Exception as e:
        return False, f"❌ Supabase Error: {e}"

def check_openai():
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        client.models.list()
        return True, "✅ OpenAI Connected"
    except Exception as e:
        return False, f"❌ OpenAI Error: {e}"

sup_status, sup_msg = check_supabase()
open_status, open_msg = check_openai()

with st.sidebar:
    st.markdown("### Connection Status")
    st.write(sup_msg)
    st.write(open_msg)
    if not (sup_status and open_status):
        st.warning("Connections failed.")

if not (sup_status and open_status):
    st.stop()

# Setup vector store and retriever
supabase_conn = st.connection("supabase", type=SupabaseConnection,
                              url=SUPABASE_URL, key=SUPABASE_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
vectorstore = SupabaseVectorStore(client=supabase_conn.client, embedding=embeddings,
                                  table_name="documents", query_name="match_documents")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

system_prompt = """তুমি একজন সহায়ক বাংলা সহকারী।
তুমি শুধুমাত্র নিচের তথ্যসূত্র দেখে উত্তর দেবে—যদি তথ্য না থাকে, দয়া করে বলো 'তথ্য নেই'।
তথ্যসূত্র:
{context}
"""
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.5, api_key=OPENAI_API_KEY)

qa_chain = create_stuff_documents_chain(llm, prompt)
rag = create_retrieval_chain(retriever, qa_chain)

query = st.text_area("আপনার প্রশ্ন লিখুন:", height=100)

if st.button("উত্তর পান / Get Answer") or query:
    with st.spinner("উত্তর খোঁজা হচ্ছে..."):
        result = rag.invoke({"input": query})
        answer = result.get("answer", "তথ্য নেই")
        st.markdown("**উত্তর:**")
        st.success(answer)
        with st.expander("তথ্যসূত্র (Retrieved Chunks)"):
            for i, doc in enumerate(result.get("context", []), 1):
                st.markdown(f"**Chunk {i}:** {doc.page_content}")

st.caption("Powered by LangChain, Supabase, OpenAI, and Streamlit")
