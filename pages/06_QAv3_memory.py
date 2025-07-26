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

# Initialize memory in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Memory management functions
def add_to_memory(question, answer):
    """Add Q&A pair to conversation history"""
    st.session_state.conversation_history.append({
        "question": question,
        "answer": answer
    })
    # Keep only last 20 conversations
    if len(st.session_state.conversation_history) > 20:
        st.session_state.conversation_history.pop(0)

def get_conversation_context():
    """Get recent conversation context for the prompt"""
    if not st.session_state.conversation_history:
        return ""
    
    # Get last 3-5 conversations for context
    recent_history = st.session_state.conversation_history[-3:]
    context_text = "\n\nসাম্প্রতিক কথোপকথন:\n"
    
    for i, conv in enumerate(recent_history, 1):
        context_text += f"প্রশ্ন {i}: {conv['question']}\nউত্তর {i}: {conv['answer']}\n\n"
    
    return context_text

def clear_memory():
    """Clear conversation history"""
    st.session_state.conversation_history = []

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
    
    # Memory controls
    st.markdown("### 🧠 Memory")
    st.write(f"💬 Conversations: {len(st.session_state.conversation_history)}/20")
    
    if st.button("🗑️ Clear Memory"):
        clear_memory()
        st.success("Memory cleared!")
        st.rerun()
    
    # Show memory toggle
    show_history = st.checkbox("Show Conversation History", value=False)
    
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

# Enhanced system prompt with memory context
system_prompt = """তুমি একজন সহায়ক বাংলা সহকারী।
তুমি শুধুমাত্র নিচের তথ্যসূত্র দেখে উত্তর দেবে—যদি তথ্য না থাকে, দয়া করে বলো 'তথ্য নেই'।

তথ্যসূত্র:
{context}

{conversation_context}
"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, api_key=OPENAI_API_KEY)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag = create_retrieval_chain(retriever, qa_chain)

# Show conversation history if enabled
if show_history and st.session_state.conversation_history:
    st.markdown("### 💭 কথোপকথনের ইতিহাস")
    with st.expander("Previous Conversations", expanded=False):
        for i, conv in enumerate(reversed(st.session_state.conversation_history), 1):
            st.markdown(f"**প্রশ্ন {len(st.session_state.conversation_history)-i+1}:** {conv['question']}")
            st.markdown(f"**উত্তর:** {conv['answer']}")
            st.markdown("---")

# Main query interface
query = st.text_area("আপনার প্রশ্ন লিখুন:", height=100, key="main_query")

if st.button("উত্তর পান / Get Answer", type="primary") or query:
    if query.strip():
        with st.spinner("উত্তর খোঁজা হচ্ছে..."):
            # Get conversation context
            conversation_context = get_conversation_context()
            
            # Invoke RAG with memory context
            result = rag.invoke({
                "input": query,
                "conversation_context": conversation_context
            })
            
            answer = result.get("answer", "তথ্য নেই")
            
            # Display answer
            st.markdown("**উত্তর:**")
            st.success(answer)
            
            # Add to memory
            add_to_memory(query, answer)
            
            # Show source documents
            with st.expander("তথ্যসূত্র (Retrieved Chunks)"):
                for i, doc in enumerate(result.get("context", []), 1):
                    st.markdown(f"**Chunk {i}:** {doc.page_content}")
            
            # Clear the input
            st.rerun()
    else:
        st.warning("দয়া করে একটি প্রশ্ন লিখুন।")

# Quick question buttons (optional)
st.markdown("### 🚀 দ্রুত প্রশ্ন")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📖 বাংলা সাহিত্য"):
        st.session_state.main_query = "বাংলা সাহিত্য সম্পর্কে বলুন"
        st.rerun()

with col2:
    if st.button("🔬 বিজ্ঞান"):
        st.session_state.main_query = "বিজ্ঞান বিষয়ে তথ্য দিন"
        st.rerun()

with col3:
    if st.button("📊 গণিত"):
        st.session_state.main_query = "গণিত সমস্যার সমাধান"
        st.rerun()

st.caption("Powered by LangChain, Supabase, OpenAI, and Streamlit")
