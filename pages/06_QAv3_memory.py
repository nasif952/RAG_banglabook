import streamlit as st
from st_supabase_connection import SupabaseConnection
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time

# Load secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="বাংলা HSC26 RAG সহায়ক", page_icon="📚")
st.title("📚 বাংলা HSC26 RAG সহায়ক")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Connection checks
@st.cache_resource
def check_connections():
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
    
    return check_supabase(), check_openai()

# Setup RAG chain
@st.cache_resource
def setup_rag_chain():
    try:
        supabase_conn = st.connection("supabase", type=SupabaseConnection,
                                      url=SUPABASE_URL, key=SUPABASE_KEY)
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        vectorstore = SupabaseVectorStore(client=supabase_conn.client, embedding=embeddings,
                                          table_name="documents", query_name="match_documents")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        system_prompt = """তুমি একজন সহায়ক বাংলা সহকারী।
তুমি শুধুমাত্র নিচের তথ্যসূত্র দেখে উত্তর দেবে—যদি তথ্য না থাকে, দয়া করে বলো 'তথ্য নেই'।

পূর্ববর্তী কথোপকথনের প্রসঙ্গ বিবেচনা করে উত্তর দাও।

তথ্যসূত্র:
{context}
"""
        
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, api_key=OPENAI_API_KEY)
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        
        return rag_chain, None
    except Exception as e:
        return None, str(e)

# Check connections
(sup_status, sup_msg), (open_status, open_msg) = check_connections()

# Sidebar
with st.sidebar:
    st.markdown("### 🔗 Connection Status")
    st.write(sup_msg)
    st.write(open_msg)
    
    st.markdown("### 💬 Chat Controls")
    st.write(f"📝 Messages: {len(st.session_state.messages)}")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat cleared!")
        time.sleep(0.5)
        st.rerun()
    
    # Export chat option
    if len(st.session_state.messages) > 0:
        chat_export = ""
        for msg in st.session_state.messages:
            role = "👤 আপনি" if msg["role"] == "user" else "🤖 সহায়ক"
            chat_export += f"{role}: {msg['content']}\n\n"
        
        st.download_button(
            "💾 Export Chat",
            data=chat_export,
            file_name=f"chat_history_{int(time.time())}.txt",
            mime="text/plain"
        )

# Check if connections are working
if not (sup_status and open_status):
    st.error("❌ Connection failed. Please check your configuration.")
    st.stop()

# Setup RAG chain if not already done
if st.session_state.rag_chain is None:
    with st.spinner("Setting up RAG system..."):
        rag_chain, error = setup_rag_chain()
        if rag_chain:
            st.session_state.rag_chain = rag_chain
            st.success("✅ RAG system ready!")
        else:
            st.error(f"❌ RAG setup failed: {error}")
            st.stop()

# Function to get conversation context
def get_conversation_context():
    if len(st.session_state.messages) == 0:
        return ""
    
    # Get last 4 messages for context (2 exchanges)
    recent_messages = st.session_state.messages[-4:]
    context = "\n\nসাম্প্রতিক কথোপকথন:\n"
    
    for msg in recent_messages:
        role = "ব্যবহারকারী" if msg["role"] == "user" else "সহায়ক"
        context += f"{role}: {msg['content']}\n"
    
    return context

# Display chat messages
st.markdown("### 💬 কথোপকথন")

# Create a container for chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 তথ্যসূত্র"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source}")

# Chat input
if prompt := st.chat_input("আপনার প্রশ্ন লিখুন..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("চিন্তা করছি..."):
            try:
                # Get conversation context
                conversation_context = get_conversation_context()
                
                # Create enhanced input with context
                enhanced_input = f"{prompt}{conversation_context}"
                
                # Get response from RAG
                result = st.session_state.rag_chain.invoke({"input": enhanced_input})
                response = result.get("answer", "তথ্য নেই")
                
                # Extract sources
                sources = []
                if "context" in result:
                    sources = [doc.page_content[:200] + "..." for doc in result["context"]]
                
                # Display response
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"দুঃখিত, একটি সমস্যা হয়েছে: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

# Quick action buttons
if len(st.session_state.messages) == 0:
    st.markdown("### 🚀 দ্রুত শুরু করুন")
    st.markdown("নিচের বিষয়গুলো সম্পর্কে জিজ্ঞাসা করতে পারেন:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = [
        ("📖", "বাংলা সাহিত্য", "বাংলা সাহিত্যের ইতিহাস সম্পর্কে বলুন"),
        ("🔬", "বিজ্ঞান", "পদার্থবিজ্ঞানের মূল নীতিগুলো কী?"),
        ("📊", "গণিত", "ক্যালকুলাসের মূল ধারণা ব্যাখ্যা করুন"),
        ("🌍", "ভূগোল", "বাংলাদেশের ভৌগোলিক অবস্থান সম্পর্কে বলুন")
    ]
    
    cols = [col1, col2, col3, col4]
    for i, (emoji, title, question) in enumerate(quick_questions):
        with cols[i]:
            if st.button(f"{emoji} {title}"):
                # Add the question as if user typed it
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    🚀 Powered by LangChain, Supabase, OpenAI, and Streamlit<br>
    💡 Tip: আপনি ধারাবাহিক প্রশ্ন করতে পারেন, সিস্টেম পূর্ববর্তী কথোপকথন মনে রাখবে
</div>
""", unsafe_allow_html=True)
