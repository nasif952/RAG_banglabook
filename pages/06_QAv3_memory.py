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

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

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
        
        # Enhanced system prompt with better context handling
        system_prompt = """তুমি একজন সহায়ক বাংলা সহকারী যিনি কথোপকথনের ধারাবাহিকতা বজায় রাখেন।

নিয়মাবলী:
1. নিচের তথ্যসূত্র ব্যবহার করে উত্তর দাও
2. যদি তথ্যসূত্রে সরাসরি উত্তর না থাকে, তবে বলো 'তথ্য নেই'
3. পূর্ববর্তী কথোপকথনের প্রসঙ্গ মনে রেখো
4. ব্যবহারকারী যদি পূর্বের প্রসঙ্গ উল্লেখ করে (যেমন "সে", "তার", "এটা", "ঐ ব্যক্তি"), তাহলে কথোপকথনের ইতিহাস অনুযায়ী বুঝে নাও
5. ব্যবহারকারী যদি আগের প্রশ্ন বা উত্তর সম্পর্কে জিজ্ঞাসা করে, তাহলে কথোপকথনের ইতিহাস থেকে উত্তর দাও

পূর্ববর্তী কথোপকথন:
{conversation_history}

বর্তমান তথ্যসূত্র:
{context}

গুরুত্বপূর্ণ: ব্যবহারকারীর প্রশ্নে যদি আগের কথোপকথনের রেফারেন্স থাকে, সেটা বিবেচনা করে উত্তর দাও।"""
        
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY)
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        
        return rag_chain, None
    except Exception as e:
        return None, str(e)

# Function to get detailed conversation context
def get_conversation_context():
    if len(st.session_state.messages) == 0:
        return ""
    
    # Get last 6 messages for better context (3 exchanges)
    recent_messages = st.session_state.messages[-6:]
    context = ""
    
    for i, msg in enumerate(recent_messages):
        role = "ব্যবহারকারী" if msg["role"] == "user" else "সহায়ক"
        context += f"{role}: {msg['content']}\n"
    
    return context

# Function to check if question refers to previous context
def needs_conversation_context(question):
    """Check if the question refers to previous conversation"""
    context_indicators = [
        'সে', 'তিনি', 'তার', 'ওর', 'এটা', 'ওটা', 'এই', 'ঐ', 
        'আগের', 'পূর্বের', 'আমি কি', 'আমি যা', 'কি জিজ্ঞেস',
        'valo', 'ভালো', 'she', 'তাহলে', 'why', 'কেন',
        'question', 'প্রশ্ন', 'ami ki', 'what i', 'so far'
    ]
    
    question_lower = question.lower()
    return any(indicator in question_lower for indicator in context_indicators)

# Check connections
(sup_status, sup_msg), (open_status, open_msg) = check_connections()

# Sidebar
with st.sidebar:
    st.markdown("### 🔗 Connection Status")
    st.write(sup_msg)
    st.write(open_msg)
    
    st.markdown("### 💬 Chat Controls")
    st.write(f"📝 Messages: {len(st.session_state.messages)}")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)
    
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

# Debug section
if st.session_state.debug_mode:
    with st.sidebar:
        st.markdown("### 🔍 Debug Info")
        if len(st.session_state.messages) > 0:
            st.write("**Last 3 Messages:**")
            for i, msg in enumerate(st.session_state.messages[-3:]):
                st.text(f"{i+1}. {msg['role']}: {msg['content'][:50]}...")
            
            st.write("**Current Context:**")
            context = get_conversation_context()
            st.text_area("Context", context, height=100, key="debug_context")

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
                
                # Check if question needs conversation context
                context_needed = needs_conversation_context(prompt)
                
                # Debug info
                if st.session_state.debug_mode:
                    st.write("🔍 **Debug Info:**")
                    st.write(f"- Context needed: {context_needed}")
                    st.write(f"- Conversation length: {len(st.session_state.messages)}")
                    if conversation_context:
                        st.write(f"- Context preview: {conversation_context[:100]}...")
                
                # Create the input for RAG
                if context_needed and conversation_context:
                    # For context-dependent questions, add more conversation history
                    rag_input = {
                        "input": prompt,
                        "conversation_history": conversation_context
                    }
                else:
                    # For independent questions, minimal context
                    rag_input = {
                        "input": prompt,
                        "conversation_history": conversation_context[-200:] if conversation_context else ""
                    }
                
                # Get response from RAG
                result = st.session_state.rag_chain.invoke(rag_input)
                response = result.get("answer", "তথ্য নেই")
                
                # Extract sources
                sources = []
                if "context" in result and result["context"]:
                    sources = [doc.page_content[:200] + "..." for doc in result["context"]]
                
                # If asking about previous questions and no good answer, try to answer from chat history
                if (response == "তথ্য নেই" and context_needed and 
                    any(word in prompt.lower() for word in ['ami ki', 'what i', 'question', 'প্রশ্ন', 'জিজ্ঞেস'])):
                    
                    # Extract questions from chat history
                    user_questions = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
                    if user_questions:
                        if "so far" in prompt.lower() or "কি কি" in prompt:
                            response = f"আপনি এ পর্যন্ত যে প্রশ্নগুলো করেছেন:\n\n"
                            for i, q in enumerate(user_questions[:-1], 1):  # Exclude current question
                                response += f"{i}. {q}\n"
                        else:
                            response = f"আপনার শেষ প্রশ্ন ছিল: '{user_questions[-2] if len(user_questions) > 1 else 'কোনো আগের প্রশ্ন নেই'}'"
                
                # Display response
                st.markdown(response)
                
                # Debug: Show what was sent to RAG
                if st.session_state.debug_mode:
                    st.write("🔧 **RAG Input:**")
                    st.json(rag_input)
                
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
        ("📖", "বাংলা সাহিত্য", "কল্যাণীর পিতার নাম কী?")
    ]
    
    cols = [col1, col2, col3, col4]
    for i, (emoji, title, question) in enumerate(quick_questions):
        with cols[i]:
            if st.button(f"{emoji} {title}"):
                # Add the question as if user typed it
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

# Tips section
if len(st.session_state.messages) > 0:
    st.markdown("""
<div style='
    background-color: #23272f;
    padding: 14px 16px;
    border-radius: 8px;
    margin-top: 12px;
    color: #f3f6fb;
    font-size: 15px;
    border-left: 4px solid #3b82f6;'
>
    <small>
    💡 <strong>টিপস:</strong><br>
    • "সে কেমন?" - আগের কথোপকথনের ব্যক্তি সম্পর্কে জিজ্ঞেস করুন<br>
    • "আমি কি কি প্রশ্ন করেছি?" - আপনার প্রশ্নের তালিকা দেখুন<br>
    • "এটা কী?" - আগের উল্লেখিত বিষয় সম্পর্কে আরও জানুন
    </small>
</div>
""", unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    🚀 Powered by LangChain, Supabase, OpenAI, and Streamlit<br>
    🧠 Enhanced with Conversation Memory & Context Awareness
</div>
""", unsafe_allow_html=True)
