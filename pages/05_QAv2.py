import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import requests
from io import StringIO
from st_supabase_connection import SupabaseConnection
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.evaluation.qa.eval_chain import QAEvalChain
from sklearn.metrics.pairwise import cosine_similarity

# -- Load secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

RAW_URL = "https://raw.githubusercontent.com/nasif952/RAG_banglabook/main/qa_gold.csv"
st.title("🤖 Batch QA Generator + Evaluator (Bengali)")

# -- Persistent storage for loaded dataframe
if "qa_df" not in st.session_state:
    st.session_state.qa_df = None

### === Data Source Tabs ===
tab1, tab2 = st.tabs(["🔗 GitHub Preloaded", "📤 Upload Your Own"])

with tab1:
    st.markdown("#### GitHub (qa_gold.csv)")
    if st.button("📥 Load qa_gold.csv from GitHub"):
        try:
            resp = requests.get(RAW_URL)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            st.session_state.qa_df = df
            st.success("Loaded qa_gold.csv from GitHub!")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")

with tab2:
    st.markdown("#### Upload your own CSV (must have 'question' & 'answer' columns)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.qa_df = df
            st.success("CSV uploaded.")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")

# Use persistent dataframe for all logic below
df = st.session_state.qa_df

if df is None:
    st.info("Please load or upload a CSV in the tabs above.")
    st.stop()

if "question" not in df.columns or "answer" not in df.columns:
    st.error("CSV must have 'question' and 'answer' columns!")
    st.stop()

df["question"] = df["question"].astype(str).str.strip()
df["answer"] = df["answer"].astype(str).str.strip()
df = df.dropna(subset=['question', 'answer']).reset_index(drop=True)
st.write(f"📊 Loaded {len(df)} Q‑A pairs")
st.dataframe(df.head(3))

### === Sidebar: Options ===
with st.sidebar:
    st.markdown("### 🔗 Connection Status")
    def check_supabase():
        try:
            sup = st.connection("supabase", type=SupabaseConnection, url=SUPABASE_URL, key=SUPABASE_KEY)
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
    st.write(sup_msg)
    st.write(open_msg)

    st.markdown("### ⚙️ Evaluation Settings")
    use_langchain_eval = st.checkbox("Use LangChain QA Evaluation", value=True)
    use_custom_eval = st.checkbox("Use Custom Bengali Evaluation", value=True)
    use_cosine_eval = st.checkbox("Use Cosine Similarity", value=False)

    st.markdown("### 📊 Processing Options")
    batch_size = st.slider("Batch Size (for large datasets)", 1, 50, 10)
    add_delay = st.checkbox("Add delays (avoid rate limits)", value=True)

if not (sup_status and open_status):
    st.warning("⚠️ Please check your connections before proceeding.")
    st.stop()

supabase_conn = st.connection("supabase", type=SupabaseConnection, url=SUPABASE_URL, key=SUPABASE_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
vectorstore = SupabaseVectorStore(client=supabase_conn.client, embedding=embeddings,
                                  table_name="documents", query_name="match_documents")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
system_prompt = """তুমি একজন সহায়ক বাংলা সহকারী।
তুমি শুধুমাত্র নিচের তথ্যসূত্র দেখে উত্তর দেবে—যদি তথ্য না থাকে, দয়া করে বলো 'তথ্য নেই'।

তথ্যসূত্র:
{context}"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt), 
    ("human", "{input}")
])
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.5, api_key=OPENAI_API_KEY)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag = create_retrieval_chain(retriever, qa_chain)

def custom_bengali_eval(question, model_answer, reference_answer, llm):
    eval_prompt = f"""তুমি একজন বাংলা ভাষার বিশেষজ্ঞ মূল্যায়নকারী।

প্রশ্ন: {question}
মডেলের উত্তর: {model_answer}
সঠিক উত্তর: {reference_answer}

নিচের মানদণ্ড অনুযায়ী মডেলের উত্তরটি মূল্যায়ন করো:
1. তথ্যগত সঠিকতা (৫০%)
2. প্রাসঙ্গিকতা (৩০%)
3. ভাষার মান ও স্পষ্টতা (২০%)

০ থেকে ১ এর মধ্যে একটি স্কোর দাও:
শুধু স্কোর দাও (যেমন: 0.8):"""
    try:
        response = llm.invoke(eval_prompt)
        score_text = response.content.strip()
        score_match = re.search(r'([0-1]\.?\d*)', score_text)
        if score_match:
            score = float(score_match.group(1))
            return min(max(score, 0.0), 1.0)
        return 0.0
    except Exception as e:
        st.warning(f"Custom evaluation failed: {e}")
        return 0.0

def bengali_text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    text1 = re.sub(r'\s+', ' ', str(text1).strip())
    text2 = re.sub(r'\s+', ' ', str(text2).strip())
    chars1 = set(text1.lower())
    chars2 = set(text2.lower())
    if not chars1 and not chars2:
        return 1.0
    if not chars1 or not chars2:
        return 0.0
    overlap = len(chars1.intersection(chars2))
    total = len(chars1.union(chars2))
    return overlap / total if total > 0 else 0.0

if st.button("🚀 Generate Model Answers and Evaluate", type="primary"):
    # 1. Generate answers
    st.subheader("🤖 Generating Model Answers")
    answers, contexts = [], []
    progress = st.progress(0)
    for i, q in enumerate(df["question"]):
        result = rag.invoke({"input": q})
        answers.append(result.get("answer", "তথ্য নেই"))
        contexts.append(" || ".join([doc.page_content for doc in result.get("context", [])]))
        progress.progress((i+1)/len(df))
        if add_delay and i % batch_size == 0:
            time.sleep(0.5)
    df["model_answer"] = answers
    df["retrieved_context"] = contexts
    st.session_state.qa_df = df  # save new columns back to session
    st.success("✅ All model answers generated!")

    # 2. LangChain Evaluation
    if use_langchain_eval:
        st.subheader("🔍 LangChain QA Eval")
        eval_chain = QAEvalChain.from_llm(ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.0, api_key=OPENAI_API_KEY))
        lc_scores = []
        lc_grades = []
        progress = st.progress(0)
        for i, row in df.iterrows():
            res = eval_chain.evaluate_strings(prediction=row["model_answer"], reference=row["answer"], input=row["question"])
            lc_scores.append(res.get("score", 0))
            lc_grades.append(res.get("value", ""))
            progress.progress((i+1)/len(df))
            if add_delay and i % batch_size == 0:
                time.sleep(0.3)
        df["langchain_score"] = lc_scores
        df["langchain_grade"] = lc_grades

    # 3. Custom Bengali Evaluation
    if use_custom_eval:
        st.subheader("🇧🇩 Custom Bengali Evaluation")
        custom_scores = []
        progress = st.progress(0)
        for i, row in df.iterrows():
            score = custom_bengali_eval(row["question"], row["model_answer"], row["answer"], llm)
            custom_scores.append(score)
            progress.progress((i+1)/len(df))
            if add_delay and i % batch_size == 0:
                time.sleep(0.3)
        df["custom_bengali_score"] = custom_scores

    # 4. Cosine Similarity
    if use_cosine_eval:
        st.subheader("📊 Cosine Similarity (Experimental)")
        emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        gt_emb = emb.embed_documents(df["answer"].astype(str).tolist())
        mod_emb = emb.embed_documents(df["model_answer"].astype(str).tolist())
        sims = cosine_similarity(np.array(gt_emb), np.array(mod_emb)).diagonal()
        df["cosine_similarity"] = sims
    else:
        st.subheader("📝 Bengali Text Similarity")
        text_sims = []
        for _, row in df.iterrows():
            sim = bengali_text_similarity(row["model_answer"], row["answer"])
            text_sims.append(sim)
        df["bengali_text_similarity"] = text_sims

    # Final Results
    st.subheader("📈 Final Results")
    if use_custom_eval and use_cosine_eval:
        df["final_score"] = (df["custom_bengali_score"] * 0.7 + df["cosine_similarity"] * 0.3)
    elif use_custom_eval:
        df["final_score"] = (df["custom_bengali_score"] * 0.8 + df.get("bengali_text_similarity", pd.Series(0, index=df.index)) * 0.2)
    elif use_langchain_eval and use_cosine_eval:
        df["final_score"] = (pd.to_numeric(df["langchain_score"], errors="coerce").fillna(0) * 0.7 + df["cosine_similarity"] * 0.3)
    else:
        df["final_score"] = pd.to_numeric(df.get("langchain_score", [0]*len(df)), errors="coerce").fillna(0)

    st.session_state.qa_df = df  # persist

    display_cols = ["question", "model_answer", "answer"]
    if use_langchain_eval:
        display_cols.extend(["langchain_score", "langchain_grade"])
    if use_custom_eval:
        display_cols.append("custom_bengali_score")
    if use_cosine_eval:
        display_cols.append("cosine_similarity")
    else:
        display_cols.append("bengali_text_similarity")
    display_cols.append("final_score")
    st.dataframe(df[display_cols], use_container_width=True)

    st.markdown("### 📊 Summary Statistics")
    summary_cols = []
    if use_langchain_eval:
        summary_cols.append("langchain_score")
    if use_custom_eval:
        summary_cols.append("custom_bengali_score")
    if use_cosine_eval:
        summary_cols.append("cosine_similarity")
    else:
        summary_cols.append("bengali_text_similarity")
    summary_cols.append("final_score")
    st.write(df[summary_cols].describe())

    avg_score = df["final_score"].mean()
    if avg_score >= 0.8:
        st.success(f"🎉 Excellent Performance! Average Score: {avg_score:.3f}")
    elif avg_score >= 0.6:
        st.info(f"👍 Good Performance! Average Score: {avg_score:.3f}")
    elif avg_score >= 0.4:
        st.warning(f"⚠️ Moderate Performance. Average Score: {avg_score:.3f}")
    else:
        st.error(f"🚨 Needs Improvement. Average Score: {avg_score:.3f}")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bengali_qa_evaluation_results_{timestamp}.csv"
    st.download_button(
        "⬇️ Download Full Evaluation CSV",
        data=df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
        type="primary"
    )
else:
    st.info("👆 Click the button above to generate model answers for all questions and run evaluation.")

st.markdown("---")
st.caption("🚀 Powered by LangChain, Supabase, OpenAI, and Streamlit - Bengali QA")
