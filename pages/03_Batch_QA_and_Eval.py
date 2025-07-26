import streamlit as st
import pandas as pd
import numpy as np
import time

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

st.title("🤖 Batch QA Generator + Evaluator")

# 1. Upload CSV
uploaded = st.file_uploader("Upload gold QA file (CSV, with columns 'question','answer')", type=["csv"])
if not uploaded:
    st.info("Upload your gold QA CSV (must have column 'question' and 'answer').")
    st.stop()
df = pd.read_csv(uploaded)
if "question" not in df.columns or "answer" not in df.columns:
    st.error("CSV must have 'question' and 'answer' columns!")
    st.stop()

# 2. Setup RAG
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
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, api_key=OPENAI_API_KEY)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag = create_retrieval_chain(retriever, qa_chain)

if st.button("🔄 Generate Model Answers and Evaluate"):
    answers, contexts = [], []
    progress = st.progress(0)
    for i, q in enumerate(df["question"]):
        with st.spinner(f"Generating ({i+1}/{len(df)}): {q[:25]}..."):
            result = rag.invoke({"input": q})
            answers.append(result.get("answer", "তথ্য নেই"))
            context_text = " || ".join([doc.page_content for doc in result.get("context", [])])
            contexts.append(context_text)
        progress.progress((i+1)/len(df))
        time.sleep(0.1)
    df["model_answer"] = answers
    df["retrieved_context"] = contexts
    st.success("All model answers generated!")

    # --- Auto Evaluation (Fixed) ---
    st.subheader("LangChain QA Eval (LLM-based)")
    eval_chain = QAEvalChain.from_llm(ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY))
    
    langchain_scores = []
    langchain_grades = []
    
    # Evaluate each question-answer pair individually
    for _, row in df.iterrows():
        try:
            # Use the correct API with named parameters
            result = eval_chain.evaluate_strings(
                prediction=row["model_answer"],
                reference=row["answer"],
                input=row["question"]
            )
            langchain_scores.append(result.get("score"))
            langchain_grades.append(result.get("value"))
        except Exception as e:
            st.warning(f"Evaluation failed for question: {row['question'][:50]}... Error: {e}")
            langchain_scores.append(0)
            langchain_grades.append("INCORRECT")
    
    df["langchain_score"] = langchain_scores
    df["langchain_grade"] = langchain_grades

    # --- Cosine Similarity Eval ---
    st.subheader("Cosine Similarity (Embeddings)")
    emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    gt_emb = emb.embed_documents(df["answer"].astype(str).tolist())
    mod_emb = emb.embed_documents(df["model_answer"].astype(str).tolist())
    sims = cosine_similarity(np.array(gt_emb), np.array(mod_emb)).diagonal()
    df["cosine_similarity"] = sims

    # --- Results ---
    df["final_score"] = pd.to_numeric(df["langchain_score"], errors="coerce").fillna(0) * 0.5 + df["cosine_similarity"] * 0.5
    st.dataframe(df[["question", "model_answer", "answer", "langchain_score", "langchain_grade", "cosine_similarity", "final_score"]])

    st.markdown("### Summary Statistics")
    st.write(df[["langchain_score","cosine_similarity","final_score"]].describe())

    st.download_button(
        "⬇️ Download Full Evaluation CSV",
        data=df.to_csv(index=False),
        file_name="qa_with_model_and_eval.csv",
        mime="text/csv"
    )
else:
    st.info("Click above to generate model answers for all questions and run evaluation.")

st.caption("Powered by LangChain, Supabase, OpenAI, and Streamlit")
