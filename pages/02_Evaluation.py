import streamlit as st
import pandas as pd
import numpy as np
from langchain.evaluation.qa.eval_chain import QAEvalChain
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="QA Evaluation", layout="wide")
st.title("📊 QA Evaluation Dashboard")

uploaded = st.file_uploader("Upload QA gold file (JSONL or CSV)", type=["jsonl","csv"])
if not uploaded:
    st.info("Upload your gold QA file.")
    st.stop()

df = pd.read_json(uploaded, lines=True) if uploaded.name.endswith(".jsonl") else pd.read_csv(uploaded)

model_answers = []
st.subheader("Enter model answers for each question:")
for i, row in df.iterrows():
    ans = st.text_area(f"{i+1}. {row['question']}", height=80, key=f"ans{i}")
    model_answers.append(ans)
df["model_answer"] = model_answers

llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.0)
eval_chain = QAEvalChain.from_llm(llm)

st.subheader("LangChain QA Evaluation Scores")
if st.button("Run LangChain Eval"):
    eval_inputs = [
        {"input": row["question"], "prediction": row["model_answer"], "reference": row["answer"]}
        for _, row in df.iterrows()
    ]
    eval_out = eval_chain.evaluate_strings(input_list=eval_inputs)
    for i, res in enumerate(eval_out):
        df.at[i, "langchain_score"] = res.get("score")
        df.at[i, "langchain_grade"] = res.get("value")
    st.dataframe(df[["question","model_answer","answer","langchain_score","langchain_grade"]])

st.subheader("Cosine Similarity Evaluation")
if st.button("Compute Cosine Similarity"):
    emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["OPENAI_API_KEY"])
    gt = emb.embed_documents(df["answer"].tolist())
    ma = emb.embed_documents(df["model_answer"].tolist())
    sims = cosine_similarity(np.array(gt), np.array(ma)).diagonal()
    df["cosine_similarity"] = sims
    st.dataframe(df[["question","cosine_similarity"]])

if "langchain_score" in df.columns and "cosine_similarity" in df.columns:
    df["final_score"] = (df["langchain_score"].astype(float) + df["cosine_similarity"]) / 2
    st.subheader("📌 Final Score Summary")
    st.dataframe(df[["question","final_score"]])
    st.markdown("### Summary Statistics")
    st.write(df[["langchain_score","cosine_similarity","final_score"]].describe())

if st.button("Download results as CSV"):
    st.download_button("Download CSV", df.to_csv(index=False), file_name="qa_eval_results.csv")
