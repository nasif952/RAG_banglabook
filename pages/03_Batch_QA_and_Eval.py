import streamlit as st
import pandas as pd
import numpy as np
import time
import re

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

st.title("🤖 Batch QA Generator + Evaluator (Bengali)")

# 1. Upload CSV
uploaded = st.file_uploader("Upload gold QA file (CSV, with columns 'question','answer')", type=["csv"])
if not uploaded:
    st.info("Upload your gold QA CSV (must have column 'question' and 'answer').")
    st.stop()

# Read CSV with proper encoding for Bengali
try:
    df = pd.read_csv(uploaded, encoding='utf-8')
except:
    df = pd.read_csv(uploaded, encoding='utf-8-sig')

if "question" not in df.columns or "answer" not in df.columns:
    st.error("CSV must have 'question' and 'answer' columns!")
    st.stop()

# Clean up any encoding issues
df["question"] = df["question"].astype(str).str.strip()
df["answer"] = df["answer"].astype(str).str.strip()

st.write(f"📊 Loaded {len(df)} question-answer pairs")
st.write("**Sample data:**")
st.dataframe(df.head(3))

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
    
    st.markdown("### Evaluation Settings")
    use_langchain_eval = st.checkbox("Use LangChain QA Evaluation", value=True)
    use_custom_eval = st.checkbox("Use Custom Bengali Evaluation", value=True)
    use_cosine_eval = st.checkbox("Use Cosine Similarity (Experimental)", value=False)
    
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

# Custom Bengali Evaluation Function
def custom_bengali_eval(question, model_answer, reference_answer, llm):
    """Custom evaluation function specifically designed for Bengali QA"""
    eval_prompt = f"""তুমি একজন বাংলা ভাষার বিশেষজ্ঞ মূল্যায়নকারী।

প্রশ্ন: {question}

মডেলের উত্তর: {model_answer}

সঠিক উত্তর: {reference_answer}

নিচের মানদণ্ড অনুযায়ী মডেলের উত্তরটি মূল্যায়ন করো:
1. তথ্যগত সঠিকতা (৫০%)
2. প্রাসঙ্গিকতা (৩০%)  
3. ভাষার মান ও স্পষ্টতা (২০%)

০ থেকে ১ এর মধ্যে একটি স্কোর দাও:
- ১.০ = সম্পূর্ণ সঠিক ও উপযুক্ত
- ০.৮ = বেশিরভাগ সঠিক, সামান্য ভুল
- ০.৬ = আংশিক সঠিক
- ০.৪ = কিছু প্রাসঙ্গিক তথ্য আছে
- ০.২ = খুবই কম প্রাসঙ্গিক
- ০.০ = সম্পূর্ণ ভুল বা অপ্রাসঙ্গিক

শুধু স্কোর দাও (যেমন: 0.8):"""
    
    try:
        response = llm.invoke(eval_prompt)
        score_text = response.content.strip()
        # Extract number from response
        score_match = re.search(r'([0-1]\.?\d*)', score_text)
        if score_match:
            score = float(score_match.group(1))
            return min(max(score, 0.0), 1.0)  # Ensure 0-1 range
        return 0.0
    except Exception as e:
        st.warning(f"Custom evaluation failed: {e}")
        return 0.0

# Simple Bengali text similarity (character-based)
def bengali_text_similarity(text1, text2):
    """Simple character-level similarity for Bengali text"""
    if not text1 or not text2:
        return 0.0
    
    # Remove extra whitespace and normalize
    text1 = re.sub(r'\s+', ' ', str(text1).strip())
    text2 = re.sub(r'\s+', ' ', str(text2).strip())
    
    # Simple character overlap ratio
    chars1 = set(text1.lower())
    chars2 = set(text2.lower())
    
    if not chars1 and not chars2:
        return 1.0
    if not chars1 or not chars2:
        return 0.0
    
    overlap = len(chars1.intersection(chars2))
    total = len(chars1.union(chars2))
    
    return overlap / total if total > 0 else 0.0

if st.button("🔄 Generate Model Answers and Evaluate"):
    # === STEP 1: Generate Model Answers ===
    st.subheader("🤖 Generating Model Answers")
    answers, contexts = [], []
    progress = st.progress(0)
    
    for i, q in enumerate(df["question"]):
        with st.spinner(f"Generating ({i+1}/{len(df)}): {q[:25]}..."):
            try:
                result = rag.invoke({"input": q})
                answers.append(result.get("answer", "তথ্য নেই"))
                context_text = " || ".join([doc.page_content for doc in result.get("context", [])])
                contexts.append(context_text)
            except Exception as e:
                st.warning(f"Generation failed for question {i+1}: {e}")
                answers.append("তথ্য নেই")
                contexts.append("")
        
        progress.progress((i+1)/len(df))
        time.sleep(0.1)
    
    df["model_answer"] = answers
    df["retrieved_context"] = contexts
    st.success("✅ All model answers generated!")

    # === STEP 2: LangChain QA Evaluation ===
    if use_langchain_eval:
        st.subheader("🔍 LangChain QA Eval (LLM-based)")
        
        debug_container = st.container()
        with debug_container:
            st.write("🔧 **Debug Information:**")
            debug_text = st.empty()
        
        try:
            debug_text.write("Creating LangChain evaluation chain...")
            eval_chain = QAEvalChain.from_llm(
                ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
            )
            debug_text.write("✅ LangChain evaluation chain created successfully")
            
            langchain_scores = []
            langchain_grades = []
            
            eval_progress = st.progress(0)
            eval_status = st.empty()
            
            for idx, (_, row) in enumerate(df.iterrows()):
                eval_status.write(f"LangChain evaluating question {idx+1}/{len(df)}...")
                
                try:
                    result = eval_chain.evaluate_strings(
                        prediction=str(row["model_answer"]),
                        reference=str(row["answer"]),
                        input=str(row["question"])
                    )
                    
                    score = result.get("score", 0)
                    grade = result.get("value", "UNKNOWN")
                    
                    langchain_scores.append(score)
                    langchain_grades.append(grade)
                    
                    debug_text.write(f"✅ Question {idx+1}: Score={score}, Grade={grade}")
                    
                except Exception as e:
                    error_msg = f"❌ LangChain eval failed for Q{idx+1}: {str(e)}"
                    debug_text.write(error_msg)
                    langchain_scores.append(0)
                    langchain_grades.append("ERROR")
                
                eval_progress.progress((idx+1)/len(df))
                time.sleep(0.5)
            
            df["langchain_score"] = langchain_scores
            df["langchain_grade"] = langchain_grades
            debug_text.write("✅ LangChain evaluation completed!")
            
        except Exception as e:
            st.error(f"LangChain evaluation setup failed: {str(e)}")
            df["langchain_score"] = [0] * len(df)
            df["langchain_grade"] = ["ERROR"] * len(df)

    # === STEP 3: Custom Bengali Evaluation ===
    if use_custom_eval:
        st.subheader("🇧🇩 Custom Bengali Evaluation")
        
        custom_scores = []
        custom_progress = st.progress(0)
        custom_status = st.empty()
        
        for idx, (_, row) in enumerate(df.iterrows()):
            custom_status.write(f"Custom evaluating question {idx+1}/{len(df)}...")
            
            score = custom_bengali_eval(
                row["question"], 
                row["model_answer"], 
                row["answer"], 
                llm
            )
            custom_scores.append(score)
            custom_progress.progress((idx+1)/len(df))
            time.sleep(0.3)
        
        df["custom_bengali_score"] = custom_scores
        st.success("✅ Custom Bengali evaluation completed!")

    # === STEP 4: Cosine Similarity (Optional) ===
    if use_cosine_eval:
        st.subheader("📊 Cosine Similarity (Experimental for Bengali)")
        st.warning("Note: Cosine similarity may not work well for Bengali due to embedding limitations")
        
        try:
            emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
            
            # Add small delay to avoid rate limits
            st.write("Computing embeddings...")
            gt_emb = emb.embed_documents(df["answer"].astype(str).tolist())
            time.sleep(1)
            mod_emb = emb.embed_documents(df["model_answer"].astype(str).tolist())
            
            sims = cosine_similarity(np.array(gt_emb), np.array(mod_emb)).diagonal()
            df["cosine_similarity"] = sims
            st.success("✅ Cosine similarity computed!")
            
        except Exception as e:
            st.error(f"Cosine similarity failed: {str(e)}")
            df["cosine_similarity"] = [0.0] * len(df)
    else:
        # Use simple Bengali text similarity instead
        st.subheader("📝 Bengali Text Similarity")
        text_sims = []
        for _, row in df.iterrows():
            sim = bengali_text_similarity(row["model_answer"], row["answer"])
            text_sims.append(sim)
        df["bengali_text_similarity"] = text_sims
        st.success("✅ Bengali text similarity computed!")

    # === STEP 5: Final Results ===
    st.subheader("📈 Final Results")
    
    # Calculate composite scores
    if use_custom_eval and use_cosine_eval:
        df["final_score"] = (df["custom_bengali_score"] * 0.7 + df["cosine_similarity"] * 0.3)
    elif use_custom_eval:
        df["final_score"] = (df["custom_bengali_score"] * 0.8 + df["bengali_text_similarity"] * 0.2)
    elif use_langchain_eval and use_cosine_eval:
        df["final_score"] = (pd.to_numeric(df["langchain_score"], errors="coerce").fillna(0) * 0.7 + df["cosine_similarity"] * 0.3)
    else:
        df["final_score"] = pd.to_numeric(df.get("langchain_score", [0]*len(df)), errors="coerce").fillna(0)

    # Display results
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
    
    st.dataframe(df[display_cols])

    # Summary Statistics
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
    
    # Performance Analysis
    st.markdown("### 🎯 Performance Analysis")
    avg_score = df["final_score"].mean()
    if avg_score >= 0.8:
        st.success(f"🎉 Excellent Performance! Average Score: {avg_score:.3f}")
    elif avg_score >= 0.6:
        st.info(f"👍 Good Performance! Average Score: {avg_score:.3f}")
    elif avg_score >= 0.4:
        st.warning(f"⚠️ Moderate Performance. Average Score: {avg_score:.3f}")
    else:
        st.error(f"🚨 Needs Improvement. Average Score: {avg_score:.3f}")

    # Download button
    st.download_button(
        "⬇️ Download Full Evaluation CSV",
        data=df.to_csv(index=False),
        file_name="bengali_qa_evaluation_results.csv",
        mime="text/csv"
    )

else:
    st.info("Click above to generate model answers for all questions and run evaluation.")

st.caption("Powered by LangChain, Supabase, OpenAI, and Streamlit - Optimized for Bengali QA")
