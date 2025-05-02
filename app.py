


import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import ast


st.set_page_config(page_title="Course Recommender", layout="wide")
openai.api_key = "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"  


@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\moham\Desktop\LLM Data\udemy_courses2.csv")
    df['embedding'] = df['embedding'].apply(ast.literal_eval).apply(np.array)
    return df

df_embeddings = load_data()


def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model='text-embedding-ada-002'
    )
    return np.array(response.data[0].embedding)

def get_recommendation_from_user_input(df_embeddings, user_input, k=10):
    user_embedding = get_embedding(user_input).reshape(1, -1)

    similarities = df_embeddings['embedding'].apply(
        lambda x: cosine_similarity(user_embedding, x.reshape(1, -1))[0][0]
    )

    df_embeddings['similarity'] = similarities
    df_sorted = df_embeddings.sort_values(by='similarity', ascending=False)

    recommendations = []
    for _, row in df_sorted.iloc[:k].iterrows():
        rec = {
            'Title': row['title'],
            'Subject':row['headline'],
            'URL':row['url'],
            'Similarity': round(row['similarity'], 3)
        }
        recommendations.append(rec)
    return recommendations


col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.image("udemy_logo.png", width=150)

with col2:
    st.markdown("""
        <h2 style='text-align: center; color: black; margin-top: 50px;'>
        Course Recommendation System using LLM & Embeddings
        </h2>
    """, unsafe_allow_html=True)

with col3:
    st.empty()

st.subheader("Find the best online courses based on your interests 👇")

user_input = st.text_input("What topics or skills are you interested in learning?")

if st.button("Show Recommendations"):
    if user_input.strip() == "":
        st.error("❌ Please enter a topic or interest to get recommendations.")
    else:
        with st.spinner("Generating recommendations... Please wait..."):
            recommendations = get_recommendation_from_user_input(df_embeddings, user_input, k=10)
            st.success(f"✅ Top {len(recommendations)} recommended courses for: **{user_input}**")
        for rec in recommendations:
            st.markdown(f"""
            <div style='font-size:18px; line-height:1.6'>
            📚 <b>{rec['Title']}</b><br>
            📝 {rec['Subject']}<br>
            🔗 <a href="{rec['URL']}" target="_blank">Course Link</a>
            </div>
            <hr>
            """, unsafe_allow_html=True)

