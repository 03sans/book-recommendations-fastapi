import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("📚 Book Recommender App")

tab1, tab2 = st.tabs(["🔍 Content-Based", "👤 Collaborative Filtering"])

with tab1:
    st.header("Content-Based Recommendation")
    title = st.text_input("Enter a book title:")
    if st.button("Get Recommendations", key="content"):
        if title.strip():
            with st.spinner("Fetching recommendations..."):
                response = requests.get(f"{API_URL}/recommend/content", params={"title": title})
                if response.status_code == 200:
                    recs = response.json().get("recommendations", [])
                    st.success("Recommendations:")
                    for i, book in enumerate(recs, 1):
                        st.write(f"{i}. {book}")
                else:
                    st.error("Failed to fetch recommendations.")

with tab2:
    st.header("Collaborative Filtering Recommendation")
    user_id = st.number_input("Enter your User ID:", min_value=0, step=1)
    if st.button("Get Recommendations", key="collab"):
        with st.spinner("Fetching recommendations..."):
            response = requests.get(f"{API_URL}/recommend/collab", params={"user_id": user_id})
            if response.status_code == 200:
                recs = response.json().get("recommendations", [])
                st.success("Recommendations:")
                for i, book in enumerate(recs, 1):
                    st.write(f"{i}. {book}")
            else:
                st.error("Failed to fetch recommendations.")