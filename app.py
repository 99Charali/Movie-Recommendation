import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Movie AI Recommender", page_icon="🍿", layout="wide")

# --- UI HEADER ---
st.title("🎬 AI Movie Recommendation System")
st.markdown("""
    Find movies based on **vibe and description** rather than just titles.
    *Built for the 2026 TMDB Dataset.*
""")

# --- RESOURCE LOADING (With Caching for GitHub/Streamlit Cloud) ---
@st.cache_resource
def load_assets():
    # 1. Load the Embedding Model (The "Brain")
    # This might take a moment on the first run as it downloads from HuggingFace
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Connect to ChromaDB (The "Memory")
    # Path must match the folder created by your embed.py
    db_path = "./vector_engine"
    
    if not os.path.exists(db_path):
        return model, None # Return model but no DB if folder is missing
        
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="movie_vectors")
    return model, collection

# Execution
with st.spinner("Initializing AI Engine... Please wait."):
    model, collection = load_assets()

# --- MAIN LOGIC ---
if collection is None:
    st.error("⚠️ Database folder './vector_engine' not found. Did you run embed.py first?")
else:
    # Search Input
    user_query = st.text_input("Describe the kind of movie you want to watch:", 
                               placeholder="e.g., A futuristic thriller about artificial intelligence and ethics")

    if user_query:
        with st.spinner("Searching through 10,000 movies..."):
            # 1. Convert user input to vector
            query_vector = model.encode([user_query]).tolist()
            
            # 2. Query the Vector DB
            results = collection.query(
                query_embeddings=query_vector,
                n_results=6
            )
            
            # 3. Display Results in a Grid
            st.divider()
            cols = st.columns(3)
            
            for idx, i in enumerate(range(len(results['ids'][0]))):
                meta = results['metadatas'][0][i]
                with cols[idx % 3]:
                    st.subheader(f"{meta['title']}")
                    st.caption(f"⭐ Rating: {meta['vote_average']}")
                    st.write(meta['overview'][:200] + "...")
                    st.button(f"View Details", key=f"btn_{idx}")