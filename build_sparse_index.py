import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.vectorstores import Chroma

vectorstore = Chroma(
    collection_name="ncert_multilingual",
    persist_directory="./chroma_ncert_db"
)

data = vectorstore._collection.get(include=["documents", "metadatas"])
documents = data["documents"]
metadatas = data["metadatas"]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),
    stop_words="english",
    max_features=20000
)

tfidf_matrix = vectorizer.fit_transform(documents)

with open("sparse_index.pkl", "wb") as f:
    pickle.dump(
        (vectorizer, tfidf_matrix, documents, metadatas),
        f
    )

print("✅ Sparse index saved to sparse_index.pkl")