"""
Run ONCE before starting the app.
python build_data.py
Takes ~20 min. Commits data/ to GitHub afterward.
"""
from src.data_pipeline.scraper import scrape_and_structure
from src.data_pipeline.embedder import build_vectorstore

print("="*50)
print("STEP 1: Scraping + structuring 9 schemes")
print("="*50)
scrape_and_structure()

print("\n" + "="*50)
print("STEP 2: Building FAISS vector store")
print("="*50)
build_vectorstore()

print("\nDONE. Now open data/schemes/ and verify the JSON files look correct.")
print("Then commit everything to GitHub.")
print("Then run: streamlit run app.py")