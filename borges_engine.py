import os
import re
import sys
import arxiv
import faiss
import logging
import pickle
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)


class ArgentineResearchEngine:
    def __init__(self, model_name="distiluse-base-multilingual-cased-v1"):
        logger.info(f"🧠 Initializing model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.raw_data = []
        self.processed_docs = []

    def _preprocess_text(self, text):
        """Internal helper to clean academic abstracts."""
        text = re.sub(r"\$.*?\$", "", text)  # Remove LaTeX math
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def fetch_data(self, limit=50):
        """Fetches new papers and ADDS them to the existing collection."""
        logger.info(f"📡 Searching for {limit} new papers...")
        client = arxiv.Client()
        query_str = '(cat:econ.GN OR cat:econ.EM OR cat:q-fin.GN) AND (Argentina OR "Latin America")'

        search = arxiv.Search(
            query=query_str,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        # Get existing URLs to avoid duplicates
        existing_urls = {item["url"] for item in self.raw_data}
        new_count = 0

        for r in client.results(search):
            if r.entry_id not in existing_urls:
                # Add to raw_data
                self.raw_data.append(
                    {
                        "title": r.title,
                        "summary": r.summary,
                        "url": r.entry_id,
                        "authors": [a.name for a in r.authors],
                        "year": r.published.year,
                    }
                )
                # Add to processed_docs
                self.processed_docs.append(self._preprocess_text(r.summary))
                new_count += 1

        logger.info(f"✨ Added {new_count} unique papers to the library.")
        return new_count

    def build_index(self):
        """
        Rebuilds the index using the TOTAL collection.
        Generates embeddings and initializes FAISS.
        """
        if not self.processed_docs:
            return

        logger.info(
            f"🔢 Vectorizing entire library ({len(self.processed_docs)} docs)..."
        )
        # Note: For very large libraries (>10,000 docs), we would
        # only encode the NEW docs, but for now, re-encoding ensures alignment.
        embeddings = self.model.encode(self.processed_docs, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        logger.info(f"✅ Library now contains {self.index.ntotal} searchable papers.")
        logger.info(f"✅ FAISS index built with {self.index.ntotal} vectors.")

    def save_index(self, folder="index_data"):
        """Saves the FAISS index and metadata to disk."""
        if not os.path.exists(folder):
            os.makedirs(folder)

        # 1. Save the FAISS index (using native FAISS tools)
        faiss.write_index(self.index, os.path.join(folder, "science.index"))

        # 2. Save the metadata (raw_data and processed_docs)
        metadata = {"raw_data": self.raw_data, "processed_docs": self.processed_docs}
        with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"💾 Index and metadata saved to folder: '{folder}'")

    def load_index(self, folder="index_data"):
        """Loads the FAISS index and metadata from disk."""
        index_path = os.path.join(folder, "science.index")
        meta_path = os.path.join(folder, "metadata.pkl")

        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            logger.error("❌ Save files not found. Run fetch and build first.")
            return False

        # 1. Load the FAISS index
        self.index = faiss.read_index(index_path)

        # 2. Load the metadata
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self.raw_data = data["raw_data"]
            self.processed_docs = data["processed_docs"]

        logger.info(f"✅ Successfully loaded {len(self.raw_data)} papers from disk.")
        return True

    def search(self, query_text, k=3):
        """Performs semantic similarity search."""
        if self.index is None:
            raise ValueError("Index not initialized. Run build_index() first.")

        clean_query = self._preprocess_text(query_text)
        query_vector = self.model.encode([clean_query]).astype("float32")
        distances, indices = self.index.search(query_vector, k)
        return distances, indices

    def update_library(self, limit=100):
        """The command to grow the database."""
        added = self.fetch_data(limit=limit)
        if added > 0:
            self.build_index()
            self.save_index()  # Save the new combined state to disk
        else:
            print("🕵️ No new papers found that weren't already in your index.")

    def show_stats(self):
        if not self.raw_data:
            print("📭 The library is empty.")
            return

        print("\n📊 BORGES INDEX LIBRARY STATISTICS")
        print("=" * 40)

        years = [item.get("year", "Unknown") for item in self.raw_data]
        year_counts = Counter(years)

        print("\n📅 Papers by Year:")
        # Sort years, putting "Unknown" at the end
        sorted_years = sorted(
            [y for y in year_counts.keys() if y != "Unknown"], reverse=True
        )
        if "Unknown" in year_counts:
            sorted_years.append("Unknown")

        for year in sorted_years:
            bar = "▇" * (year_counts[year])
            print(f"  {year}: {year_counts[year]:02d} {bar}")  # Top Authors
        # We flatten the list of author lists
        all_authors = [auth for item in self.raw_data for auth in item["authors"]]
        top_authors = Counter(all_authors).most_common(5)

        print("\n✍️ Top Contributing Authors:")
        for author, count in top_authors:
            print(f"  • {author}: {count} papers")
        print("-" * 40)

    def display_results(self, query_text, distances, indices):
        """Prints the search results in a clean format."""
        print("\n" + "🔎" + f" Semantic Search Results for: '{query_text}'")
        print("=" * 60)

        for i, idx in enumerate(indices[0]):
            paper = self.raw_data[idx]
            dist = distances[0][i]
            author_list = ", ".join(paper["authors"])

            print(f"Rank {i + 1} | [Distance Score: {dist:.4f}]")
            print(f"📄 Title:   {paper['title']}")
            print(f"✍️ Authors: {author_list}")
            print(f"🔗 Link:    {paper['url']}")
            print(f"📝 Summary: {self.processed_docs[idx][:190]}...")
            print("-" * 60)

    def run_interactive_assistant(self):
        print("\n" + "=" * 60)
        print("🇦🇷 ARGENTINE HISTORY & ECON RESEARCH ASSISTANT")
        print("=" * 60)
        print("Commands:")
        print("  • Type a topic to search")
        print("  • '/limit [number]' - Update library size (e.g., /limit 200)")
        print("  • 'exit' or 'quit' - Close the assistant")

        while True:
            try:
                user_input = input("\n🔎 Query or Command: ").strip()

                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                if user_input.lower() == "/stats":
                    self.show_stats()
                    continue
                if user_input.startswith("/limit"):
                    try:
                        # Extract the number from the string "/limit 500"
                        new_limit = int(user_input.split()[1])
                        self.update_library(limit=new_limit)
                    except (IndexError, ValueError):
                        print("❌ Please provide a valid number. Example: /limit 150")
                    continue

                # Regular Search
                distances, indices = self.search(user_input)
                self.display_results(user_input, distances, indices)

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    engine = ArgentineResearchEngine()
    INDEX_FOLDER = "argentine_econ_history"

    # Check if we already have data saved
    if engine.load_index(INDEX_FOLDER):
        logger.info("🚀 Starting with cached data...")
    else:
        logger.info("📡 No local data found. Fetching from arXiv...")
        engine.fetch_data(limit=100)
        engine.build_index()
        engine.save_index(INDEX_FOLDER)

    engine.run_interactive_assistant()
