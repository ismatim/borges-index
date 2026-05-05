# 📚 BorgesIndex: Semantic Research Explorer

**BorgesIndex** is a multilingual semantic search engine designed to navigate the "infinite library" of Argentine scientific research. By leveraging **FAISS** and **Sentence-Transformers**, it allows researchers to find deep conceptual connections in **Economics and History**, bridging the gap between Spanish queries and English academic papers.

---

## ✨ Key Features

- **Multilingual Semantic Search:** Search in Spanish (e.g., _"crisis de deuda"_) and retrieve relevant results in English (e.g., _"Sovereign default analysis"_) using the `distiluse` model.
- **Additive Knowledge Base:** The library grows with you. New fetches are deduplicated and appended to your local archive, allowing you to build a massive personal database over time.
- **FAISS-Powered Speed:** Instantaneous retrieval and similarity ranking even as your collection grows to thousands of documents.
- **Library Statistics:** Built-in visualization tools to see the distribution of your research by **Year** and **Top Authors**.
- **Persistence:** Saves your index and metadata locally; no need to re-download or re-vectorize data between sessions.

---

## 🚀 Installation

Ensure you have Python 3.10+ installed.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/borges-index.git BorgesIndex
   cd BorgesIndex
   ```

2. **Install dependencies:**

   ```bash
   pip install arxiv faiss-cpu sentence-transformers numpy
   ```

---

## 🛠️ Usage

Run the main script to start the interactive research assistant:

```bash
python borges_engine.py
```

### **Interactive Commands**

While the assistant is running, use these commands:

- **`[Search Term]`**: Type any topic (e.g., _"historia agraria"_ or _"inflation dynamics"_) for a semantic search.
- **`/stats`**: View the visual bar chart of papers by year and see the most frequent authors.
- **`/limit [number]`**: Fetch more papers from arXiv and add them to your current library (e.g., `/limit 150`).
- **`exit` or `quit`**: Safely close the assistant and save your index to disk.

---

## ⚙️ Customization

The engine is modular and easy to adapt for other research interests.

### **1. Modify the Research Focus**

To change the categories or countries being searched, edit the `query_str` in the `fetch_data` method:

```python
# Change to search for Physics in Argentina:
query_str = 'cat:physics.* AND (Argentina OR "CONICET")'
```

### **2. Adjusting Summary Snippets**

To change how much of the abstract is shown in the search results, update the slice in the `display_results` method:

```python
print(f"📝 Summary: {self.processed_docs[idx][:300]}...")
```

---

## 🧠 Technical Overview

- **Model:** `distiluse-base-multilingual-cased-v1`. Maps Spanish and English into a shared 512-dimensional vector space.
- **Vector Engine:** `faiss.IndexFlatL2`. Calculates Euclidean distance for high-speed similarity search.
- **Data Source:** [arXiv API](https://arxiv.org/help/api/index).
- **Storage:** \* `borges.index`: Binary FAISS index for vector search.
  - `metadata.pkl`: Serialized Python dictionary containing titles, URLs, years, and authors.

---

## 🇦🇷 Why "BorgesIndex"?

The project is named after **Jorge Luis Borges**, the Argentine master of metaphysical fiction. Inspired by his story _"The Library of Babel,"_ this tool treats the vast output of scientific research as an interconnected universe of meaning where every concept is linked, regardless of the language it was written in.

---

### **Project Structure**

```text
BorgesIndex/
├── borges_engine.py          # Main Engine class and CLI logic
├── argentine_econ_history/   # Local data storage (created on first run)
│   ├── borges.index          # Saved FAISS vectors
│   └── metadata.pkl          # Saved paper information
└── README.md                 # This file
```
