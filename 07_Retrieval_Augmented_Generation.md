# Retrieval-Augmented Generation (RAG)

## 1. RAG Architecture

### 1.1 RAG Pipeline

```
┌──────────────────────────────────────────────────┐
│ User Query                                       │
└─────────────────┬────────────────────────────────┘
                  │
         ┌────────▼──────────┐
         │ Query Embedding   │
         └────────┬──────────┘
                  │
    ┌─────────────▼──────────────┐
    │ Vector Similarity Search   │
    │ (Find relevant documents)  │
    └─────────────┬──────────────┘
                  │
    ┌─────────────▼──────────────┐
    │ Retrieved Context          │
    │ (Top-K documents)          │
    └─────────────┬──────────────┘
                  │
    ┌─────────────▼──────────────┐
    │ Augmented Prompt           │
    │ (Query + Context)          │
    └─────────────┬──────────────┘
                  │
    ┌─────────────▼──────────────┐
    │ LLM Generation             │
    │ (Generate answer)          │
    └─────────────┬──────────────┘
                  │
         ┌────────▼──────────┐
         │ Final Answer      │
         └───────────────────┘
```

### 1.2 RAG Benefits

| Benefit | Impact | Mechanism |
|---------|--------|----------|
| **Reduced hallucination** | 40-60% fewer false facts | Grounding in retrieved docs |
| **Current information** | Always up-to-date | Can retrieve from live sources |
| **Explainability** | Can cite sources | Retrieved documents are traceable |
| **Cost reduction** | 30-50% fewer tokens | Retrieved context is typically shorter |
| **Domain adaptation** | Works on custom data | Query against proprietary knowledge |

## 2. Retrieval Systems

### 2.1 Embedding Models

```python
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

class EmbeddingSelector:
    """
    Choosing embedding model involves trade-offs:
    """
    
    MODELS = {
        "OpenAI text-embedding-3-small": {
            "dimensions": 1536,
            "cost": "$0.02 / 1M tokens",
            "latency": "~100ms",
            "quality": "Very High"
        },
        "OpenAI text-embedding-3-large": {
            "dimensions": 3072,
            "cost": "$0.13 / 1M tokens",
            "latency": "~150ms",
            "quality": "Highest"
        },
        "Sentence-BERT (local)": {
            "dimensions": 384,
            "cost": "Free (self-hosted)",
            "latency": "~20ms",
            "quality": "High"
        },
        "BGE-Large (local)": {
            "dimensions": 1024,
            "cost": "Free (self-hosted)",
            "latency": "~50ms",
            "quality": "Very High"
        }
    }
    
    @staticmethod
    def select(use_case: str) -> str:
        selection = {
            "production_high_volume": "OpenAI text-embedding-3-small",
            "highest_quality": "OpenAI text-embedding-3-large",
            "private_data": "BGE-Large (local)",
            "latency_critical": "Sentence-BERT (local)"
        }
        return selection.get(use_case, "OpenAI text-embedding-3-small")
```

### 2.2 Vector Search Algorithms

```python
import numpy as np

class VectorSearchAlgorithms:
    """
    Different similarity metrics for retrieval
    """
    
    @staticmethod
    def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        Cosine similarity: measures angle between vectors
        Range: [-1, 1] (typically [0, 1] for embeddings)
        
        Formula: cos(θ) = A·B / (||A|| ||B||)
        
        Properties:
        - Direction-based (ignores magnitude)
        - Efficient
        - Most common for embeddings
        """
        # Normalize
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(doc_norms, query_norm.T).flatten()
        return similarities
    
    @staticmethod
    def euclidean_distance(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        Euclidean distance: straight-line distance
        Range: [0, ∞]
        
        Formula: d = sqrt(Σ(a_i - b_i)²)
        
        Properties:
        - Magnitude-sensitive
        - Slower than cosine
        - Useful for spatial clustering
        """
        differences = doc_vecs - query_vec
        distances = np.linalg.norm(differences, axis=1)
        return -distances  # Negate for ranking (higher = better)
    
    @staticmethod
    def maximum_inner_product(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        Inner product: measure of alignment
        Range: [-1, 1] for normalized vectors
        
        Formula: a·b = Σ(a_i × b_i)
        
        Properties:
        - Fast (single dot product)
        - Magnitude-sensitive
        - Common in approximate search (MIPS)
        """
        return np.dot(doc_vecs, query_vec.T).flatten()
```

### 2.3 Approximate Nearest Neighbors (ANN)

```python
class ANNSearch:
    """
    Approximate Nearest Neighbors for fast retrieval
    
    Algorithms:
    - HNSW: Hierarchical Navigable Small World
    - Locality Sensitive Hashing (LSH)
    - Product Quantization (PQ)
    - Inverted File Index (IVF)
    """
    
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
    
    # HNSW Properties
    HNSW = {
        "complexity_build": "O(n log n)",
        "complexity_search": "O(log n)",
        "memory_overhead": "16-40 bytes per vector",
        "accuracy": "99%+",
        "libraries": ["hnswlib", "Weaviate", "Pinecone"]
    }
    
    # LSH Properties
    LSH = {
        "complexity_build": "O(n)",
        "complexity_search": "O(1) in hashtable",
        "memory_overhead": "Low",
        "accuracy": "90-95%",
        "libraries": ["FALCONN", "datasketch"]
    }
    
    # PQ Properties
    PQ = {
        "compression": "1-8 bytes per vector (vs 4KB for FP32)",
        "accuracy": "95%+",
        "speed": "10-100x faster",
        "libraries": ["Faiss", "Hnswlib"]
    }
```

## 3. Document Chunking and Preparation

### 3.1 Chunking Strategies

```python
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChunkingStrategy:
    """
    Smart document chunking balances:
    - Semantic coherence
    - Retrieval precision
    - Context completeness
    """
    
    # Strategy 1: Fixed size with overlap
    @staticmethod
    def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
        """
        Simple but effective: divide into fixed-size chunks with overlap
        
        Pros: Fast, deterministic
        Cons: May cut mid-sentence
        """
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        return chunks
    
    # Strategy 2: Semantic chunking (sentence-aware)
    @staticmethod
    def semantic_chunking(text: str, target_size: int = 500) -> list:
        """
        Split by sentences, combine until reaching target size
        
        Pros: Respects sentence boundaries
        Cons: Chunks of variable size
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < target_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    # Strategy 3: Recursive hierarchical
    @staticmethod
    def recursive_chunking(text: str, chunk_size: int = 500) -> list:
        """
        Try to split by paragraph, then sentence, then character
        
        Pros: Optimal semantic boundaries
        Cons: Complex logic
        """
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=100
        )
        return splitter.split_text(text)
    
    # Strategy 4: Metadata-aware
    @staticmethod
    def metadata_aware_chunking(text: str, structure: dict) -> list:
        """
        Use document structure (headings, sections) for chunking
        
        structure: {"h1": 1000, "h2": 500, "p": 200}
        """
        # Parse markdown/structured text
        # Split by headings respecting hierarchy
        # Return chunks with metadata
        pass
```

### 3.2 Chunk Enrichment

```python
class ChunkEnrichment:
    """
    Add metadata and context to chunks for better retrieval
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def add_summaries(self, chunks: list) -> list:
        """
        Add summary to each chunk for better retrieval
        """
        enriched = []
        
        for chunk in chunks:
            summary_prompt = f"Summarize in 1 sentence:\n{chunk}"
            summary = self.llm.invoke(summary_prompt)
            
            enriched.append({
                "content": chunk,
                "summary": summary,
                "embedding": None  # Will be computed
            })
        
        return enriched
    
    def add_keywords(self, chunks: list) -> list:
        """
        Extract keywords for keyword-based retrieval
        """
        enriched = []
        
        for chunk in chunks:
            keyword_prompt = f"Extract 5 key terms:\n{chunk}\nTerms: "
            keywords = self.llm.invoke(keyword_prompt).split(",")
            
            enriched.append({
                "content": chunk,
                "keywords": [k.strip() for k in keywords],
                "embedding": None
            })
        
        return enriched
```

## 4. Ranking and Re-ranking

### 4.1 Retrieval Ranking

```python
from rank_bm25 import BM25Okapi

class HybridRetrieval:
    """
    Combine dense (semantic) and sparse (keyword) retrieval
    
    Algorithm:
    1. Dense search (vector similarity)
    2. Sparse search (BM25 keyword matching)
    3. Combine scores with weighting
    """
    
    def __init__(self, documents: list, embeddings_model):
        self.documents = documents
        self.embeddings = embeddings_model
        
        # Setup BM25
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def retrieve(self, query: str, k: int = 5, alpha: float = 0.5) -> list:
        """
        Hybrid retrieval combining dense and sparse
        
        alpha: weight for semantic vs keyword (0-1)
        """
        # Dense retrieval (semantic)
        query_embedding = self.embeddings.embed_query(query)
        semantic_scores = self._compute_semantic_scores(query_embedding)
        
        # Sparse retrieval (BM25)
        keyword_scores = self._compute_bm25_scores(query)
        
        # Combine scores
        combined_scores = [
            alpha * semantic_scores[i] + (1 - alpha) * keyword_scores[i]
            for i in range(len(self.documents))
        ]
        
        # Return top-k
        top_indices = sorted(
            range(len(combined_scores)),
            key=lambda i: combined_scores[i],
            reverse=True
        )[:k]
        
        return [self.documents[i] for i in top_indices]
    
    def _compute_semantic_scores(self, query_embedding):
        # Compute cosine similarity
        pass
    
    def _compute_bm25_scores(self, query):
        # Use BM25 scores
        return self.bm25.get_scores(query.split())
```

### 4.2 Re-ranking

```python
class CrossEncoderReranker:
    """
    Re-rank initial retrieval results using cross-encoder
    
    Cross-encoder: Takes (query, document) pairs, outputs relevance score
    More accurate but slower than bi-encoder
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        """
        Re-rank documents using cross-encoder
        
        Typical accuracy improvement: 5-15% over initial ranking
        """
        # Prepare pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score pairs
        scores = self.model.predict(pairs)
        
        # Sort by score
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, score in ranked[:top_k]]
```

## 5. Advanced RAG Patterns

### 5.1 Iterative Refinement

```python
class IterativeRAG:
    """
    Multiple retrieval passes to refine answer
    
    Algorithm:
    1. Initial retrieval based on query
    2. Generate partial answer
    3. Identify missing information
    4. Reformulate query for missing info
    5. Retrieve additional context
    6. Refine answer
    """
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def answer_with_iteration(self, query: str, max_iterations: int = 3) -> dict:
        all_context = []
        answer = None
        
        for iteration in range(max_iterations):
            # Retrieve context
            context = self.retriever.retrieve(query, k=5)
            all_context.extend(context)
            
            # Generate answer
            answer = self._generate_answer(query, all_context)
            
            # Check if answer is complete
            if self._is_complete(answer):
                break
            
            # Identify gaps and reformulate
            gaps = self._identify_gaps(answer)
            query = f"{query}. Also need: {gaps}"
        
        return {
            "answer": answer,
            "iterations": iteration + 1,
            "sources": all_context
        }
    
    def _generate_answer(self, query: str, context: list) -> str:
        prompt = f"""
Based on this context, answer the question.

Context:
{chr(10).join(context)}

Question: {query}

Answer:"""
        return self.llm.invoke(prompt)
    
    def _is_complete(self, answer: str) -> bool:
        # Check for uncertainty markers
        markers = ["unclear", "unable", "insufficient", "unclear"]
        return not any(m in answer.lower() for m in markers)
    
    def _identify_gaps(self, answer: str) -> str:
        prompt = f"What information is missing from this answer?\n{answer}"
        return self.llm.invoke(prompt)
```

## Key Takeaways

1. **RAG reduces hallucination**: Ground generation in retrieved facts
2. **Chunking matters**: Smart chunking improves retrieval quality
3. **Hybrid search wins**: Combine semantic and keyword approaches
4. **Reranking improves**: Cross-encoders significantly boost accuracy
5. **Iteration refines**: Multiple passes catch missing information

## References

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020): https://arxiv.org/abs/2005.11401
- Vector Database Guide: https://www.pinecone.io/learn/vector-database/
- LangChain RAG: https://python.langchain.com/docs/use_cases/question_answering/
