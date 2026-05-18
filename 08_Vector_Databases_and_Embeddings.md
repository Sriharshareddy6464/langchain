# Vector Databases and Embeddings

## 1. Embedding Models Deep Dive

### 1.1 Embedding Model Taxonomy

```
Embedding Models
├── General Purpose
│   ├── OpenAI text-embedding-3 (SOTA commercial)
│   ├── Sentence-BERT / All-MiniLM (Popular open-source)
│   └── BGE (Chinese-friendly, competitive with OpenAI)
├── Domain-Specific
│   ├── Medical: PubMedBERT
│   ├── Legal: Legal-BERT
│   └── Code: CodeBERT
└── Multilingual
    ├── mBERT (100+ languages)
    ├── XLM-RoBERTa
    └── LaBSE
```

### 1.2 Embedding Quality Metrics

```python
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

class EmbeddingEvaluator:
    """
    Measure embedding quality through standardized benchmarks
    """
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def evaluate_similarity(self, pairs: list) -> dict:
        """
        Evaluate on semantic similarity task
        
        Input: List of (sentence1, sentence2, human_score) tuples
        Output: Spearman correlation with human judgments
        """
        embeddings1 = []
        embeddings2 = []
        human_scores = []
        
        for sent1, sent2, score in pairs:
            emb1 = self.model.encode(sent1)
            emb2 = self.model.encode(sent2)
            
            embeddings1.append(emb1)
            embeddings2.append(emb2)
            human_scores.append(score)
        
        # Compute cosine similarities
        model_scores = [
            1 - cosine(e1, e2)
            for e1, e2 in zip(embeddings1, embeddings2)
        ]
        
        # Correlation with human judgments
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(human_scores, model_scores)
        
        return {
            "correlation": correlation,
            "p_value": p_value,
            "model_scores": model_scores
        }
    
    def benchmark_performance(self, test_datasets: dict) -> dict:
        """
        Benchmark on standard datasets:
        - STS (Semantic Textual Similarity)
        - TREC (Information Retrieval)
        - NLI (Natural Language Inference)
        """
        results = {}
        
        for dataset_name, dataset in test_datasets.items():
            results[dataset_name] = self.evaluate_similarity(dataset)
        
        # Compute average
        avg_correlation = np.mean([
            r["correlation"] for r in results.values()
        ])
        
        return {
            "results_by_dataset": results,
            "average_correlation": avg_correlation
        }

# MTEB Leaderboard (reference scores)
EMBEDDING_BENCHMARKS = {
    "text-embedding-3-large": {"avg_score": 0.875, "dims": 3072, "cost_high": True},
    "text-embedding-3-small": {"avg_score": 0.862, "dims": 1536, "cost_low": True},
    "bge-large-en-v1.5": {"avg_score": 0.866, "dims": 1024, "cost_none": True},
    "all-MiniLM-L6-v2": {"avg_score": 0.842, "dims": 384, "cost_none": True},
}
```

### 1.3 Embedding Dimensions and Trade-offs

```python
class DimensionAnalysis:
    """
    Impact of embedding dimensions on quality vs storage/speed
    """
    
    # Dimensional reduction analysis
    DIMENSION_TRADEOFFS = {
        "full_dim": {
            "description": "Original dimension (typically 768-3072)",
            "quality": "100%",
            "storage": "100%",
            "search_speed": "100%",
            "use_case": "Maximum accuracy needed"
        },
        "reduced_512": {
            "description": "Reduced via PCA",
            "quality": "98%",
            "storage": "66%",
            "search_speed": "120%",  # Better cache locality
            "use_case": "High-volume retrieval"
        },
        "reduced_256": {
            "description": "Aggressive reduction",
            "quality": "94%",
            "storage": "33%",
            "search_speed": "140%",
            "use_case": "Cost-sensitive deployment"
        },
    }
    
    @staticmethod
    def reduce_dimensions(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        \"\"\"\n        Reduce embedding dimensions via PCA\n        \n        Preserves top principal components\n        \"\"\"\n        from sklearn.decomposition import PCA\n        \n        pca = PCA(n_components=target_dim)\n        reduced = pca.fit_transform(embeddings)\n        \n        # Explained variance\n        variance_ratio = np.sum(pca.explained_variance_ratio_)\n        \n        print(f\"Explained variance: {variance_ratio:.2%}\")\n        return reduced\n    
    @staticmethod\n    def quantize_embeddings(embeddings: np.ndarray, dtype: str = \"float16\") -> np.ndarray:\n        \"\"\"\n        Quantize embeddings to reduce storage\n        \n        float32: 4 bytes per value\n        float16: 2 bytes per value (50% reduction, ~0.1% accuracy loss)\n        int8: 1 byte per value (75% reduction, ~1% accuracy loss)\n        \"\"\"\n        if dtype == \"float16\":\n            return embeddings.astype(np.float16)\n        elif dtype == \"int8\":\n            # Normalize and convert to int8\n            normalized = (embeddings + 1) / 2  # [-1, 1] -> [0, 1]\n            return (normalized * 127).astype(np.int8)\n        return embeddings
```

## 2. Vector Database Internals

### 2.1 HNSW (Hierarchical Navigable Small World)

```python
import heapq
from typing import List, Tuple

class HSNWIndex:
    \"\"\"\n    HNSW Algorithm (used in Weaviate, Pinecone, Hnswlib)\n    \n    Complexity:\n    - Insert: O(log N)\n    - Search: O(log N)\n    - Space: O(N) with ~M edges per node\n    \"\"\"\n    
    def __init__(self, dimension: int, max_neighbors: int = 16, ef_construction: int = 200):\n        self.dimension = dimension\n        self.max_neighbors = max_neighbors  # M parameter\n        self.ef_construction = ef_construction  # Size of dynamic list\n        self.vectors = {}  # id -> vector\n        self.graph = {}   # id -> [neighbor_ids]\n        self.entry_point = None\n    \n    def insert(self, vector_id: int, vector: np.ndarray):\n        \"\"\"\n        Insert vector with HNSW algorithm\n        \n        Algorithm:\n        1. Find nearest neighbor in graph\n        2. Add connections at all levels\n        3. Update entry point if necessary\n        \"\"\"\n        if self.entry_point is None:\n            self.entry_point = vector_id\n            self.vectors[vector_id] = vector\n            self.graph[vector_id] = []\n            return\n        \n        self.vectors[vector_id] = vector\n        \n        # Find nearest neighbors\n        candidates = self._search_layer(\n            vector,\n            [self.entry_point],\n            ef=self.ef_construction\n        )\n        \n        # Connect new node\n        self.graph[vector_id] = [cand_id for cand_id, _ in candidates[:self.max_neighbors]]\n        \n        # Update reverse connections (prune if necessary)\n        for neighbor_id, dist in candidates[:self.max_neighbors]:\n            if len(self.graph[neighbor_id]) < self.max_neighbors:\n                self.graph[neighbor_id].append(vector_id)\n            else:\n                # Prune: keep only closest neighbors\n                neighbor_dists = [\n                    (nid, self._distance(self.vectors[nid], self.vectors[neighbor_id]))\n                    for nid in self.graph[neighbor_id]\n                ]\n                neighbor_dists.sort(key=lambda x: x[1])\n                self.graph[neighbor_id] = [nid for nid, _ in neighbor_dists[:self.max_neighbors]]\n    \n    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:\n        \"\"\"\n        Search for k nearest neighbors\n        \"\"\"\n        candidates = self._search_layer(\n            query_vector,\n            [self.entry_point],\n            ef=k * 2  # ef parameter controls accuracy vs speed\n        )\n        \n        return candidates[:k]\n    \n    def _search_layer(self, query: np.ndarray, entry_points: List[int], ef: int):\n        \"\"\"\n        Search single layer (greedy nearest neighbor search)\n        \n        ef: Size of dynamic list (higher = more accurate but slower)\n        \"\"\"\n        visited = set()\n        candidates = []\n        w = []\n        \n        for point in entry_points:\n            dist = self._distance(query, self.vectors[point])\n            heapq.heappush(candidates, (-dist, point))\n            heapq.heappush(w, (dist, point))\n            visited.add(point)\n        \n        lowerbound = max(candidates)[0]\n        \n        while candidates:\n            current_dist, current = heapq.heappop(candidates)\n            current_dist = -current_dist\n            \n            if current_dist > lowerbound:\n                break\n            \n            # Check neighbors of current\n            for neighbor in self.graph.get(current, []):\n                if neighbor not in visited:\n                    visited.add(neighbor)\n                    dist = self._distance(query, self.vectors[neighbor])\n                    \n                    if dist < lowerbound or len(w) < ef:\n                        heapq.heappush(candidates, (-dist, neighbor))\n                        heapq.heappush(w, (dist, neighbor))\n                        lowerbound = max(w)[0]\n        \n        return sorted(w, key=lambda x: x[0])\n    \n    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:\n        \"\"\"Euclidean distance\"\"\"\n        return np.linalg.norm(v1 - v2)\n```

### 2.2 Inverted File Index with Product Quantization (IVF+PQ)

```python
class IVFPQIndex:
    \"\"\"\n    Product Quantization approach (used in Faiss)\n    \n    Compression: Reduce 4KB vector to 32-128 bytes\n    Speed: 100-1000x faster search\n    Accuracy: 95%+ preserved\n    \"\"\"\n    
n def __init__(self, dimension: int, n_clusters: int = 100, n_subquantizers: int = 16):\n        self.dimension = dimension\n        self.n_clusters = n_clusters  # IVF clusters\n        self.n_subquantizers = n_subquantizers  # PQ components\n        self.subvector_dim = dimension // n_subquantizers\n        \n        self.cluster_centers = None\n        self.pq_codebooks = {}  # cluster_id -> quantizer\n        self.vectors_by_cluster = {}  # cluster_id -> [vector_ids]\n    \n    def fit(self, vectors: np.ndarray):\n        \"\"\"\n        Train IVF clusters and PQ quantizers\n        \"\"\"\n        # Step 1: K-means on full vectors (IVF)\n        from sklearn.cluster import KMeans\n        kmeans = KMeans(n_clusters=self.n_clusters)\n        clusters = kmeans.fit_predict(vectors)\n        self.cluster_centers = kmeans.cluster_centers_\n        \n        # Step 2: For each cluster, train PQ quantizer\n        for cluster_id in range(self.n_clusters):\n            cluster_mask = clusters == cluster_id\n            cluster_vectors = vectors[cluster_mask]\n            \n            if len(cluster_vectors) > 0:\n                # Train PQ on subvectors\n                pq = ProductQuantizer(\n                    subvector_dim=self.subvector_dim,\n                    n_subquantizers=self.n_subquantizers\n                )\n                pq.fit(cluster_vectors)\n                self.pq_codebooks[cluster_id] = pq\n                self.vectors_by_cluster[cluster_id] = cluster_vectors\n    \n    def search(self, query: np.ndarray, k: int = 10, n_probe: int = 10) -> List[Tuple[int, float]]:\n        \"\"\"\n        Search: find nearby clusters, then search within clusters\n        \"\"\"\n        # Find nearest clusters\n        distances_to_clusters = np.linalg.norm(\n            self.cluster_centers - query,\n            axis=1\n        )\n        nearest_clusters = np.argsort(distances_to_clusters)[:n_probe]\n        \n        # Search within each cluster\n        candidates = []\n        for cluster_id in nearest_clusters:\n            if cluster_id in self.pq_codebooks:\n                pq = self.pq_codebooks[cluster_id]\n                cluster_vectors = self.vectors_by_cluster[cluster_id]\n                \n                # Use PQ to approximate distances\n                dists = pq.distance_to_vectors(query, cluster_vectors)\n                \n                for i, dist in enumerate(dists):\n                    heapq.heappush(candidates, (dist, (cluster_id, i)))\n        \n        return [(idx, dist) for dist, idx in sorted(candidates)[:k]]\n```

## 3. Vector Database Comparison

### 3.1 Feature Comparison

```python
VECTOR_DB_COMPARISON = {
    "Pinecone": {
        "type": \"Managed Service\",
        \"index_type\": \"Proprietary (HNSW-based)\",
        \"filters\": \"Yes\",
        \"metadata_filtering\": \"Full support\",
        \"hybrid_search\": \"Yes\",
        \"cost_model\": \"Pay-per-use (1M vectors ~$0.50/month)\",
        \"latency\": \"<100ms\",
        \"uptime_sla\": \"99.95%\",
        \"best_for\": \"Production apps needing reliability\",
    },
    \"Weaviate\": {
        \"type\": \"Self-hosted or managed\",
        \"index_type\": \"HNSW\",
        \"filters\": \"Yes (BooleanFilter)\",
        \"metadata_filtering\": \"Full support\",
        \"hybrid_search\": \"Yes (BM25 + vector)\",
        \"cost_model\": \"Open source (free) + ops cost\",
        \"latency\": \"<50ms (self-hosted)\",
        \"uptime_sla\": \"Depends on deployment\",
        \"best_for\": \"Custom deployments, no vendor lock-in\",
    },
    \"Chroma\": {
        \"type\": \"Python library + optional server\",
        \"index_type\": \"Configurable (HNSW default)\",
        \"filters\": \"Yes\",
        \"metadata_filtering\": \"Partial\",
        \"hybrid_search\": \"No (vector only)\",
        \"cost_model\": \"Free (open source)\",
        \"latency\": \"<10ms (in-process)\",
        \"best_for\": \"Prototyping, small-scale RAG\",
    },
    \"Milvus\": {
        \"type\": \"Self-hosted (cloud available)\",
        \"index_type\": \"HNSW, IVF, DISKANN\",
        \"filters\": \"Yes (scalar filters)\",
        \"metadata_filtering\": \"Full support\",
        \"hybrid_search\": \"Yes\",
        \"cost_model\": \"Free (open source) + cloud pricing\",
        \"latency\": \"<100ms\",
        \"best_for\": \"Large-scale on-prem deployments\",
    },
    \"Qdrant\": {
        \"type\": \"Self-hosted or cloud\",
        \"index_type\": \"Custom engine\",
        \"filters\": \"Yes\",
        \"metadata_filtering\": \"Excellent support\",
        \"hybrid_search\": \"Yes\",
        \"cost_model\": \"Free (OSS) or cloud pricing\",
        \"latency\": \"<10ms (self-hosted)\",
        \"best_for\": \"Low-latency search, filtering-heavy workloads\",
    },
}
```

### 3.2 Selection Decision Tree

```python
class VectorDBSelector:
    \"\"\"\n    Select optimal vector database for use case
    \"\"\"\n    \n    @staticmethod\n    def recommend(requirements: dict) -> str:\n        \"\"\"\n        requirements:\n        - scale: \"small\" (<1M), \"medium\" (1M-100M), \"large\" (>100M)\n        - latency: \"<10ms\", \"<100ms\", \"flexible\"\n        - filtering: boolean\n        - hybrid_search: boolean\n        - self_hosted: boolean\n        - budget: \"free\", \"low\", \"high\"\n        \"\"\"\n        \n        scale = requirements.get(\"scale\")\n        latency = requirements.get(\"latency\")\n        self_hosted = requirements.get(\"self_hosted\")\n        budget = requirements.get(\"budget\")\n        \n        # Decision tree\n        if self_hosted:\n            if scale == \"small\" and latency == \"<10ms\":\n                return \"Chroma or Qdrant\"\n            elif scale in [\"medium\", \"large\"]:\n                return \"Milvus or Weaviate\"\n            else:\n                return \"Qdrant (best balanced)\"\n        else:\n            # Managed service\n            if budget == \"free\":\n                return \"None (must self-host)\"\n            elif requirements.get(\"reliability_critical\"):\n                return \"Pinecone (99.95% SLA)\"\n            else:\n                return \"Pinecone or cloud Weaviate\"\n```

## 4. Indexing Strategies

### 4.1 Batch Indexing

```python
class BatchIndexing:\n    \"\"\"\n    Efficient bulk insertion of vectors\n    \"\"\"\n    \n    def __init__(self, vector_store, batch_size: int = 1000):\n        self.vector_store = vector_store\n        self.batch_size = batch_size\n    \n    def index_documents(self, documents: List[dict], embedding_model):\n        \"\"\"\n        Index documents in batches\n        \n        documents: [{\"id\": \"...\", \"text\": \"...\"}, ...]\n        \"\"\"\n        batches = [\n            documents[i:i + self.batch_size]\n            for i in range(0, len(documents), self.batch_size)\n        ]\n        \n        for batch in batches:\n            # Embed batch (parallel)\n            embeddings = embedding_model.embed_documents(\n                [doc[\"text\"] for doc in batch]\n            )\n            \n            # Index batch\n            metadata = [{\"id\": doc[\"id\"], **doc.get(\"metadata\", {})} for doc in batch]\n            self.vector_store.add_vectors(\n                vectors=embeddings,\n                metadatas=metadata,\n                ids=[doc[\"id\"] for doc in batch]\n            )\n            \n            print(f\"Indexed {len(batch)} documents\")\n\n# Performance benchmarks\nINDEXING_BENCHMARKS = {\n    \"1M vectors\": {\n        \"time_hnsw\": \"~5 minutes\",\n        \"time_ivf_pq\": \"~2 minutes\",\n        \"memory_hnsw\": \"~8GB\",\n        \"memory_ivf_pq\": \"~2GB\",\n    },\n    \"100M vectors\": {\n        \"time_hnsw\": \"~8 hours\",\n        \"time_ivf_pq\": \"~1 hour\",\n        \"memory_hnsw\": \"~800GB\",\n        \"memory_ivf_pq\": \"~40GB\",\n    }\n}\n```\n\n## 5. Performance Optimization\n\n### 5.1 Query Optimization\n\n```python\nclass QueryOptimizer:\n    \"\"\"\n    Optimize vector search performance\n    \"\"\"\n    \n    @staticmethod\n    def optimize_ef_parameter(vector_db, ground_truth: List[int], queries: List[np.ndarray]):\n        \"\"\"\n        Find optimal ef value (accuracy vs speed trade-off)\n        \"\"\"\n        ef_values = [10, 50, 100, 200, 500]\n        results = []\n        \n        for ef in ef_values:\n            recall = 0\n            search_time = 0\n            \n            for i, query in enumerate(queries):\n                start = time.time()\n                results_ef = vector_db.search(query, k=10, ef=ef)\n                search_time += time.time() - start\n                \n                # Compute recall\n                retrieved_ids = [r[0] for r in results_ef]\n                relevant = len(set(retrieved_ids) & set(ground_truth[i]))\n                recall += relevant / len(ground_truth[i])\n            \n            results.append({\n                \"ef\": ef,\n                \"recall\": recall / len(queries),\n                \"latency_ms\": (search_time / len(queries)) * 1000\n            })\n        \n        return results\n    \n    @staticmethod\n    def cache_frequent_queries(query_cache: dict, queries: List[str], k: int = 10):\n        \"\"\"\n        Cache results for frequently repeated queries\n        \n        Typical hit rate: 20-40% on real workloads\n        \"\"\"\n        from collections import Counter\n        query_counts = Counter(queries)\n        \n        # Cache top queries\n        for query, count in query_counts.most_common(k):\n            # Pre-compute and cache\n            query_cache[query] = query_cache.get(query, [])\n```\n\n## Key Takeaways\n\n1. **HNSW is fast**: O(log N) search with minimal tuning\n2. **Quantization compresses**: 10-100x size reduction\n3. **Filtering is complex**: Choose DB with strong filter support\n4. **Hybrid search wins**: Combine vector + keyword approaches\n5. **Benchmarking matters**: Measure ef, batch size, and other hyperparameters\n\n## References\n\n- HNSW Paper: https://arxiv.org/abs/1802.02413\n- Product Quantization: https://arxiv.org/abs/1411.2590\n- Vector DB Benchmarks: https://github.com/erikbern/ann-benchmarks\n- Pinecone Blog: https://www.pinecone.io/blog/\n