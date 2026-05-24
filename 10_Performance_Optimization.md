# Performance Optimization

## 1. Latency Analysis and Optimization

### 1.1 Latency Breakdown

```
Total End-to-End Latency
├── Network Overhead (5-20ms)
│   ├── DNS lookup (1-5ms)
│   ├── TCP handshake (5-10ms)
│   └── TLS negotiation (5-20ms)
├── Request Serialization (1-5ms)
├── LLM API Call (500-5000ms)
│   ├── Queue time (0-1000ms)
│   ├── Processing (300-3000ms)
│   └── Network roundtrip (50-100ms)
├── Response Parsing (5-50ms)
├── Retrieval (if RAG) (50-500ms)
│   ├── Query encoding (10-50ms)
│   ├── Vector search (10-100ms)
│   └── Re-ranking (20-200ms)
└── Post-processing (5-50ms)

Typical Distribution:
- LLM call: 80-90% (dominant)
- Retrieval: 5-10%
- Network/serialization: 2-5%
```

### 1.2 Latency Optimization Techniques

```python
from datetime import datetime
import asyncio

class LatencyOptimizer:
    """
    Reduce end-to-end latency
    """
    
    def __init__(self):
        self.latency_breakdown = {}
    
    # Technique 1: Request Batching
    def batch_requests(self, requests: list, batch_size: int = 32):
        """
        Batch multiple requests to amortize overhead
        
        Benefit: 2-3x throughput improvement
        Trade-off: Adds ~100ms latency per request
        """
        batches = [
            requests[i:i + batch_size]
            for i in range(0, len(requests), batch_size)
        ]
        
        results = []
        for batch in batches:
            # Process batch with single API call
            batch_response = self._process_batch(batch)
            results.extend(batch_response)
        
        return results
    
    # Technique 2: Streaming Responses
    async def stream_response(self, query: str, llm):
        """
        Stream tokens instead of waiting for full response
        
        Benefit: First token in 100-200ms (vs 1-2s for full)
        Use case: Real-time user interfaces
        """
        first_token_time = None
        
        async for token in llm.astream(query):
            if first_token_time is None:
                first_token_time = datetime.now()
                time_to_first = (first_token_time - datetime.now()).total_seconds() * 1000
                print(f"Time to first token: {time_to_first:.0f}ms")
            
            yield token
    
    # Technique 3: Concurrent Operations
    async def concurrent_retrieval_and_processing(self, query: str):
        """
        Parallelize retrieval and initial processing
        
        Benefit: Hide retrieval latency (50-100ms savings)
        """
        # Start retrieval and query understanding in parallel
        retrieval_task = asyncio.create_task(self._retrieve_context(query))
        understanding_task = asyncio.create_task(self._understand_query(query))
        
        # Wait for both
        context, understood_query = await asyncio.gather(
            retrieval_task,
            understanding_task
        )
        
        return context, understood_query
    
    # Technique 4: Model Selection for Speed
    def select_fast_model(self, complexity: str) -> str:
        """
        Use faster models for simple tasks
        
        Model Latency Comparison:
        - gpt-3.5-turbo: 300-800ms (fast)
        - gpt-4: 1000-3000ms (slow, more capable)
        - gpt-4-turbo: 700-2000ms (balanced)
        """
        model_selection = {
            "simple_qa": "gpt-3.5-turbo",      # 400ms avg
            "reasoning": "gpt-4-turbo",        # 1200ms avg
            "complex_analysis": "gpt-4",       # 2000ms avg
        }
        return model_selection.get(complexity, "gpt-3.5-turbo")
    
    # Technique 5: Request Prefetching
    def prefetch_common_queries(self, llm, common_queries: list):
        """
        Warm up cache with common queries
        
        Benefit: Subsequent requests hit cache (10-50ms)
        """
        for query in common_queries:
            # Execute to populate cache
            try:
                llm.invoke(query)
            except:
                pass  # Silent fail
    
    def measure_latency(self, func, *args, **kwargs) -> dict:
        """
        Detailed latency measurement
        """
        start = datetime.now()
        result = func(*args, **kwargs)
        total = (datetime.now() - start).total_seconds() * 1000
        
        return {
            "total_ms": total,
            "result": result
        }
```

### 1.3 Latency Percentile Analysis

```python
import numpy as np
from collections import defaultdict

class LatencyAnalyzer:
    """
    Analyze latency distribution for SLA compliance
    """
    
    def __init__(self):
        self.latencies = defaultdict(list)
    
    def analyze_percentiles(self, endpoint: str, percentiles: list = [50, 95, 99, 99.9]):
        """
        Compute latency percentiles
        
        Typical SLA targets:
        - p50: <200ms (median user experience)
        - p95: <1000ms (most users happy)
        - p99: <3000ms (outlier tolerance)
        """
        latencies = sorted(self.latencies[endpoint])
        
        if not latencies:
            return {}
        
        results = {}
        for p in percentiles:
            index = int(len(latencies) * p / 100)
            results[f"p{p}"] = latencies[index]
        
        return results
    
    def identify_bottlenecks(self) -> dict:
        """
        Find slowest operations
        """
        bottlenecks = {}
        
        for endpoint, latencies in self.latencies.items():
            avg = np.mean(latencies)
            max_latency = np.max(latencies)
            percentile_99 = np.percentile(latencies, 99)
            
            bottlenecks[endpoint] = {
                "avg_ms": avg,
                "p99_ms": percentile_99,
                "max_ms": max_latency,
                "outlier_ratio": len([l for l in latencies if l > percentile_99 * 2]) / len(latencies)
            }
        
        return bottlenecks
    
    def identify_performance_regression(self, baseline: dict, current: dict):
        """
        Detect latency regressions vs baseline
        
        Threshold: >10% increase triggers alert
        """
        regressions = []
        
        for endpoint in baseline:
            baseline_p99 = baseline[endpoint]["p99_ms"]
            current_p99 = current[endpoint]["p99_ms"]
            
            regression_pct = (current_p99 - baseline_p99) / baseline_p99
            
            if regression_pct > 0.1:  # >10% increase
                regressions.append({
                    "endpoint": endpoint,
                    "baseline_p99": baseline_p99,
                    "current_p99": current_p99,
                    "regression_pct": regression_pct * 100
                })
        
        return regressions
```

## 2. Throughput and Scalability

### 2.1 Concurrency Management

```python
import concurrent.futures
from queue import Queue
import threading

class ConcurrencyController:
    """
    Manage concurrent requests while respecting API limits
    """
    
    def __init__(self, max_concurrent: int = 10, max_requests_per_minute: int = 60):
        self.max_concurrent = max_concurrent
        self.max_rpm = max_requests_per_minute
        self.semaphore = threading.Semaphore(max_concurrent)
        self.rate_limiter = TokenBucket(
            capacity=max_requests_per_minute,
            refill_rate=max_requests_per_minute / 60
        )
    
    def process_batch(self, items: list, process_func, max_workers: int = None):
        """
        Process items concurrently with resource limits
        """
        max_workers = max_workers or self.max_concurrent
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for item in items:
                # Respect rate limit
                self.rate_limiter.wait_until_available()
                
                # Submit with semaphore
                future = executor.submit(self._execute_with_semaphore, process_func, item)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"error": str(e)})
        
        return results
    
    def _execute_with_semaphore(self, func, item):
        """Execute function while respecting concurrency limit"""
        with self.semaphore:
            return func(item)

# Throughput benchmark
THROUGHPUT_BENCHMARKS = {
    "sequential": {
        "requests_per_second": 0.5,     # 1 request per 2 seconds
        "api_calls": 100,
        "total_time_seconds": 200
    },
    "concurrent_10": {
        "requests_per_second": 5,       # 5 req/s with 10 concurrent
        "api_calls": 100,
        "total_time_seconds": 20
    },
    "concurrent_32": {
        "requests_per_second": 12,      # 12 req/s with 32 concurrent
        "api_calls": 100,
        "total_time_seconds": 8.5
    }
}
```

### 2.2 Load Balancing Strategies

```python
class LoadBalancer:
    """
    Distribute load across multiple models/endpoints
    """
    
    def __init__(self, endpoints: list):
        self.endpoints = endpoints
        self.request_counts = {ep: 0 for ep in endpoints}
        self.error_rates = {ep: 0.0 for ep in endpoints}
    
    def round_robin(self) -> str:
        """Simple round-robin"""
        endpoint = self.endpoints[min(
            range(len(self.endpoints)),
            key=lambda i: self.request_counts[self.endpoints[i]]
        )]
        self.request_counts[endpoint] += 1
        return endpoint
    
    def least_error_rate(self) -> str:
        """Route to healthiest endpoint"""
        return min(
            self.endpoints,
            key=lambda ep: self.error_rates[ep]
        )
    
    def latency_aware(self, latencies: dict) -> str:
        """Route to fastest responding endpoint"""
        return min(
            self.endpoints,
            key=lambda ep: latencies.get(ep, float('inf'))
        )
    
    def weighted_random(self, weights: dict):
        """Weighted random selection"""
        import random
        selected = random.choices(
            self.endpoints,
            weights=[weights.get(ep, 1) for ep in self.endpoints],
            k=1
        )[0]
        self.request_counts[selected] += 1
        return selected
```

## 3. Memory Optimization

### 3.1 Token Budget Management

```python
import tiktoken

class TokenBudgetManager:
    """
    Manage token usage to optimize cost and latency
    """
    
    def __init__(self, model: str, max_tokens: int = 2048):
        self.model = model
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
    
    def estimate_cost(self, prompt: str, estimated_output_tokens: int = 500) -> float:
        """
        Estimate API cost
        
        OpenAI pricing example:
        - gpt-4: $30 per 1M input, $60 per 1M output
        """
        input_tokens = len(self.encoding.encode(prompt))
        
        PRICING = {
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
        }
        
        pricing = PRICING.get(self.model, {})
        
        input_cost = input_tokens * pricing.get("input", 0)
        output_cost = estimated_output_tokens * pricing.get("output", 0)
        
        return input_cost + output_cost
    
    def trim_context_to_budget(self, context: str, budget_tokens: int) -> str:
        """
        Trim context to fit token budget
        """
        tokens = self.encoding.encode(context)
        
        if len(tokens) <= budget_tokens:
            return context
        
        # Keep most important parts (start, middle important sections, end)
        important_ratio = 0.5
        important_count = int(budget_tokens * important_ratio)
        
        important_tokens = tokens[:important_count]
        trimmed = self.encoding.decode(important_tokens)
        
        return trimmed + "\n[... context trimmed ...]"
    
    def optimize_prompt_length(self, query: str, examples: list, max_budget: int = 2000):
        """
        Use as many examples as fit in token budget
        """
        query_tokens = len(self.encoding.encode(query))
        budget_for_examples = max_budget - query_tokens
        
        selected_examples = []
        current_tokens = 0
        
        for example in examples:
            example_tokens = len(self.encoding.encode(str(example)))
            
            if current_tokens + example_tokens <= budget_for_examples:
                selected_examples.append(example)
                current_tokens += example_tokens
            else:
                break
        
        return selected_examples
```

### 3.2 Memory Profiling

```python
import tracemalloc

class MemoryProfiler:
    """
    Monitor memory usage in LLM applications
    """
    
    def __init__(self):
        tracemalloc.start()
    
    def profile_function(self, func, *args, **kwargs):
        """Profile memory usage of a function"""
        tracemalloc.reset_peak()
        current, peak = tracemalloc.get_traced_memory()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        
        return {
            "result": result,
            "memory_mb": peak / 1024 / 1024
        }
    
    def identify_memory_leaks(self, func, iterations: int = 10):
        """
        Run function repeatedly to detect memory leaks
        """
        memory_trend = []
        
        for i in range(iterations):
            tracemalloc.reset_peak()
            func()
            _, peak = tracemalloc.get_traced_memory()
            memory_trend.append(peak / 1024 / 1024)
        
        # Check if memory is growing
        early_avg = sum(memory_trend[:3]) / 3
        late_avg = sum(memory_trend[-3:]) / 3
        
        growth_pct = (late_avg - early_avg) / early_avg
        
        if growth_pct > 0.1:  # >10% growth
            print(f"⚠️  Possible memory leak: {growth_pct*100:.1f}% growth")
        
        return memory_trend
```

## 4. Resource Utilization

### 4.1 CPU and GPU Optimization

```python
class GPUOptimizer:
    """
    Optimize GPU utilization for inference
    """
    
    def __init__(self):
        self.gpu_available = self._check_gpu()
    
    def _check_gpu(self) -> bool:
        """Check if CUDA/GPU available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def batch_process_gpu(self, inputs: list, batch_size: int = 32):
        """
        Process on GPU in batches for efficiency
        
        Benefits:
        - Amortizes overhead
        - Better GPU utilization (60-80% vs 20% for small batches)
        - Lower cost per token
        """
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Process batch on GPU
            batch_results = self._process_batch_gpu(batch)
            results.extend(batch_results)
        
        return results
    
    def monitor_gpu_utilization(self):
        """Monitor GPU metrics"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {"available": False}
            
            return {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "memory_cached_gb": torch.cuda.memory_cached() / 1e9,
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        except:
            return {"available": False, "error": "CUDA not available"}
```

### 4.2 Cost-Performance Trade-offs

```python
class CostOptimizer:
    """
    Balance cost vs performance
    """
    
    def choose_model_for_slo(self, slo_latency_ms: int, cost_budget: float):
        """
        Select model based on latency SLO and budget
        """
        models = {
            "gpt-3.5-turbo": {"latency_p99": 800, "cost_per_1m_tokens": 1.50},
            "gpt-4": {"latency_p99": 2500, "cost_per_1m_tokens": 90.00},
            "gpt-4-turbo": {"latency_p99": 1500, "cost_per_1m_tokens": 40.00},
            "claude-3-opus": {"latency_p99": 3000, "cost_per_1m_tokens": 75.00},
        }
        
        candidates = [
            model for model, spec in models.items()
            if spec["latency_p99"] <= slo_latency_ms
        ]
        
        if not candidates:
            return None, "No model meets latency SLO"
        
        # Pick cheapest that meets SLO
        best = min(
            candidates,
            key=lambda m: models[m]["cost_per_1m_tokens"]
        )
        
        return best, models[best]["cost_per_1m_tokens"]
```

## 5. Benchmarking and Profiling

### 5.1 Comprehensive Benchmarking

```python
class Benchmark:
    """
    Comprehensive performance benchmarking
    """
    
    def run_benchmark(self, chain, test_cases: list, num_runs: int = 3):
        """
        Run benchmark with multiple runs for stability
        """
        results = {
            "latencies": [],
            "tokens_input": [],
            "tokens_output": [],
            "costs": [],
            "errors": []
        }
        
        for run in range(num_runs):
            for test_case in test_cases:
                try:
                    start = time.time()
                    
                    response = chain.invoke(test_case["input"])
                    
                    latency = (time.time() - start) * 1000
                    
                    results["latencies"].append(latency)
                    results["tokens_input"].append(test_case.get("input_tokens", 0))
                    results["tokens_output"].append(test_case.get("output_tokens", 0))
                    results["costs"].append(test_case.get("estimated_cost", 0))
                    
                except Exception as e:
                    results["errors"].append(str(e))
        
        return self._summarize_results(results)
    
    def _summarize_results(self, results: dict) -> dict:
        """Summarize benchmark results"""
        import statistics
        
        latencies = results["latencies"]
        
        return {
            "latency_p50_ms": statistics.quantiles(latencies, n=100)[49],
            "latency_p95_ms": statistics.quantiles(latencies, n=100)[94],
            "latency_p99_ms": statistics.quantiles(latencies, n=100)[98],
            "latency_mean_ms": statistics.mean(latencies),
            "throughput_qps": 1000 / statistics.mean(latencies),
            "error_rate": len(results["errors"]) / len(latencies),
            "total_cost_usd": sum(results["costs"])
        }
```

## Key Takeaways

1. **Latency is dominated by LLM call**: 80-90% of total time
2. **Concurrency is critical**: 10x throughput improvement with proper management
3. **Streaming reduces perceived latency**: First token in 100-200ms vs 1-2s
4. **Token budgets drive cost**: Every token costs money and time
5. **Benchmarking reveals reality**: Measure before and after optimization

## References

- AWS Well-Architected Framework: https://aws.amazon.com/architecture/well-architected/
- Optimization Handbook: https://www.datadoghq.com/blog/optimization/
- LangChain Performance: https://python.langchain.com/docs/guides/evaluation/
