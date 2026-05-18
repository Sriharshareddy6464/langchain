# Production Patterns and Best Practices

## 1. Error Handling and Resilience

### 1.1 Error Classification

```python
from enum import Enum
from typing import Callable, Any

class ErrorSeverity(Enum):
    TRANSIENT = "transient"      # Retry
    CIRCUIT_BREAKER = "breaker"  # Back off
    FATAL = "fatal"              # Fail fast
    DEGRADED = "degraded"        # Use fallback

class ErrorHandler:
    """
    Classify and handle different error types
    """
    
    ERROR_TAXONOMY = {
        # Transient errors (safe to retry)
        RateLimitError: {
            "severity": ErrorSeverity.TRANSIENT,
            "retry_delay": "exponential",
            "max_retries": 3,
            "backoff_factor": 2
        },
        TimeoutError: {
            "severity": ErrorSeverity.TRANSIENT,
            "retry_delay": "exponential",
            "max_retries": 5,
            "backoff_factor": 1.5
        },
        ConnectionError: {
            "severity": ErrorSeverity.TRANSIENT,
            "retry_delay": "exponential",
            "max_retries": 3,
            "backoff_factor": 2
        },
        
        # Circuit breaker (back off hard)
        ServiceUnavailableError: {
            "severity": ErrorSeverity.CIRCUIT_BREAKER,
            "cooldown": 60,  # seconds
            "failure_threshold": 5
        },
        
        # Fatal errors (fail fast)
        ValueError: {
            "severity": ErrorSeverity.FATAL,
            "action": "log_and_fail"
        },
        AuthenticationError: {
            "severity": ErrorSeverity.FATAL,
            "action": "alert_ops"
        },
        
        # Degraded service (use fallback)
        LowConfidenceError: {
            "severity": ErrorSeverity.DEGRADED,
            "fallback_strategy": "use_cached_result"
        }
    }
    
    def __init__(self):
        self.circuit_breaker_state = {}  # service -> state
        self.error_counts = {}  # service -> count
    
    def handle_error(self, error: Exception, context: dict) -> Any:
        """
        Route error to appropriate handler
        """
        error_type = type(error)
        config = self.ERROR_TAXONOMY.get(error_type, {})
        severity = config.get("severity", ErrorSeverity.FATAL)
        
        if severity == ErrorSeverity.TRANSIENT:
            return self._handle_transient(error, config, context)
        elif severity == ErrorSeverity.CIRCUIT_BREAKER:
            return self._handle_circuit_breaker(error, config, context)
        elif severity == ErrorSeverity.FATAL:
            return self._handle_fatal(error, config, context)
        elif severity == ErrorSeverity.DEGRADED:
            return self._handle_degraded(error, config, context)
    
    def _handle_transient(self, error, config, context):
        """Retry with exponential backoff"""
        max_retries = config.get("max_retries", 3)
        backoff_factor = config.get("backoff_factor", 2)
        
        for attempt in range(max_retries):
            wait_time = backoff_factor ** attempt
            print(f"Retry attempt {attempt + 1}/{max_retries}, waiting {wait_time}s")
            time.sleep(wait_time)
            
            try:
                return context["retry_func"]()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                continue
    
    def _handle_circuit_breaker(self, error, config, context):
        """
        Circuit breaker pattern:
        - Closed: Normal operation
        - Open: Reject requests
        - Half-open: Test if recovered
        """
        service = context.get("service", "unknown")
        
        # Track failures
        self.error_counts[service] = self.error_counts.get(service, 0) + 1
        
        failure_threshold = config.get("failure_threshold", 5)
        if self.error_counts[service] >= failure_threshold:
            # Open circuit
            self.circuit_breaker_state[service] = "open"
            cooldown = config.get("cooldown", 60)
            raise Exception(f"Circuit breaker open. Cooldown {cooldown}s")
    
    def _handle_fatal(self, error, config, context):
        """Fail fast, alert ops"""
        action = config.get("action", "log_and_fail")
        if action == "alert_ops":
            self._alert_operations(error, context)
        raise error
    
    def _handle_degraded(self, error, config, context):
        """Use fallback"""
        fallback = config.get("fallback_strategy")
        if fallback == "use_cached_result":
            return context.get("cached_result", None)
        return None
    
    def _alert_operations(self, error, context):
        """Send alert to ops team"""
        print(f"ALERT: {error} in {context}")
```

### 1.2 Timeout Management

```python
import signal
import threading

class TimeoutManager:
    """
    Handle timeouts at multiple levels:
    1. Request-level timeout
    2. Overall operation timeout
    3. Graceful degradation
    """
    
    def __init__(self):
        self.default_timeout = 30  # seconds
        self.hard_limit = 60  # absolute max
    
    def execute_with_timeout(
        self,
        func: Callable,
        args: tuple = (),
        timeout: int = None,
        fallback: Callable = None
    ) -> Any:
        """
        Execute function with timeout, use fallback if exceeded
        """
        timeout = timeout or self.default_timeout
        timeout = min(timeout, self.hard_limit)  # Safety cap
        
        result = {}
        exception = {}
        
        def target():
            try:
                result["value"] = func(*args)
            except Exception as e:
                exception["value"] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            if fallback:
                return fallback()
            else:
                raise TimeoutError(f"Operation exceeded {timeout}s")
        
        if "value" in exception:
            raise exception["value"]
        
        return result.get("value")
```

## 2. Monitoring and Observability

### 2.1 Metrics Collection

```python
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class MetricPoint:
    """Single metric measurement"""
    name: str
    value: float
    timestamp: datetime
    tags: dict  # labels: model, endpoint, status
    
class MetricsCollector:
    """
    Collect operational metrics for observability
    """
    
    CRITICAL_METRICS = {
        "latency_p50": {"threshold": 100, "unit": "ms"},       # 50th percentile
        "latency_p95": {"threshold": 500, "unit": "ms"},       # 95th percentile
        "latency_p99": {"threshold": 1000, "unit": "ms"},      # 99th percentile
        "error_rate": {"threshold": 0.01, "unit": "ratio"},    # <1%
        "token_usage": {"threshold": None, "unit": "tokens"},  # Track for cost
        "cache_hit_rate": {"threshold": None, "unit": "ratio"},
        "throughput": {"threshold": None, "unit": "req/s"},
    }
    
    def __init__(self):
        self.metrics = []
        self.aggregates = {}
    
    def record_latency(self, endpoint: str, latency_ms: float, status: str = "success"):
        """Record API latency"""
        self.metrics.append(MetricPoint(
            name="latency",
            value=latency_ms,
            timestamp=datetime.now(),
            tags={"endpoint": endpoint, "status": status}
        ))
    
    def record_error(self, endpoint: str, error_type: str):
        """Track errors"""
        self.metrics.append(MetricPoint(
            name="error_count",
            value=1,
            timestamp=datetime.now(),
            tags={"endpoint": endpoint, "error_type": error_type}
        ))
    
    def record_tokens(self, model: str, input_tokens: int, output_tokens: int):
        """Track token usage for cost"""
        self.metrics.append(MetricPoint(
            name="token_usage",
            value=input_tokens + output_tokens,
            timestamp=datetime.now(),
            tags={"model": model, "type": "input_output"}
        ))
    
    def compute_percentiles(self, metric_name: str, percentiles: list = [50, 95, 99]):
        """
        Compute latency percentiles
        
        Used for SLO/SLA monitoring
        """
        values = [m.value for m in self.metrics if m.name == metric_name]
        values.sort()
        
        results = {}
        for p in percentiles:
            index = int(len(values) * p / 100)
            results[f"p{p}"] = values[index]
        
        return results
    
    def check_slo(self) -> dict:
        """
        Check Service Level Objectives
        """
        slo_status = {}
        
        # Latency SLO
        latencies = self.compute_percentiles("latency")
        slo_status["latency_p99"] = {
            "current": latencies.get("p99"),
            "threshold": 1000,
            "healthy": latencies.get("p99", 0) < 1000
        }
        
        # Error rate SLO
        total_errors = sum(1 for m in self.metrics if m.name == "error_count")
        total_requests = len([m for m in self.metrics if m.name == "latency"])
        error_rate = total_errors / max(total_requests, 1)
        
        slo_status["error_rate"] = {
            "current": error_rate,
            "threshold": 0.01,
            "healthy": error_rate < 0.01
        }
        
        return slo_status
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        output = ""
        
        for metric in self.metrics:
            labels = ",".join([f'{k}="{v}"' for k, v in metric.tags.items()])
            output += f'{metric.name}{{{labels}}} {metric.value}\n'
        
        return output
```

### 2.2 Structured Logging

```python
import logging
import json

class StructuredLogger:
    """
    JSON structured logging for machine parsing
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log_request(self, request_id: str, endpoint: str, input_tokens: int):
        """Log incoming request"""
        self.logger.info(json.dumps({
            "level": "INFO",
            "event": "request_start",
            "request_id": request_id,
            "endpoint": endpoint,
            "input_tokens": input_tokens,
            "timestamp": datetime.now().isoformat()
        }))
    
    def log_response(self, request_id: str, status: str, latency_ms: float, output_tokens: int):
        """Log response"""
        self.logger.info(json.dumps({
            "level": "INFO",
            "event": "request_complete",
            "request_id": request_id,
            "status": status,
            "latency_ms": latency_ms,
            "output_tokens": output_tokens,
            "timestamp": datetime.now().isoformat()
        }))
    
    def log_error(self, request_id: str, error_type: str, error_message: str, traceback: str):
        """Log errors with full context"""
        self.logger.error(json.dumps({
            "level": "ERROR",
            "event": "request_error",
            "request_id": request_id,
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback,
            "timestamp": datetime.now().isoformat()
        }))
```

## 3. Caching Strategies

### 3.1 Multi-Level Caching

```python
from functools import lru_cache
import hashlib

class MultiLevelCache:
    """
    Three-level caching strategy:
    1. In-memory (fast, small)
    2. Redis (shared, medium)
    3. Disk (persistent, large)
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self.memory_cache = {}  # In-process
        self.redis_client = None  # Optional Redis
        self.disk_path = "./cache"  # Fallback disk cache
    
    def get(self, key: str) -> Any:
        """Retrieve from cache (try L1, L2, L3)"""
        
        # Level 1: In-memory
        if key in self.memory_cache:
            value, expiry = self.memory_cache[key]
            if datetime.now() < expiry:
                return value
            else:
                del self.memory_cache[key]
        
        # Level 2: Redis
        if self.redis_client:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    # Promote to L1
                    self.memory_cache[key] = (
                        cached,
                        datetime.now() + timedelta(seconds=self.ttl)
                    )
                    return cached
            except:
                pass
        
        # Level 3: Disk
        try:
            with open(f"{self.disk_path}/{key}", "r") as f:
                cached = json.load(f)
                # Promote to L2 and L1
                self.memory_cache[key] = (
                    cached,
                    datetime.now() + timedelta(seconds=self.ttl)
                )
                return cached
        except:
            pass
        
        return None
    
    def set(self, key: str, value: Any):
        """Store in all cache levels"""
        expiry = datetime.now() + timedelta(seconds=self.ttl)
        
        # L1: Memory
        self.memory_cache[key] = (value, expiry)
        
        # L2: Redis
        if self.redis_client:
            try:
                self.redis_client.setex(key, self.ttl, value)
            except:
                pass
        
        # L3: Disk
        try:
            with open(f"{self.disk_path}/{key}", "w") as f:
                json.dump(value, f)
        except:
            pass
```

### 3.2 Smart Caching Decisions

```python
class SmartCacheDecider:
    """
    Decide whether to cache based on:
    - Cost of computation
    - Cache invalidation frequency
    - Query patterns
    """
    
    def should_cache(self, query: str, metadata: dict) -> bool:
        """
        Heuristics for caching decision
        """
        
        # Don't cache user-specific queries (high invalidation)
        if metadata.get("user_id"):
            return False
        
        # Cache expensive operations
        if metadata.get("estimated_cost_cents", 0) > 1:  # >1 cent
            return True
        
        # Cache time-sensitive queries (news, weather)
        # based on cache invalidation frequency
        ttl = metadata.get("cache_ttl_seconds", 0)
        if ttl > 60:  # >1 minute is worth caching
            return True
        
        return False
    
    def compute_cache_key(self, query: str, model: str) -> str:
        """
        Generate cache key
        
        Include model in key since different models give different answers
        """
        key_material = f"{query}:{model}"
        return hashlib.sha256(key_material.encode()).hexdigest()[:16]
```

## 4. Rate Limiting

### 4.1 Token Bucket Algorithm

```python
import time
from threading import Lock

class TokenBucket:
    """
    Token bucket algorithm for rate limiting
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        capacity: Max tokens (requests)
        refill_rate: Tokens per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = Lock()
    
    def _refill(self):
        """Top up tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens"""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_until_available(self, tokens: int = 1):
        """Block until tokens available"""
        while not self.consume(tokens):
            time.sleep(0.01)

# Example: Rate limit to 100 requests per minute
limiter = TokenBucket(capacity=100, refill_rate=100/60)
```

### 4.2 Distributed Rate Limiting

```python
class DistributedRateLimiter:
    """
    Rate limiting across multiple servers using Redis
    """
    
    def __init__(self, redis_client, key_prefix: str = "rate_limit"):
        self.redis = redis_client
        self.key_prefix = key_prefix
    
    def check_limit(self, user_id: str, limit_per_minute: int = 60) -> bool:
        """
        Check if user exceeded rate limit
        
        Returns: True if request allowed, False if rate limited
        """
        key = f"{self.key_prefix}:{user_id}:{int(time.time() // 60)}"
        
        try:
            current = self.redis.incr(key)
            
            # Set expiry on first request
            if current == 1:
                self.redis.expire(key, 60)
            
            return current <= limit_per_minute
        except:
            # Fail open (allow) if Redis fails
            return True
```

## 5. Deployment Patterns

### 5.1 Blue-Green Deployment

```python
class BlueGreenDeployment:
    """
    Zero-downtime deployments with instant rollback
    """
    
    def __init__(self):
        self.blue_version = "v1"
        self.green_version = "v2"
        self.active_version = "blue"
    
    def deploy_new_version(self, new_model_path: str, test_percentage: float = 0.1):
        """
        1. Deploy new version (green)
        2. Route X% of traffic for testing
        3. Monitor metrics
        4. Cutover if healthy
        """
        
        # Step 1: Deploy green
        self._deploy_version("green", new_model_path)
        
        # Step 2: Canary test (10% traffic)
        self._route_traffic(
            blue_pct=1 - test_percentage,
            green_pct=test_percentage
        )
        
        # Step 3: Monitor
        metrics = self._collect_metrics(duration_seconds=300)
        
        # Step 4: Decision
        if self._metrics_healthy(metrics):
            self._route_traffic(blue_pct=0, green_pct=1)  # 100% green
            self.active_version = "green"
            print("✓ Deployment successful")
        else:
            self._rollback()
            print("✗ Rolled back to blue")
    
    def _rollback(self):
        """Instant rollback"""
        self._route_traffic(blue_pct=1, green_pct=0)
        self.active_version = "blue"
    
    def _metrics_healthy(self, metrics: dict) -> bool:
        """Check if canary metrics are healthy"""
        return (
            metrics["error_rate"] < 0.01 and
            metrics["latency_p99"] < 1000 and
            metrics["throughput"] > 50
        )
```

## Key Takeaways

1. **Error handling is layered**: Transient, circuit breaker, fatal, degraded
2. **Observability saves debugging**: Metrics, logs, traces
3. **Caching reduces costs**: Multi-level caching for different latency profiles
4. **Rate limiting prevents cascades**: Protect downstream services
5. **Blue-green enables safety**: Zero-downtime deployments with rollback

## References

- SRE Handbook: https://sre.google/resources/
- Prometheus Metrics: https://prometheus.io/docs/
- Circuit Breaker Pattern: https://martinfowler.com/bliki/CircuitBreaker.html
