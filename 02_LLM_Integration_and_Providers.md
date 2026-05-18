# LLM Integration and Providers

## 1. Provider Integration Taxonomy

### 1.1 Provider Categories

```
LLM Providers
├── Commercial APIs
│   ├── OpenAI (GPT-3.5, GPT-4, GPT-4V)
│   ├── Anthropic (Claude 2, Claude 3)
│   ├── Google (Gemini, PaLM)
│   ├── Cohere
│   └── Mistral
├── Open Source (Self-hosted)
│   ├── Meta Llama
│   ├── Mistral 7B/13B
│   ├── Phi
│   └── Others
└── Hybrid
    ├── Groq (Inference engine)
    ├── Replicate
    └── HuggingFace Inference API
```

## 2. OpenAI Integration

### 2.1 API Structure

**Endpoint Architecture**:
```
Base URL: https://api.openai.com/v1

Completions (Legacy):
POST /completions
Body: {
    "model": "text-davinci-003",
    "prompt": "...",
    "temperature": 0.7,
    "max_tokens": 256
}

Chat Completions (Current):
POST /chat/completions
Body: {
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "temperature": 0.7,
    "functions": [...],  // Optional function calling
    "tools": [...]       // Tool use (GPT-4 Turbo+)
}
```

### 2.2 Cost Model and Token Economics

**Token Counting Algorithm**:
- Typically: 1 token ≈ 4 characters ≈ 0.75 words
- Actual: Use `tiktoken` library for accurate counts

```python
import tiktoken

def count_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Pricing (as of 2024)
OPENAI_PRICING = {
    "gpt-4": {
        "input": 0.03 / 1000,      # $30 per 1M tokens
        "output": 0.06 / 1000,     # $60 per 1M tokens
    },
    "gpt-3.5-turbo": {
        "input": 0.0005 / 1000,    # $0.50 per 1M tokens
        "output": 0.0015 / 1000,   # $1.50 per 1M tokens
    }
}

def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = OPENAI_PRICING[model]
    input_cost = input_tokens * pricing["input"]
    output_cost = output_tokens * pricing["output"]
    return input_cost + output_cost
```

### 2.3 Function Calling (Deprecated) vs. Tools (Current)

**Function Calling** (Older approach):
```json
{
  "functions": [
    {
      "name": "get_weather",
      "description": "Get weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City name"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
  ]
}
```

Response includes `function_call` field.

**Tools** (Current standard):
```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": { ... }
      }
    }
  ]
}
```

Response includes `tool_calls` array in `message.content`.

### 2.4 Context Window Management

```
GPT-4 Turbo: 128K context
GPT-4: 8K or 32K
GPT-3.5-turbo: 4K or 16K

Strategy: Sliding Window with Overlap
┌─────────────────────────────────────────┐
│ System Prompt (always included)         │
├─────────────────────────────────────────┤
│ Conversation History (with truncation)  │
├─────────────────────────────────────────┤
│ Retrieved Context (if RAG enabled)      │
├─────────────────────────────────────────┤
│ Current User Query                      │
├─────────────────────────────────────────┤
│ Reserved Buffer for Response            │
└─────────────────────────────────────────┘

Total ≤ context_limit - reserved_for_output
```

**Truncation Strategies**:
1. **Summarization**: Compress old messages
2. **Sliding Window**: Keep most recent N messages
3. **Importance Sampling**: Keep high-relevance messages
4. **Hierarchical**: Recursive summaries

## 3. Anthropic Claude Integration

### 3.1 API Differences from OpenAI

```python
# Anthropic uses different message format
{
    "messages": [
        {
            "role": "user",
            "content": "..."
        }
    ],
    "system": "You are a helpful assistant",  # Separate field
    "model": "claude-3-opus-20240229",
    "max_tokens": 1024
}
```

**Key Differences**:
- System prompt is separate from messages
- Stricter message ordering (user → assistant → user)
- No built-in function calling until Claude 3
- Better reasoning capabilities (longer thinking)

### 3.2 Constitutional AI (CAI)

Anthropic uses Constitutional AI for safety:

```
Generation → Critique → Revision → Filtering

Principles applied:
- Harmlessness: Avoid harmful content
- Honesty: Acknowledge uncertainty
- Helpfulness: Provide accurate information
```

**Practical implication**: Claude responses are often more verbose but safer.

### 3.3 Long Context (200K Tokens)

```python
# Leveraging 200K context for document analysis

def analyze_large_document(document: str, query: str, model: str = "claude-3-opus-20240229"):
    """
    Process documents up to ~200K tokens efficiently.
    
    Best practices:
    1. Put entire document in user message
    2. Ask specific questions to guide extraction
    3. Use tags for structural clarity
    """
    messages = [
        {
            "role": "user",
            "content": f"""
<document>
{document}
</document>

Query: {query}

Provide structured analysis.
"""
        }
    ]
    
    return client.messages.create(
        model=model,
        max_tokens=2000,
        messages=messages
    )
```

## 4. Local and Open-Source Models

### 4.1 Ollama Integration

**Architecture**:
```
Application → Ollama Server (HTTP API) → Model Engine
```

```python
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama2:70b",
    base_url="http://localhost:11434",
    temperature=0.7,
    top_p=0.9
)

# Models typically available:
# - llama2 (7b, 13b, 70b)
# - mistral (7b)
# - neural-chat (7b)
# - dolphin-mixtral (8x7b)
```

### 4.2 Model Selection Criteria

| Criterion | Consideration |
|-----------|--------------|
| **Context Length** | Llama2: 4K, Mistral: 8K, Llama2-200K: 200K |
| **Inference Speed** | Smaller = faster (7B vs 70B is ~10x) |
| **Quality** | Larger models generally better but slower |
| **Quantization** | Q4/Q5: 30-40% size, 5-10% quality loss |
| **Vram Required** | Roughly 2GB per billion parameters (FP16) |

**Quantization Impact**:
```
Full Precision (FP32): 70B model = 280GB
FP16: 70B model = 140GB  
Int8: 70B model = 70GB
Int4 (Q4): 70B model = 35GB (with ~5% quality loss)

Latency Impact:
FP32: 100ms per token
Q4: 80ms per token (20% improvement, storage advantage)
```

### 4.3 Inference Optimization with Ollama

```python
# Pull model with specific settings
ollama pull llama2:70b

# Run with CUDA GPU support
OLLAMA_NUM_GPU=1 ollama serve

# Monitor performance
# - Check GPU utilization: nvidia-smi
# - Monitor first token latency (time to first token)
# - Monitor token generation speed (tokens/sec)
```

## 5. Multi-Provider Strategy

### 5.1 Provider Fallback Chain

```python
from langchain.llms import OpenAI, Anthropic
from langchain.base_language import BaseLanguageModel

class ProviderFallback:
    def __init__(self, providers: List[BaseLanguageModel]):
        self.providers = providers
        self.current_index = 0
    
    async def invoke(self, input: str, **kwargs) -> str:
        """Attempt providers in order, fallback on failure"""
        for i, provider in enumerate(self.providers):
            try:
                result = await provider.ainvoke(input, **kwargs)
                return result
            except Exception as e:
                if i == len(self.providers) - 1:
                    raise
                # Log and continue to next provider
                print(f"Provider failed: {e}, trying next...")
                continue

# Usage
fallback = ProviderFallback([
    ChatOpenAI(model="gpt-4"),
    ChatAnthropic(model="claude-3-opus"),
    ChatCohere(),
    Ollama(model="llama2:70b")
])

response = await fallback.invoke("Complex reasoning task")
```

### 5.2 Cost Optimization Strategy

```python
class CostOptimizedLLM:
    """Route requests to minimize cost while maintaining quality"""
    
    def __init__(self):
        self.gpt35 = ChatOpenAI(model="gpt-3.5-turbo")  # Cheap
        self.gpt4 = ChatOpenAI(model="gpt-4")           # Expensive
        self.claude = ChatAnthropic(model="claude-3-opus")  # Medium
    
    def should_use_gpt4(self, task_complexity: str) -> bool:
        """Complex reasoning tasks need GPT-4"""
        complex_keywords = ["reason", "analyze", "explain", "compare"]
        return any(kw in task_complexity.lower() for kw in complex_keywords)
    
    async def invoke(self, input: str, complexity: str = "low"):
        if self.should_use_gpt4(complexity):
            return await self.gpt4.ainvoke(input)
        else:
            return await self.gpt35.ainvoke(input)
```

## 6. Rate Limiting and Quota Management

### 6.1 Token Bucket Algorithm

```python
from collections import deque
from time import time

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        """
        capacity: Max tokens available
        refill_rate: Tokens per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time()
    
    def _refill(self):
        now = time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now
    
    def consume(self, tokens: int) -> bool:
        """Attempt to consume tokens, return success"""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def wait_until_available(self, tokens: int):
        """Block until tokens available"""
        while not self.consume(tokens):
            sleep(0.01)
```

### 6.2 Rate Limit Handling

```python
class RateLimitHandler:
    """Handle OpenAI's rate limits gracefully"""
    
    MAX_RETRIES = 3
    INITIAL_WAIT = 1  # seconds
    
    async def call_with_retry(self, llm: BaseLLM, prompt: str) -> str:
        wait_time = self.INITIAL_WAIT
        
        for attempt in range(self.MAX_RETRIES):
            try:
                return await llm.ainvoke(prompt)
            except RateLimitError as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                
                # Extract retry_after from error if available
                retry_after = getattr(e, 'retry_after', None)
                wait_time = retry_after or (wait_time * 2)
                
                print(f"Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
```

## 7. Custom Provider Implementation

### 7.1 Implementing Custom LLM

```python
from langchain_core.language_models.llm import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult

class CustomLLM(LLM):
    """Template for implementing custom LLM"""
    
    model_name: str
    api_endpoint: str
    api_key: str
    
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Make HTTP request to custom model"""
        response = requests.post(
            self.api_endpoint,
            json={
                "prompt": prompt,
                "model": self.model_name,
                "stop": stop,
                **kwargs
            },
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        
        if response.status_code != 200:
            raise ValueError(f"API error: {response.text}")
        
        return response.json()["text"]
    
    @property
    def _identifying_params(self) -> dict:
        """For serialization and debugging"""
        return {
            "model_name": self.model_name,
            "api_endpoint": self.api_endpoint,
        }
```

## 8. Production Considerations

### 8.1 Authentication and Secret Management

```python
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Environment-based (recommended for production)
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4"
)

# Using secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)
import boto3

def get_api_key():
    client = boto3.client('secretsmanager')
    secret = client.get_secret_value(SecretId='openai-api-key')
    return secret['SecretString']

llm = ChatOpenAI(api_key=get_api_key())
```

### 8.2 Observability: Logging Provider Calls

```python
from langchain.callbacks.tracers import LangChainTracer
from langsmith import Client

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
tracer = LangChainTracer(client=client, project_name="my-app")

# Track all LLM calls
result = llm.invoke(
    prompt,
    callbacks=[tracer],
    tags=["production", "user-id-123"]
)
```

## Key Takeaways

1. **Provider selection is fundamental**: Cost, speed, capability trade-offs
2. **Token counting is essential**: For cost estimation and context management
3. **Fallback strategies are critical**: For resilience in production
4. **Context windows vary dramatically**: Design with provider constraints in mind
5. **Monitoring is non-negotiable**: Track API usage, failures, and performance

## References

- OpenAI API Documentation: https://platform.openai.com/docs/
- Anthropic API Documentation: https://docs.anthropic.com/
- Ollama Documentation: https://github.com/ollama/ollama
- LangChain LLM Integration: https://python.langchain.com/docs/modules/model_io/
