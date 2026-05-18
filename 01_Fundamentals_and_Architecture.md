# LangChain Fundamentals and Architecture

## 1. Core Concepts

### 1.1 LangChain Philosophy

LangChain is a framework for developing applications powered by language models. It operates on three fundamental principles:

1. **Composability** - Applications are built through composition of components (models, prompts, memory, tools)
2. **Abstraction** - Provides unified interfaces across heterogeneous LLM providers
3. **Extensibility** - Components are swappable and custom implementations are first-class citizens

### 1.2 System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                    │
│  (Agents, Chains, RAG, Custom Workflows)               │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Composition Layer                          │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐             │
│  │  Chains    │ │ Memory   │ │ Tools    │             │
│  └────────────┘ └──────────┘ └──────────┘             │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│             Primitive Layer                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │ Language     │ │ Embeddings   │ │ Output       │  │
│  │ Models       │ │ Models       │ │ Parsers      │  │
│  └──────────────┘ └──────────────┘ └──────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           Integration Layer                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │ LLM APIs     │ │ Vector DBs   │ │ External     │  │
│  │              │ │              │ │ APIs         │  │
│  └──────────────┘ └──────────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 2. Core Components

### 2.1 Language Models (LLMs)

**Definition**: A function that maps token sequences to probability distributions over vocabularies.

$$P(x_{t+1} | x_1, x_2, ..., x_t; \theta)$$

**LangChain Abstraction**:
- Unified interface across OpenAI, Anthropic, Cohere, local models, etc.
- Streaming vs. non-streaming responses
- Cost tracking and rate limiting
- Retry logic with exponential backoff

**Key Considerations**:
- **Temperature** (τ): Controls sampling randomness. Range [0, 2]
  - τ → 0: Deterministic, greedy selection
  - τ = 1: Standard softmax sampling
  - τ → ∞: Uniform distribution
  
- **Top-K Sampling**: Only sample from top-K highest probability tokens
  - Prevents low-probability "tail" tokens
  - Typical K: 40-50
  
- **Top-P (Nucleus) Sampling**: Sample from smallest set of tokens with cumulative probability ≥ p
  - More adaptive than top-K
  - Typical p: 0.9-0.95
  
- **Presence/Frequency Penalties**: Reduce probability of repeated tokens
  - $p_i = p_i \cdot \exp(-\alpha \cdot \text{count}_i)$
  - Common α: 0.1-1.0

### 2.2 Prompt Templates

**Definition**: Structured format for constructing inputs with placeholders for dynamic content.

```python
# Conceptual template
template = """
System: {system_context}
User Query: {user_input}
Relevant Context: {retrieved_context}
Previous Exchanges: {chat_history}

Response:
"""
```

**Template Variables** have semantics:
- Input variables: From user/runtime
- Partial variables: Set at initialization
- Output variables: Extracted from LLM response

**Validation**: 
- All required variables must be bound before execution
- Type checking for complex templates
- Length constraints to prevent token overflow

### 2.3 Output Parsers

**Purpose**: Convert unstructured LLM outputs to structured formats.

**Parsing Strategies**:

1. **Regex Parsing**: Pattern matching for specific structures
   - Fast but brittle
   - Fails on whitespace variations

2. **LLM-based Parsing**: Use another LLM call to parse
   - More robust
   - Adds latency and cost

3. **Pydantic Integration**: Validate against schema
   ```python
   class Response(BaseModel):
       reasoning: str
       answer: str
       confidence: float  # 0-1
   ```

4. **Function Calling**: Use native API function calling when available
   - Reduces parsing errors
   - Faster than text parsing
   - Model must support function calling

**Repair Strategies**:
- When parsing fails, attempt to fix invalid output
- Re-prompt with correction instructions
- Fallback to partial parsing

## 3. Execution Model

### 3.1 Runnable Protocol

LangChain 0.1+ uses a universal `Runnable` interface:

```
Input → Runnable → Output
```

**Core Methods**:
- `invoke(input)`: Synchronous execution
- `batch(inputs)`: Batch processing
- `stream(input)`: Streaming output
- `ainvoke(input)`: Async execution

**Composition**:
- Pipes (`|`): Sequential composition
- Maps: Parallel branches
- Fallbacks: Error recovery
- Recursion: Self-referential execution

### 3.2 Execution Flow with Error Handling

```
1. Input Validation
   ↓
2. Template Rendering with Variable Binding
   ↓
3. Token Counting & Budget Check
   ↓
4. LLM API Call (with retry logic)
   ↓
5. Response Parsing
   ├─ Success → Return structured output
   └─ Failure → Attempt repair or fallback
```

**Retry Strategy**:
```
max_retries = 3
for attempt in range(max_retries):
    try:
        response = call_llm()
        return response
    except RateLimitError:
        wait(exponential_backoff(attempt))
    except (TokenLimitError, ValidationError) as e:
        return fallback_handler(e)
```

## 4. State Management

### 4.1 Run Context

Each execution maintains context:
- **run_id**: Unique execution identifier (UUID)
- **parent_run_id**: For nested execution tracking
- **metadata**: Custom key-value pairs for observability
- **tags**: For filtering and analytics

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    run_name="main_query",
    tags=["production", "user_id:12345"],
    metadata={"session_id": "abc", "model": "gpt-4"},
    max_tokens=2000
)
```

### 4.2 Callback System

Callbacks hook into execution lifecycle:

```
START_EVENT → RUNNING → STREAM_EVENT → END_EVENT → ERROR_EVENT
```

**Callback Types**:
- `on_llm_start(serialized, prompts, **kwargs)`
- `on_llm_new_token(token, **kwargs)`
- `on_llm_end(result, **kwargs)`
- `on_llm_error(error, **kwargs)`
- Custom callbacks for chains, tools, etc.

## 5. Provider Abstraction

### 5.1 LLM Base Class Interface

```python
class BaseLLM(BaseLanguageModel):
    def generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Callbacks] = None,
        **kwargs
    ) -> LLMResult:
        """Generate responses for multiple prompts"""
        pass
    
    def invoke(self, input: str, **kwargs) -> str:
        """Simplified single-prompt interface"""
        pass
```

### 5.2 Provider-Specific Considerations

| Provider | Token Limit | Cost Model | Special Features |
|----------|-------------|-----------|------------------|
| OpenAI | 128K (GPT-4T) | Per 1M tokens | Function calling, Vision |
| Anthropic | 200K | Per 1M tokens | Long context, Constitutional AI |
| Cohere | 4K-100K | Per 1M tokens | Rerank, Embed endpoints |
| Local (Ollama) | Configurable | None | Privacy, Full control |

## 6. Design Patterns

### 6.1 Factory Pattern for Model Creation

```python
def get_llm(provider: str, **kwargs) -> BaseLLM:
    providers = {
        "openai": ChatOpenAI,
        "anthropic": ChatAnthropic,
        "cohere": ChatCohere,
    }
    return providers[provider](**kwargs)
```

### 6.2 Strategy Pattern for Parsing

```python
class OutputStrategy(ABC):
    @abstractmethod
    def parse(self, text: str) -> Any:
        pass

class JSONStrategy(OutputStrategy):
    def parse(self, text: str) -> dict:
        return json.loads(text)

class PydanticStrategy(OutputStrategy):
    def parse(self, text: str) -> BaseModel:
        return self.schema.parse_raw(text)
```

### 6.3 Chain of Responsibility for Error Handling

```python
Handler → (Success) → Return
   ↓
   (Failure)
   ↓
Next Handler → Continue or Fail
```

## 7. Threading and Concurrency

### 7.1 Batch Processing

LangChain supports:
- **Sequential batching**: Process in order
- **Parallel batching**: Concurrent requests (respecting rate limits)
- **Streaming batching**: Return results as available

**Rate Limit Aware**:
- Tracks API call counts
- Implements token budgets
- Automatic throttling

### 7.2 Async Support

```python
# Concurrent execution
results = await asyncio.gather(*[
    llm.ainvoke(prompt) for prompt in prompts
])

# Streaming with async
async for token in llm.astream(prompt):
    yield token
```

## 8. Reference Architecture for LLM Application

```
┌──────────────────────────────────────────┐
│        User Interface Layer              │
│  (Web UI, CLI, API Endpoint)             │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│        Request Processing                 │
│  (Auth, Rate Limiting, Validation)       │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│        Agent/Chain Execution             │
│  (Decision making, tool selection)       │
└──────────────┬───────────────────────────┘
               │
    ┌──────────┼──────────┬──────────┐
    │          │          │          │
    ↓          ↓          ↓          ↓
  Memory    Retrieval   Tools      LLM
  System     System      System    System
    │          │          │          │
    └──────────┼──────────┴──────────┘
               │
┌──────────────▼───────────────────────────┐
│        Response Processing                │
│  (Parsing, Formatting, Logging)          │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│        Observability Layer                │
│  (Metrics, Tracing, Error Reporting)     │
└──────────────────────────────────────────┘
```

## Key Takeaways

1. **Composability is fundamental**: Every component implements a unified interface
2. **Provider abstraction reduces friction**: Switch between models with minimal code changes
3. **Execution model is transparent**: Callbacks and run context enable full observability
4. **Error handling is critical**: Production apps need robust retry and fallback strategies
5. **Batch/async processing is essential**: For scaling beyond single-request latency

## References

- LangChain Documentation: https://python.langchain.com/docs/
- LCEL (LangChain Expression Language): https://python.langchain.com/docs/expression_language/
- Runnable Protocol: https://api.python.langchain.com/en/latest/runnables/
