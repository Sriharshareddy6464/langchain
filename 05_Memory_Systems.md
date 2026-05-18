# Memory Systems in LangChain

## 1. Memory Fundamentals

### 1.1 Memory Architecture

```
Conversation Flow:
┌────────────────────────────────────────────┐
│ User Input                                 │
└────────────────┬─────────────────────────────┘
                 │
                 ↓
        ┌────────────────┐
        │ Memory Retrieval│
        │ (Get context)  │
        └────────┬───────┘
                 │
                 ↓
        ┌────────────────────────┐
        │ LLM Processing         │
        │ (Input + Memory)       │
        └────────┬───────────────┘
                 │
                 ↓
        ┌────────────────────────┐
        │ Memory Update          │
        │ (Save new context)     │
        └────────┬───────────────┘
                 │
                 ↓
        ┌────────────────┐
        │ User Response  │
        └────────────────┘
```

### 1.2 Memory Types

```
Buffer Memory
├─ ConversationBufferMemory: Full history
├─ ConversationSummaryMemory: Summarized history
├─ ConversationEntityMemory: Track entities
└─ ConversationKGMemory: Knowledge graph

Window Memory
├─ ConversationBufferWindowMemory: Last N messages
└─ ConversationTokenBufferMemory: Last N tokens

Vector Memory (Semantic)
├─ Pinecone
├─ Chroma
├─ Weaviate
└─ FAISS
```

## 2. Buffer Memory Systems

### 2.1 Simple Buffer Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import ChatOpenAI

# Create memory
memory = ConversationBufferMemory()

# Store conversation
memory.chat_memory.add_user_message("Hi, I'm Alice")
memory.chat_memory.add_ai_message("Nice to meet you, Alice!")
memory.chat_memory.add_user_message("What's my name?")

# Retrieve formatted memory
context = memory.load_memory_variables({})  
# Returns: {"history": "Human: Hi, I'm Alice\nAI: ..."}
```

### 2.2 Summary Memory

```python
from langchain.memory import ConversationSummaryMemory

class SummaryMemorySystem:
    """
    Compress conversation history with periodic summarization
    
    Approach:
    1. Maintain fixed buffer of recent messages
    2. Periodically summarize older messages
    3. Combine summary + buffer for context
    """
    
    def __init__(self, llm, buffer_size: int = 5):
        self.llm = llm
        self.buffer_size = buffer_size
        self.memory = ConversationSummaryMemory(
            llm=llm,
            buffer="\n"
        )
    
    def add_message(self, role: str, content: str):
        if role == "user":
            self.memory.chat_memory.add_user_message(content)
        else:
            self.memory.chat_memory.add_ai_message(content)
        
        # Trigger summarization if buffer exceeds limit
        if len(self.memory.chat_memory.messages) > self.buffer_size:
            self._trigger_summarization()
    
    def _trigger_summarization(self):
        """Summarize older messages"""
        messages = self.memory.chat_memory.messages
        
        # Keep recent messages
        recent = messages[-self.buffer_size:]
        old = messages[:-self.buffer_size]
        
        # Summarize old messages
        old_text = "\n".join([m.content for m in old])
        summary_prompt = f"Summarize this conversation:\n{old_text}"
        summary = self.llm.invoke(summary_prompt)
        
        # Rebuild memory
        self.memory.chat_memory.messages = recent
        # Prepend summary (implementation depends on LangChain version)
    
    def get_context(self) -> str:
        return self.memory.load_memory_variables({}).get("history", "")
```

### 2.3 Token-Aware Window Memory

```python
import tiktoken

class TokenWindowMemory:
    """
    Keep conversation within token budget
    
    Algorithm:
    1. Count tokens in each message
    2. Remove oldest messages when limit exceeded
    3. Ensure at least N recent messages
    """
    
    def __init__(self, token_limit: int = 2000, model: str = "gpt-3.5-turbo"):
        self.token_limit = token_limit
        self.encoding = tiktoken.encoding_for_model(model)
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._enforce_token_limit()
    
    def _enforce_token_limit(self):
        """Remove oldest messages if token count exceeds limit"""
        total_tokens = self._count_tokens()
        
        # Keep removing oldest messages until under limit
        while total_tokens > self.token_limit and len(self.messages) > 2:
            self.messages.pop(0)
            total_tokens = self._count_tokens()
    
    def _count_tokens(self) -> int:
        """Count total tokens in all messages"""
        total = 0
        for msg in self.messages:
            tokens = self.encoding.encode(msg["content"])
            total += len(tokens)
        return total
    
    def get_context(self) -> str:
        """Format messages as context"""
        context = ""
        for msg in self.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        return context
    
    def get_token_count(self) -> int:
        return self._count_tokens()
```

## 3. Entity-Based Memory

### 3.1 Entity Tracking

```python
class EntityMemory:
    """
    Track important entities (people, places, concepts) across conversation
    
    Algorithm:
    1. Extract entities from each message
    2. Update entity knowledge base
    3. Reference entities when relevant
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.entities = {}  # entity_name -> properties
        self.messages = []
    
    def add_message(self, content: str):
        # Extract entities
        entities = self._extract_entities(content)
        
        # Update entity knowledge base
        for entity_name, properties in entities.items():
            if entity_name not in self.entities:
                self.entities[entity_name] = []
            self.entities[entity_name].append(properties)
        
        self.messages.append(content)
    
    def _extract_entities(self, text: str) -> dict:
        """Use LLM to extract entities"""
        prompt = f"""
Extract entities and their properties from this text:
{text}

Format as JSON:
{{
    "entity_name": {{"property1": "value1", "property2": "value2"}}
}}

JSON:"""
        response = self.llm.invoke(prompt)
        try:
            return json.loads(response)
        except:
            return {}
    
    def get_entity_context(self) -> str:
        """Get current understanding of all tracked entities"""
        context = "Known Entities:\n"
        for entity, properties_list in self.entities.items():
            context += f"\n{entity}:\n"
            # Consolidate properties (latest takes precedence)
            consolidated = {}
            for props in properties_list:
                consolidated.update(props)
            
            for key, value in consolidated.items():
                context += f"  - {key}: {value}\n"
        
        return context
```

## 4. Vector-Based Memory (Semantic)

### 4.1 Semantic Similarity Memory

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

class SemanticMemory:
    """
    Use vector similarity to retrieve relevant past conversations
    
    Algorithm:
    1. Embed all previous messages
    2. On new query, find semantically similar messages
    3. Use top-K similar messages as context
    """
    
    def __init__(self, embedding_model=None):
        self.embeddings = embedding_model or OpenAIEmbeddings()
        self.vector_store = None
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        
        # Rebuild vector store
        self._rebuild_vector_store()
    
    def _rebuild_vector_store(self):
        """Rebuild FAISS index with all messages"""
        if not self.messages:
            return
        
        docs = [
            Document(
                page_content=msg["content"],
                metadata={"role": msg["role"], "index": i}
            )
            for i, msg in enumerate(self.messages)
        ]
        
        self.vector_store = FAISS.from_documents(
            docs,
            self.embeddings
        )
    
    def retrieve_relevant_context(self, query: str, k: int = 3) -> str:
        """Get most relevant past messages"""
        if not self.vector_store:
            return ""
        
        docs = self.vector_store.similarity_search(query, k=k)
        
        context = "Relevant past context:\n"
        for doc in docs:
            role = doc.metadata.get("role", "unknown")
            context += f"{role}: {doc.page_content}\n"
        
        return context
```

### 4.2 Retrieval-Augmented Memory

```python
class RAGMemory:
    """
    Combine semantic retrieval with summarization for optimal memory usage
    
    Strategy:
    1. Store conversation in vector DB
    2. Retrieve relevant snippets for current query
    3. Summarize retrieved snippets
    4. Combine with recent message buffer
    """
    
    def __init__(self, llm, embeddings, vector_store, buffer_size: int = 5):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.buffer_size = buffer_size
        self.recent_messages = []
    
    def add_message(self, role: str, content: str):
        # Add to recent buffer
        self.recent_messages.append({"role": role, "content": content})
        
        # Trim buffer
        if len(self.recent_messages) > self.buffer_size:
            self.recent_messages = self.recent_messages[-self.buffer_size:]
        
        # Store in vector DB
        self._store_in_db(role, content)
    
    def _store_in_db(self, role: str, content: str):
        """Add message to vector store"""
        from langchain.schema import Document
        doc = Document(
            page_content=content,
            metadata={"role": role, "timestamp": str(datetime.now())}
        )
        self.vector_store.add_documents([doc])
    
    def build_context(self, query: str, k: int = 3) -> str:
        """Build context combining retrieval + recent messages"""
        # Retrieve relevant snippets
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        
        # Summarize if too much retrieved
        retrieved_text = "\n".join([
            f"{doc.metadata.get('role', 'unknown')}: {doc.page_content}"
            for doc in relevant_docs
        ])
        
        if len(retrieved_text) > 1000:
            summary_prompt = f"Summarize this conversation:\n{retrieved_text}"
            retrieved_text = self.llm.invoke(summary_prompt)
        
        # Combine with recent messages
        recent_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.recent_messages
        ])
        
        return f"""Previous context:
{retrieved_text}

Recent messages:
{recent_text}"""
```

## 5. Advanced Memory Patterns

### 5.1 Hierarchical Memory

```python
class HierarchicalMemory:
    """
    Multi-level memory: Interaction → Session → Long-term
    
    Levels:
    1. Working Memory: Current conversation (recent messages)
    2. Episode Memory: This session's highlights (summarized)
    3. Semantic Memory: Long-term facts and patterns
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.working = []  # Recent messages
        self.episodes = []  # Session summaries
        self.semantic = {}  # Long-term facts
    
    def add_interaction(self, user_msg: str, ai_msg: str):
        # Add to working memory
        self.working.append({"user": user_msg, "ai": ai_msg})
        
        # Periodically consolidate to episode memory
        if len(self.working) >= 10:
            self._consolidate_to_episode()
        
        # Extract key facts to semantic memory
        self._extract_facts(user_msg, ai_msg)
    
    def _consolidate_to_episode(self):
        """Create session summary"""
        conversation = "\n".join([
            f"User: {i['user']}\nAI: {i['ai']}"
            for i in self.working
        ])
        
        prompt = f"Summarize this conversation session:\n{conversation}"
        summary = self.llm.invoke(prompt)
        
        self.episodes.append({"summary": summary, "timestamp": datetime.now()})
        self.working = []  # Clear working memory
    
    def _extract_facts(self, user_msg: str, ai_msg: str):
        """Extract and store key facts"""
        prompt = f"""
Extract key facts from this exchange:
User: {user_msg}
AI: {ai_msg}

List facts as 'key: value' format.
"""
        facts_text = self.llm.invoke(prompt)
        
        # Parse and store
        for line in facts_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                self.semantic[key.strip()] = value.strip()
```

## 6. Memory Performance

### 6.1 Memory Optimization Strategies

| Strategy | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Buffer** | Simple, all info | Unbounded growth | Short sessions |
| **Window** | Fixed size | Recent info only | Long conversations |
| **Summary** | Compressed | Info loss | Very long talks |
| **Vector** | Semantic | Slow retrieval | Complex context |
| **Hierarchical** | Balanced | Complex | Complex reasoning |

### 6.2 Benchmarking Memory Systems

```python
import time

class MemoryBenchmark:
    def __init__(self, memory_system):
        self.memory = memory_system
    
    def benchmark_add(self, num_messages: int):
        """Time message addition"""
        start = time.time()
        for i in range(num_messages):
            self.memory.add_message("user", f"Message {i}")
        elapsed = time.time() - start
        
        return elapsed / num_messages  # Time per message
    
    def benchmark_retrieval(self, query: str, iterations: int = 100):
        """Time context retrieval"""
        start = time.time()
        for _ in range(iterations):
            _ = self.memory.get_context()
        elapsed = time.time() - start
        
        return elapsed / iterations  # Time per retrieval
```

## Key Takeaways

1. **Buffer memory is simple**: Good for short conversations
2. **Summaries compress**: Reduce tokens but lose detail
3. **Windows are practical**: Balance recency and cost
4. **Vectors enable semantics**: Find relevant context efficiently
5. **Hierarchies scale**: Different memory levels for different timescales

## References

- Memory Documentation: https://python.langchain.com/docs/modules/memory/
- Conversation Patterns: https://www.promptingguide.ai/
