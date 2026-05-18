# Chains and Composition Patterns

## 1. Chain Fundamentals

### 1.1 What is a Chain?

**Definition**: A sequence of computational steps where output of one becomes input to the next.

```
Input → Component1 → Component2 → Component3 → Output
```

**Chain vs. Runnable**:
- Chain: Legacy abstraction (still supported)
- Runnable: Modern unified interface (0.1+)

### 1.2 Chain Types

```
┌─ Sequential Chains (Linear execution)
│  ├─ SimpleChain: Single LLM call
│  ├─ LLMChain: Prompt template + LLM
│  └─ SequentialChain: Multiple steps
│
├─ Branching Chains (Conditional logic)
│  ├─ RouterChain: Route to different chains
│  ├─ MultiPromptChain: Select prompt based on input
│  └─ ConditionalChain: If/else logic
│
└─ Composition Chains (Complex patterns)
   ├─ AnalyzeDocumentChain
   ├─ QAChain
   └─ RetrievalQAChain
```

## 2. LCEL (LangChain Expression Language)

### 2.1 Basic Composition with Pipes

```python
from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import StrOutputParser

# Individual components
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short essay about {topic}."
)

llm = OpenAI(temperature=0.7)
output_parser = StrOutputParser()

# Compose with pipes (|)
chain = prompt | llm | output_parser

# Execute
result = chain.invoke({"topic": "artificial intelligence"})
print(result)  # Essay about AI
```

**How pipes work**:
```
prompt.invoke({"topic": "AI"}) 
  → PromptValue(text="Write a short essay about AI.")
  
llm.invoke(PromptValue) 
  → AIMessage(content="AI is...")
  
output_parser.invoke(AIMessage) 
  → "AI is..."
```

### 2.2 Advanced LCEL Patterns

**Parallel Processing**:
```python
from langchain_core.runnables import RunnableParallel

# Run multiple chains in parallel
chain_a = prompt_a | llm
chain_b = prompt_b | llm

parallel_chain = RunnableParallel(
    result_a=chain_a,
    result_b=chain_b,
    result_c={"text": prompt_c | llm}  # Can mix dict and chains
)

# Results come back as dict
results = parallel_chain.invoke({"input": "..."})  
# {"result_a": "...", "result_b": "...", "result_c": {...}}
```

**Conditional Routing**:
```python
from langchain_core.runnables import RunnableBranch

# Route based on condition
def is_numeric(x):
    try:
        float(x["input"])
        return True
    except:
        return False

math_chain = prompt_math | llm
general_chain = prompt_general | llm

router = RunnableBranch(
    (is_numeric, math_chain),
    (lambda x: True, general_chain)  # Default
)

result = router.invoke({"input": "Calculate 2+2"})  # Uses math_chain
```

**Mapping and Transformation**:
```python
from langchain_core.runnables import RunnableMap

# Transform input before passing to chain
def extract_text(x):
    return {"text": x["document"].split("\n")[0]}

chain = (
    RunnableMap(extract_text) 
    | prompt 
    | llm
)

# Input: {"document": "Line 1\nLine 2"}
# Extract: {"text": "Line 1"}
# Pass to prompt, then llm
```

## 3. Sequential Chain Patterns

### 3.1 Simple Sequential Chains

```python
class DocumentAnalysisChain:
    """Multi-step document analysis"""
    
    def __init__(self, llm):
        self.llm = llm
        self.extract_chain = self._build_extract_chain()
        self.summarize_chain = self._build_summarize_chain()
        self.classify_chain = self._build_classify_chain()
    
    def _build_extract_chain(self):
        prompt = PromptTemplate(
            input_variables=["document"],
            template="Extract key points from:\n{document}"
        )
        return prompt | self.llm
    
    def _build_summarize_chain(self):
        prompt = PromptTemplate(
            input_variables=["key_points"],
            template="Summarize these points:\n{key_points}"
        )
        return prompt | self.llm
    
    def _build_classify_chain(self):
        prompt = PromptTemplate(
            input_variables=["summary"],
            template="Classify this summary as: technical, business, or general\n{summary}"
        )
        return prompt | self.llm
    
    def analyze(self, document: str):
        # Step 1: Extract
        key_points = self.extract_chain.invoke({"document": document})
        
        # Step 2: Summarize
        summary = self.summarize_chain.invoke({"key_points": key_points})
        
        # Step 3: Classify
        classification = self.classify_chain.invoke({"summary": summary})
        
        return {
            "key_points": key_points,
            "summary": summary,
            "classification": classification
        }
```

### 3.2 Error Handling in Chains

```python
from langchain_core.runnables import RunnableWithFallback

class RobustChain:
    """Chain with fallback strategies"""
    
    def __init__(self, primary_llm, fallback_llm):
        self.primary = primary_llm
        self.fallback = fallback_llm
    
    def build_chain(self, prompt_template: str):
        primary_chain = (
            PromptTemplate.from_template(prompt_template) 
            | self.primary
        )
        
        fallback_chain = (
            PromptTemplate.from_template(prompt_template) 
            | self.fallback
        )
        
        # Use fallback if primary fails
        return RunnableWithFallback(
            runnable=primary_chain,
            fallbacks=[fallback_chain]
        )

# Usage
chain = RobustChain(ChatOpenAI(), ChatAnthropic()).build_chain(
    "Answer this question: {question}"
)
result = chain.invoke({"question": "What is AI?"})
```

**Retry Logic**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RetryableChain:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def invoke_with_retry(self, chain, input_data):
        return chain.invoke(input_data)

# Automatically retries on failure
chain = RetryableChain()
result = chain.invoke_with_retry(my_chain, {"input": "..."})
```

## 4. Router Chains

### 4.1 Multi-Prompt Router

```python
class ContextAwareRouter:
    """Route to appropriate handler based on context"""
    
    def __init__(self, llm):
        self.llm = llm
        self.router_chain = self._build_router()
        self.chains = {
            "math": self._build_math_chain(),
            "programming": self._build_programming_chain(),
            "general": self._build_general_chain()
        }
    
    def _build_router(self):
        """Classify input to determine which chain to use"""
        prompt = PromptTemplate(
            input_variables=["input"],
            template="""Classify this query as:
- 'math': If it's about mathematics or calculations
- 'programming': If it's about coding or software
- 'general': Otherwise

Query: {input}
Classification: """
        )
        return prompt | self.llm
    
    def _build_math_chain(self):
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Solve this math problem step by step:\n{question}"
        )
        return prompt | self.llm
    
    def _build_programming_chain(self):
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer this programming question with code examples:\n{question}"
        )
        return prompt | self.llm
    
    def _build_general_chain(self):
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer this question:\n{question}"
        )
        return prompt | self.llm
    
    def invoke(self, query: str) -> str:
        # Route based on classification
        classification = self.router_chain.invoke({"input": query}).strip().lower()
        
        # Select appropriate chain
        if "math" in classification:
            selected_chain = self.chains["math"]
        elif "programming" in classification:
            selected_chain = self.chains["programming"]
        else:
            selected_chain = self.chains["general"]
        
        # Execute
        return selected_chain.invoke({"question": query})
```

## 5. Advanced Composition Patterns

### 5.1 Tree-Based Execution

```python
class DecisionTreeChain:
    """Multi-level decision making"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def build_tree(self):
        """Build decision tree for task decomposition"""
        
        # Level 1: Understand intent
        intent_prompt = PromptTemplate(
            input_variables=["input"],
            template="What is the user asking for?\n{input}\nIntent: "
        )
        intent_chain = intent_prompt | self.llm
        
        # Level 2: Decompose into subtasks
        subtask_prompt = PromptTemplate(
            input_variables=["intent"],
            template="Break this into 3 subtasks:\n{intent}\nSubtasks: "
        )
        subtask_chain = subtask_prompt | self.llm
        
        # Level 3: Execute subtasks
        def execute_subtasks(subtasks_str):
            subtasks = subtasks_str.split('\n')
            results = []
            for task in subtasks:
                exec_prompt = PromptTemplate(
                    input_variables=["task"],
                    template="Execute this subtask:\n{task}\nResult: "
                )
                result = (exec_prompt | self.llm).invoke({"task": task})
                results.append(result)
            return results
        
        return intent_chain, subtask_chain, execute_subtasks
```

### 5.2 Pipeline with State

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PipelineState:
    input: str
    intermediate_results: Dict[str, Any]
    errors: list
    
class StatefulChain:
    """Chain that maintains state across steps"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def process(self, input_text: str) -> PipelineState:
        state = PipelineState(
            input=input_text,
            intermediate_results={},
            errors=[]
        )
        
        # Step 1: Validate
        try:
            state.intermediate_results["validation"] = self._validate(input_text)
        except Exception as e:
            state.errors.append(("validation", str(e)))
            return state
        
        # Step 2: Process
        try:
            state.intermediate_results["processing"] = self._process(
                state.intermediate_results["validation"]
            )
        except Exception as e:
            state.errors.append(("processing", str(e)))
            return state
        
        # Step 3: Format
        try:
            state.intermediate_results["output"] = self._format(
                state.intermediate_results["processing"]
            )
        except Exception as e:
            state.errors.append(("formatting", str(e)))
        
        return state
    
    def _validate(self, text: str) -> str:
        # Validate input
        if len(text.strip()) == 0:
            raise ValueError("Empty input")
        return text
    
    def _process(self, text: str) -> str:
        prompt = PromptTemplate.from_template("Process: {text}")
        return (prompt | self.llm).invoke({"text": text})
    
    def _format(self, text: str) -> str:
        prompt = PromptTemplate.from_template("Format nicely: {text}")
        return (prompt | self.llm).invoke({"text": text})
```

## 6. Composition Best Practices

### 6.1 Testing Chains

```python
import unittest

class TestDocumentChain(unittest.TestCase):
    def setUp(self):
        self.chain = DocumentAnalysisChain(llm=FakeLLM())
    
    def test_chain_execution(self):
        result = self.chain.analyze("Test document")
        self.assertIn("key_points", result)
        self.assertIn("summary", result)
        self.assertIn("classification", result)
    
    def test_error_handling(self):
        with self.assertRaises(ValueError):
            self.chain.analyze("")
```

### 6.2 Chain Debugging

```python
from langchain.callbacks import StdOutCallbackHandler

# Enable verbose logging
chain = (
    prompt 
    | llm.with_config(callbacks=[StdOutCallbackHandler()])
)

# Or use debug mode
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    run_name="my_chain",
    tags=["debug"],
    metadata={"purpose": "testing"}
)

result = chain.invoke(input, config=config)
```

## Key Takeaways

1. **LCEL is powerful**: Pipes provide intuitive composition
2. **Routing handles complexity**: Route to appropriate handlers
3. **Composition is testable**: Each component can be tested independently
4. **State matters**: Track intermediate results and errors
5. **Observability is essential**: Use callbacks and logging

## References

- LCEL Documentation: https://python.langchain.com/docs/expression_language/
- Runnable Interface: https://api.python.langchain.com/en/latest/runnables/
