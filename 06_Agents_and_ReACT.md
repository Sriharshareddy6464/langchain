# Agents and ReACT Framework

## 1. Agent Fundamentals

### 1.1 What is an Agent?

**Definition**: An autonomous system that:
1. **Perceives** environment/input
2. **Reasons** about available actions
3. **Acts** by executing tool calls
4. **Observes** results and adjusts

```
┌─────────────────────────────────────┐
│ Agent Loop (Think-Act-Observe)      │
├─────────────────────────────────────┤
│ 1. Think: LLM decides next action   │
│ 2. Act:   Execute tool/action       │
│ 3. Observe: Get result              │
│ 4. Repeat until goal reached        │
└─────────────────────────────────────┘
```

### 1.2 Agent Types

```
Agent Taxonomy:
├─ ReACT Agents: Reason-Act-Collect framework
├─ Tool-Using Agents: Call external tools
├─ Function Calling Agents: Use model's native function calling
├─ OpenAI Assistants: Stateful agents with persistent state
└─ Custom Agents: Implement own decision logic
```

## 2. ReACT Framework

### 2.1 ReACT Components

**ReACT**: Reasoning + Acting

```
Thought: What should I do?
Action: Which tool to use?
Action Input: Parameters for tool
─────────────────────────────
Observation: Tool result
─────────────────────────────
Thought: What does this mean?
Action: Next action
...
─────────────────────────────
Thought: I have the answer
Final Answer: ...
```

### 2.2 ReACT Agent Implementation

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.llms import OpenAI

class ReACTAgent:
    def __init__(self, llm, tools: list):
        self.llm = llm
        self.tools = tools
        self.agent = self._build_agent()
    
    def _build_agent(self):
        # Define ReACT prompt template
        prompt_template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            early_stopping_method="force"
        )
    
    def run(self, query: str) -> str:
        return self.agent.run(query)

# Define tools
tools = [
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Useful for math calculations"
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for finding information"
    )
]

# Create agent
agent = ReACTAgent(llm=OpenAI(), tools=tools)
response = agent.run("What is the capital of France?")
```

### 2.3 Thought Chain Analysis

```python
class ThoughtAnalyzer:
    """Analyze agent's reasoning steps"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_reasoning(self, thoughts: List[str]) -> dict:
        """
        Analyze quality of agent's reasoning
        """
        analysis = {
            "thought_count": len(thoughts),
            "quality_score": 0,
            "issues": [],
            "improvements": []
        }
        
        # Check thought quality
        for i, thought in enumerate(thoughts):
            if len(thought.strip()) < 10:
                analysis["issues"].append(f"Thought {i}: Too brief")
            
            if "?" not in thought and "should" not in thought:
                analysis["issues"].append(f"Thought {i}: Not questioning")
        
        # Score quality (0-100)
        analysis["quality_score"] = max(0, 100 - len(analysis["issues"]) * 20)
        
        return analysis
```

## 3. Tool and Action Space

### 3.1 Tool Design

```python
from langchain.tools import Tool, tool
from typing import Optional

# Declarative approach
@tool
def calculate(expression: str) -> str:
    """Evaluate mathematical expression. Use for calculations."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# Imperative approach
weather_tool = Tool(
    name="WeatherSearch",
    func=get_weather,
    description="""Useful when you need to answer questions about weather.
    Input should be a location string.""",
    return_direct=False  # Include in context vs direct return
)

# Advanced: Tool with input schema
from langchain_core.pydantic_v1 import BaseModel, Field

class WeatherInput(BaseModel):
    location: str = Field(description="Location to get weather for")
    unit: str = Field(description="Temperature unit (C/F)", default="C")

@tool(args_schema=WeatherInput)
def get_weather_typed(location: str, unit: str = "C") -> str:
    """Get weather for a location"""
    # Implementation
    pass
```

### 3.2 Action Space Definition

```python
class ActionSpace:
    """Define available actions/tools for agent"""
    
    def __init__(self):
        self.tools = {}
        self.capabilities = {
            "search": False,
            "compute": False,
            "database": False,
            "external_api": False
        }
    
    def register_tool(self, name: str, tool_func, description: str, category: str):
        """Register a tool"""
        self.tools[name] = {
            "func": tool_func,
            "description": description,
            "category": category
        }
        
        # Update capabilities
        if category in self.capabilities:
            self.capabilities[category] = True
    
    def get_available_tools(self) -> List[Tool]:
        """Return tool objects"""
        tools = []
        for name, config in self.tools.items():
            tools.append(Tool(
                name=name,
                func=config["func"],
                description=config["description"]
            ))
        return tools
    
    def get_tool_descriptions(self) -> str:
        """Format tool descriptions for prompt"""
        descriptions = []
        for name, config in self.tools.items():
            descriptions.append(f"{name}: {config['description']}")
        return "\n".join(descriptions)
```

## 4. Multi-Step Reasoning

### 4.1 Planning Before Acting

```python
class PlanningAgent:
    """
    Plan first, then execute (better than immediate action)
    
    Algorithm:
    1. Create plan (decompose into subtasks)
    2. Execute plan step by step
    3. Adapt if needed
    """
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.plan = None
    
    def create_plan(self, query: str) -> List[str]:
        """Decompose query into steps"""
        prompt = f"""
Break down this task into 3-5 concrete steps:

Task: {query}

Steps:
1. 
2. 
3. 
"""
        response = self.llm.invoke(prompt)
        # Parse steps
        steps = [s.strip() for s in response.split('\n') if s.strip()]
        self.plan = steps
        return steps
    
    def execute_plan(self, query: str) -> str:
        """Execute planned steps"""
        steps = self.create_plan(query)
        
        results = {}
        for i, step in enumerate(steps):
            # Execute each step
            action_prompt = f"""
Step {i+1}: {step}

To accomplish this, what action should I take?
Available actions: {', '.join([t.name for t in self.tools])}

Action: """
            action = self.llm.invoke(action_prompt)
            
            # Execute action
            result = self._execute_action(action)
            results[step] = result
        
        return results
    
    def _execute_action(self, action: str):
        # Find and execute tool
        for tool in self.tools:
            if tool.name.lower() in action.lower():
                return tool.func(action)
        return "Action not found"
```

### 4.2 Backtracking and Adaptation

```python
class AdaptiveAgent:
    """Agent that backtracks on failures and tries alternatives"""
    
    def __init__(self, llm, tools, max_backtrack: int = 3):
        self.llm = llm
        self.tools = tools
        self.max_backtrack = max_backtrack
        self.execution_history = []
    
    def run_with_backtracking(self, query: str) -> str:
        """Execute with ability to backtrack"""
        return self._run_recursive(query, backtrack_depth=0)
    
    def _run_recursive(self, query: str, backtrack_depth: int) -> str:
        if backtrack_depth >= self.max_backtrack:
            return "Max backtracking attempts exceeded"
        
        # Try to find solution
        action = self._decide_action(query)
        
        try:
            result = self._execute_action(action)
            
            # Check if result is satisfactory
            if self._is_satisfactory(result):
                return result
            else:
                # Backtrack: try different approach
                alternative_query = f"{query} (previous approach failed: {result})"
                return self._run_recursive(
                    alternative_query,
                    backtrack_depth + 1
                )
        
        except Exception as e:
            # On error, try alternative
            alternative_query = f"{query} (error: {str(e)})"
            return self._run_recursive(
                alternative_query,
                backtrack_depth + 1
            )
    
    def _is_satisfactory(self, result: str) -> bool:
        """Evaluate if result meets quality threshold"""
        # Check for error indicators
        error_keywords = ["error", "failed", "unable", "unknown"]
        return not any(kw in result.lower() for kw in error_keywords)
```

## 5. Agent Optimization

### 5.1 Prompt Optimization for Agents

```python
class AgentPromptOptimizer:
    """
    Optimize agent system prompt for better performance
    """
    
    SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant with access to the following tools:

{tools_description}

Your workflow:
1. THINK: Analyze the problem carefully
2. CHOOSE: Select the right tool for this step
3. ACT: Call the tool with appropriate inputs
4. OBSERVE: Understand the result
5. REPEAT or CONCLUDE

Key principles:
- Use tools strategically to gather information
- Combine results from multiple tools
- Always verify facts before stating them
- Ask clarifying questions if ambiguous
- Admit uncertainty rather than guessing
"""
    
    def __init__(self, llm):
        self.llm = llm
        self.base_prompt = self.SYSTEM_PROMPT_TEMPLATE
    
    def optimize_for_domain(self, domain: str, examples: List[dict]) -> str:
        """
        Customize prompt for specific domain
        """
        prompt = f"""
Optimize this system prompt for {domain} tasks:

{self.base_prompt}

Examples of tasks in this domain:
{chr(10).join([f"- {ex['task']}: expected {ex['expected']}" for ex in examples[:3]])}

Provide an optimized version of the system prompt.
"""
        
        optimized = self.llm.invoke(prompt)
        return optimized
```

### 5.2 Tool Selection Strategy

```python
class SmartToolSelector:
    """
    Select most relevant tools for query (reduce hallucination)
    
    Algorithm:
    1. Embed all tool descriptions
    2. Embed query
    3. Rank tools by semantic similarity
    4. Include top-K tools in agent prompt
    """
    
    def __init__(self, llm, embeddings, tools):
        self.llm = llm
        self.embeddings = embeddings
        self.tools = tools
        self.tool_embeddings = self._embed_tools()
    
    def _embed_tools(self):
        """Embed tool descriptions"""
        embeddings = {}
        for tool in self.tools:
            embedding = self.embeddings.embed_query(tool.description)
            embeddings[tool.name] = embedding
        return embeddings
    
    def select_tools(self, query: str, k: int = 3) -> List[Tool]:
        """Select most relevant tools"""
        query_embedding = self.embeddings.embed_query(query)
        
        # Rank by similarity
        similarities = {}
        for tool_name, tool_embedding in self.tool_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, tool_embedding)
            similarities[tool_name] = similarity
        
        # Get top-K
        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        selected_names = [name for name, _ in ranked[:k]]
        
        # Return tool objects
        return [t for t in self.tools if t.name in selected_names]
    
    def _cosine_similarity(self, a, b):
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

## Key Takeaways

1. **ReACT is powerful**: Reasoning + Acting is more reliable than planning alone
2. **Tool selection matters**: Don't expose all tools if not needed
3. **Planning helps**: Decompose complex tasks into steps
4. **Backtracking recovers**: Adapt when approach fails
5. **Iteration is key**: Most problems need multiple thought-action cycles

## References

- "ReACT: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2023): https://arxiv.org/abs/2210.03629
- LangChain Agents: https://python.langchain.com/docs/modules/agents/
