---
title: "AI & Agent Glossary Cheatsheet"
date: 2026-06-25
summary: "A quick-reference guide to currently trending AI, Agent, and Large Language Model (LLM) terms and technologies. Features live search and filter."
type: "cheatsheets"
---

## Agents & Architectures | 智能体与核心架构

### AI Agent | 智能体
- **Definition**: An AI system that can autonomously perceive its environment, reason, plan, and invoke external tools to execute tasks in order to achieve a specific goal. Compared to traditional single-turn chatbots, agents possess autonomy, memory, and action capabilities.
- **In Simple Terms**: "An AI assistant with hands, feet, and the ability to self-reflect." You give it a high-level goal (e.g., "book the cheapest flight to Beijing tomorrow"), and it will search flights, compare prices, manage time, call APIs, and self-correct if it makes a mistake.
- **How to Implement**: Structured around the "Brain (LLM) + Memory + Planning + Tools" architecture. Often implemented using the ReAct (Reason + Act) loop where the model alternates between reasoning and executing actions.
- **Code**:
  ```python
  # Simple Agent Decision Loop (ReAct Core Logic)
  class SimpleAgent:
      def __init__(self, llm, tools):
          self.llm = llm
          self.tools = tools
          self.memory = []

      def run(self, task):
          prompt = f"Task: {task}\nAvailable Tools: {list(self.tools.keys())}\nThink step-by-step and select tools."
          while True:
              response = self.llm.generate(prompt + "\n".join(self.memory))
              # Parse thought, action, and action_input from model response
              thought, action, action_input = self.parse_response(response)
              print(f"Thought: {thought}")
              if action == "Finish":
                  return action_input
              
              # Execute the tool
              result = self.tools[action](action_input)
              # Store in memory for next step decision
              self.memory.append(f"Thought: {thought}\nAction: {action}({action_input})\nObservation: {result}")
  ```

### Skill & Plugin | 技能与插件
- **Definition**: Pre-defined action specifications and execution logics configured in an agent or LLM platform. A skill can be a specific system prompt, executable code, or an API schema that extends the baseline capabilities of an agent.
- **In Simple Terms**: "An AI's expansion pack or tool belt." A raw LLM might not know the square root of 1024 offhand, but with a "calculator" skill plugged in, it can invoke it to yield the correct answer.
- **How to Implement**: Typically defined in config files (like JSON Schema or YAML) describing its inputs, outputs, description, and the underlying executor logic.
- **Code**:
  ```yaml
  # Skill Configuration (Example)
  name: "GoogleSearch"
  description: "Search Google for up-to-date web information"
  parameters:
    properties:
      query:
        type: "string"
        description: "The search query"
    required: ["query"]
  ```

### Multi-Agent System | 多智能体系统
- **Definition**: A system where multiple AI agents collaborate, communicate, and distribute tasks to solve complex problems. Each agent plays a distinct role (e.g., developer, QA, product manager) to advance the overall objective.
- **In Simple Terms**: "A virtual AI department or team." A single AI might make mistakes, so we build an "AI R&D department": an AI coder writes code, an AI QA checks for bugs, and an AI PM manages requirements.
- **How to Implement**: Assign roles and define communication protocols (like message queues or graph structures) using frameworks like LangGraph, AutoGen, or CrewAI.
- **Code**:
  ```python
  # Multi-Agent Collaboration using CrewAI
  from crewai import Agent, Task, Crew

  # Define Coder Agent
  coder = Agent(
      role='Senior Python Developer',
      goal='Write clean, efficient Python code',
      backstory='An expert in Python programming'
  )

  # Define Tester Agent
  tester = Agent(
      role='QA Engineer',
      goal='Find bugs and write unit tests',
      backstory='A detail-oriented software tester'
  )

  # Assemble the crew and run tasks
  crew = Crew(agents=[coder, tester], tasks=[...])
  result = crew.kickoff()
  ```

### Tool Use & Function Calling | 工具使用与函数调用
- **Definition**: The process where a large language model generates structured output (such as JSON) indicating its intent to invoke an external API/function, and processes the output returned by the API to continue answering the prompt.
- **In Simple Terms**: "AI requesting you to run a tool for it." The LLM doesn't execute code itself; instead, it outputs: "Please call `get_weather(city='Beijing')`", and waits for you to run it and feed the weather results back.
- **How to Implement**: Declaring functions in the API call using the `tools` parameter; the model then replies with a `tool_calls` object if a tool is needed.
- **Code**:
  ```python
  import openai

  # Declare tools
  tools = [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get real-time weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          }
        }
      }
  }]

  # Request OpenAI
  response = openai.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": "How is the weather in Beijing?"}],
      tools=tools
  )
  # Parse the tool call instruction
  tool_call = response.choices[0].message.tool_calls[0]
  print(f"Model requests calling: {tool_call.function.name} with args: {tool_call.function.arguments}")
  ```

### Chain of Thought (CoT) | 思维链
- **Definition**: A prompt engineering technique or training methodology that encourages LLMs to produce intermediate reasoning steps before generating the final answer, significantly improving accuracy on math, logic, and reasoning tasks.
- **In Simple Terms**: "Letting the AI show its work." Instead of forcing it to jump straight to the answer, guide it by saying "Let's think step-by-step" to let it reason out loud.
- **How to Implement**: Append "Let's think step by step" to the prompt, or provide few-shot examples with reasoning paths. Advanced models (like O1/O3) have native CoT built-in.
- **Code**:
  ```python
  # Zero-shot CoT Prompting
  prompt = "If Alice has 5 apples and Bob has twice as many. Alice gives Bob 3. How many does Bob have now? Let's think step-by-step and then output the final answer."
  ```

---

## Data & RAG | 数据处理与检索增强

### Crawl4AI & OpenCrawl | 大模型爬虫
- **Definition**: Web scraping and parsing tools tailored for feeding Large Language Models. `Crawl4AI` is a popular asynchronous LLM-friendly crawler in Python, while `OpenCrawl`/`Common Crawl` represent the broader open-source scraping ecosystem. They strip advertisements and navigation, exporting raw web pages into clean Markdown.
- **In Simple Terms**: "An AI-friendly food processor." Standard web scrapers return messy HTML full of code, while AI crawlers clean and output markdown that models can read quickly and accurately.
- **How to Implement**: Using asynchronous fetching libraries along with Headless browsers (like Playwright) to load dynamic JavaScript, then stripping HTML down to markdown.
- **Code**:
  ```python
  # Crawl websites to Markdown with crawl4ai
  import asyncio
  from crawl4ai import AsyncWebCrawler

  async def main():
      async with AsyncWebCrawler() as crawler:
          result = await crawler.arun(url="https://news.ycombinator.com")
          # Access the parsed markdown directly
          print(result.markdown[:500])

  asyncio.run(main())
  ```

### RAG (Retrieval-Augmented Generation) | 检索增强生成
- **Definition**: A system architecture that combines document retrieval with text generation. When a question is received, the system retrieves relevant documents from an external corpus and prefixes them to the prompt as context, grounding the LLM's answer.
- **In Simple Terms**: "Open-book exam." Instead of relying purely on its pre-trained memory (which can hallucinate), the AI searches your knowledge base first, takes the relevant page snippets, and answers your question based on them.
- **How to Implement**: User query $\rightarrow$ Semantic retrieval of document chunks $\rightarrow$ Prompt construction (Context + Query) $\rightarrow$ LLM generation.
- **Code**:
  ```python
  # Simple RAG flow pseudo-code
  def simple_rag(query, vector_db, llm):
      # 1. Retrieve the top 3 relevant chunks
      retrieved_docs = vector_db.search(query, k=3)
      context = "\n".join(retrieved_docs)
      
      # 2. Construct the open-book exam prompt
      prompt = f"Answer the query based on the context.\n\nContext:\n{context}\n\nQuery: {query}"
      
      # 3. Generate the response
      return llm.generate(prompt)
  ```

### Vector Database | 向量数据库
- **Definition**: A database optimized to store, index, and query high-dimensional vector embeddings. It facilitates fast "nearest neighbor search" (similarity search) based on distance metrics in vector spaces.
- **In Simple Terms**: "A conceptual search engine." If you search for "tomato", a relational database might miss "pomodoro", but a vector database knows they are close in meaning and returns them both.
- **How to Implement**: Employs indexing algorithms (e.g., HNSW, IVF) to calculate Cosine Similarity or Euclidean Distance over millions of high-dimensional vectors.
- **Code**:
  ```python
  import chromadb

  # Initialize client and collection
  chroma_client = chromadb.Client()
  collection = chroma_client.create_collection(name="ai_glossary")

  # Add documents (embeddings are generated automatically)
  collection.add(
      documents=["Agents can use tools autonomously", "Vector databases index embeddings"],
      metadatas=[{"source": "agent"}, {"source": "db"}],
      ids=["doc1", "doc2"]
  )

  # Search
  results = collection.query(
      query_texts=["How do AI agents work?"],
      n_results=1
  )
  print(results['documents'])
  ```

### Embedding | 嵌入向量
- **Definition**: A numerical representation that maps high-dimensional, discrete tokens or data (like text or images) into a low-dimensional, continuous vector space where semantically similar items reside closer together.
- **In Simple Terms**: "Coordinates of meaning." Everything is converted into a list of numbers. For example, "cat" might be `[0.2, 0.8, -0.1]`, "dog" `[0.2, 0.7, -0.2]`, and "brick" `[-0.9, 0.1, 0.5]`. Cat and dog coordinates are very close.
- **How to Implement**: Utilizes deep learning models (like BERT or sentence-transformers) to output a fixed-size array of floats (e.g., 1536 dimensions) for any input text.
- **Code**:
  ```python
  import sentence_transformers
  import numpy as np

  # Load open-source model
  model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
  
  # Encode sentences
  vec1 = model.encode("AI changes the world")
  vec2 = model.encode("Artificial intelligence transforms everything")
  
  # Calculate cosine similarity
  similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
  print(f"Similarity: {similarity:.4f}")
  ```

---

## LLM Parameters & Concepts | 模型参数与核心概念

### Context Window | 上下文窗口
- **Definition**: The maximum number of tokens an LLM can process in a single inference call (including system prompt, history, current query, and the generated response).
- **In Simple Terms**: "The AI's active memory (RAM)." If the model has a 128k token window, you can feed it a whole book in one prompt; if you go over the limit, it will forget the earlier parts or reject the request.
- **How to Implement**: Computationally constrained by the quadratic scaling of Attention. Advanced techniques like Rotary Position Embeddings (RoPE) interpolation and FlashAttention enable windows up to millions of tokens.

### System Prompt | 系统提示词
- **Definition**: A high-priority instruction set at the beginning of a conversation that defines the AI's identity, guidelines, output format rules, and safety bounds.
- **In Simple Terms**: "The AI's backstory and rules." Like giving an actor instructions before they go on stage: "You are a helpful, professional customer support agent. Speak only in JSON and do not mention internal system specs."
- **How to Implement**: Set by passing a message block with the `role` key set to `"system"` in the API chat completions payload.
- **Code**:
  ```python
  messages = [
      # Define system behavior
      {"role": "system", "content": "You are a professional assistant who outputs answers in JSON only."},
      # User input
      {"role": "user", "content": "Explain APIs briefly."}
  ]
  ```

### Temperature & Top-P | 温度与核采样
- **Definition**: Two key decoding parameters that control the creativity and randomness of the model's token selection. `Temperature` scales the logits to control probability flatness; `Top-P` limits candidate selection to a cumulative probability threshold.
- **In Simple Terms**:
  - **Temperature**: "AI's imagination dial." At 0, it chooses the absolute safest, most predictable word every time. At 1.0, it is creative, but more prone to hallucination.
  - **Top-P**: "The candidate filter." A Top-P of 0.95 means the model only picks from the top 95% most logical words, completely discarding the bottom 5% oddball words.
- **Code**:
  ```python
  # API call parameters
  response = openai.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": "Write a short poem about coding"}],
      temperature=0.8, # More creative
      top_p=0.95       # Keep vocabulary clean
  )
  ```

### Token | 词元
- **Definition**: The basic unit of data processed by an LLM. It can represent a single character, a sub-word unit, or an entire word depending on the tokenizer.
- **In Simple Terms**: "The AI's alphabet cards." LLMs don't read words or letters directly. They split text into tokens, converting each token into a number (ID). For instance, 100 English words map to roughly 130 tokens.
- **Code**:
  ```python
  import tiktoken

  # Load tokenizer for GPT-4
  enc = tiktoken.encoding_for_model("gpt-4")
  text = "Hello world! Deep learning"
  
  # Tokenize text
  tokens = enc.encode(text)
  print(f"Token IDs: {tokens}")
  print(f"Token Count: {len(tokens)}")
  ```

---

## Fine-Tuning & Safety | 微调、对齐与安全

### Fine-Tuning | 模型微调
- **Definition**: The process of taking a pre-trained base model and training it further on a smaller, domain-specific dataset (such as medical records or customer logs) to adapt it for specific tasks. High-efficiency techniques include LoRA and QLoRA.
- **In Simple Terms**: "Specialized job training." A base model is like a college graduate with general knowledge. Fine-tuning is training them on your company's proprietary manuals so they become a specialized assistant for your business.

### RLHF & RLAIF | 人类反馈强化学习与 AI 反馈强化学习
- **Definition**: Aligning a model's behavior with human preferences by training a Reward Model on human preference comparisons, then optimizing the policy using reinforcement learning (like PPO). RLAIF substitutes humans with high-quality AI models to judge and score completions.
- **In Simple Terms**: "A tutor grading the AI's homework." Since models can initially generate harmful or false statements, we have humans or super-AIs grade their responses, rewarding helpful answers and penalizing toxic ones until the model behaves.

### Guardrails | 护栏与安全边界
- **Definition**: Software frameworks (e.g., Llama Guard, Guardrails AI, NeMo Guardrails) that run verification checks on LLM inputs and outputs to block jailbreaks, prompt injections, and filter out toxic or sensitive content.
- **In Simple Terms**: "The AI's bodyguard and quality checker." Regardless of what the model wants to say, the guardrail checks the output: "Does this contain private passwords or hate speech?" If it does, it swaps the output with a generic warning.
- **Code**:
  ```python
  # Simple output guardrail demonstration
  def output_guardrail(ai_response):
      sensitive_phrases = ["private password", "how to build a bomb", "bypass safety"]
      for phrase in sensitive_phrases:
          if phrase in ai_response.lower():
              return "I apologize, but I cannot assist with that request."
      return ai_response
  ```
