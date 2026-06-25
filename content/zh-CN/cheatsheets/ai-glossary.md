---
title: "热门 AI 与智能体速查表"
date: 2026-06-25
summary: "当前热门的 AI、智能体（Agent）及大语言模型（LLM）核心词汇与技术速查手册。支持实时搜索与过滤。"
type: "cheatsheets"
---

## 智能体与核心架构 | Agents & Architectures

### 智能体 | AI Agent
- **定义**: 能够自主感知环境、进行推理和规划，并调用各种外部工具来执行任务以达到特定目标的 AI 系统。与单次问答的 Chatbot 相比，Agent 具备自主决策、长期/短期记忆和工具执行能力。
- **大白话**: “有手有脚、会自我反思的 AI 助理”。你给它一个终极目标（比如“订一张明天去北京最便宜的机票”），它自己会上网查机票、对比价格、计算时间、调用购票接口，甚至如果中途出错，它还会自我修正。
- **怎么实现**: 通过“大模型大脑（LLM）+ 记忆（Memory）+ 规划（Planning）+ 工具集（Tools）”的模式。利用 ReAct（Reason + Act）等提示词框架让模型交替进行“思考”和“行动”。
- **代码**:
  ```python
  # 简易 Agent 决策循环 (ReAct 核心逻辑)
  class SimpleAgent:
      def __init__(self, llm, tools):
          self.llm = llm
          self.tools = tools
          self.memory = []

      def run(self, task):
          prompt = f"任务: {task}\n可用工具: {list(self.tools.keys())}\n请一步步思考并选择工具执行。"
          while True:
              response = self.llm.generate(prompt + "\n".join(self.memory))
              # 解析出模型输出的 Thought (思考) 和 Action (行动)
              thought, action, action_input = self.parse_response(response)
              print(f"思考: {thought}")
              if action == "Finish":
                  return action_input
              
              # 调用工具执行行动
              result = self.tools[action](action_input)
              # 将结果存入记忆，供下一步决策参考
              self.memory.append(f"Thought: {thought}\nAction: {action}({action_input})\nObservation: {result}")
  ```

### 技能与插件 | Skill & Plugin
- **定义**: 预设在智能体或大模型平台中的动作规范和执行逻辑。Skill 可以是特定的系统提示词、一段可执行的代码或 API 调用描述，用来扩展 Agent 的底层能力范围。
- **大白话**: “AI 的技能槽或外挂背包”。比如刚出厂的 AI 算不出“1024的立方根”，但给它插上一个“计算器”插件，它就会调用这个技能来输出正确结果。
- **怎么实现**: 通常以配置文件的形式定义（如 YAML frontmatter 或 JSON Schema），描述技能的输入、输出、描述信息以及底层执行的具体逻辑。
- **代码**:
  ```yaml
  # 技能配置文件 (SKILL.md / skill.yaml 示例)
  name: "GoogleSearch"
  description: "使用谷歌搜索来获取最新的网络实时信息"
  parameters:
    properties:
      query:
        type: "string"
        description: "搜索关键词"
    required: ["query"]
  ```

### 多智能体系统 | Multi-Agent System
- **定义**: 多个 AI Agent 相互协作、分工和通信以解决复杂问题的系统。每个 Agent 可以扮演不同的角色（例如程序员、测试员、产品经理），共同推进任务。
- **大白话**: “AI 部门或虚拟团队”。一个 AI 单打独斗容易犯错，那就组建一个“AI 研发部”：AI 程序员写代码，AI 测试员找 Bug，AI 产品经理把控需求，大家在一个群里协作。
- **怎么实现**: 定义不同的智能体角色，通过共享的通信总线（或消息队列）进行交互。通常使用 LangGraph、AutoGen 或 CrewAI 等框架实现。
- **代码**:
  ```python
  # 使用 CrewAI 概念的多智能体协作示意
  from crewai import Agent, Task, Crew

  # 定义程序员 Agent
  coder = Agent(
      role='Senior Python Developer',
      goal='Write clean, efficient Python code',
      backstory='An expert in Python programming'
  )

  # 定义测试员 Agent
  tester = Agent(
      role='QA Engineer',
      goal='Find bugs and write unit tests',
      backstory='A detail-oriented software tester'
  )

  # 组装任务和团队进行协作
  crew = Crew(agents=[coder, tester], tasks=[...])
  result = crew.kickoff()
  ```

### 工具使用与函数调用 | Tool Use & Function Calling
- **定义**: 大模型生成结构化数据（如 JSON），指示调用特定的外部函数/API，并将调用结果返回给模型以继续生成答案的技术。
- **大白话**: “AI 认领工具箱并说出用法”。大模型自己不运行代码，但它可以通过分析输入，决定说出：“我需要运行 `get_weather(city='Beijing')`”，让你（客户端）帮它运行并把天气结果喂回给它。
- **怎么实现**: 在发送请求时，通过 `tools` 参数向 API 声明可用函数，大模型识别后会在响应中返回 `tool_calls`。
- **代码**:
  ```python
  import openai

  # 声明工具
  tools = [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取指定城市的实时天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          }
        }
      }
  }]

  # 发送请求
  response = openai.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": "北京天气怎么样？"}],
      tools=tools
  )
  # 解析模型返回的工具调用指令
  tool_call = response.choices[0].message.tool_calls[0]
  print(f"大模型要求调用函数: {tool_call.function.name}，参数: {tool_call.function.arguments}")
  ```

### 思维链 | Chain of Thought (CoT)
- **定义**: 一种提示词工程技术或模型训练方法，通过引导大模型在输出最终答案之前，先生成中间推理步骤，从而显著提高复杂推理、数学和逻辑任务的准确率。
- **大白话**: “让 AI 在心里打草稿”。不要让它直接给出答案，而是引导它“让我们一步一步地思考：首先... 其次... 所以结论是...”。
- **怎么实现**: 在 Prompt 中加入“Let's think step by step”，或者在 Few-Shot 示例中给出详细的解题步骤；在 O1/O3 等原生推理模型中，模型会自动在后台运行 CoT。
- **代码**:
  ```python
  # 提示词级 CoT (Zero-shot CoT)
  prompt = "已知小明有5个苹果，小红的苹果是小明的两倍。小明给了小红3个后，小红还剩多少个？请一步一步思考并给出最终答案。"
  ```

---

## 数据处理与检索增强 | Data & RAG

### 大模型爬虫 | Crawl4AI & OpenCrawl
- **定义**: 专门为大语言模型（LLM）数据填充而设计的网页爬取与数据清洗工具。其中，`Crawl4AI` 是非常流行的 LLM 友好型 Python 异步爬虫；而 `OpenCrawl` / `Common Crawl` 则代表更广泛的开源/公共网页抓取生态。它们能自动过滤广告、导航栏，并将 HTML 直接转换为纯净的 Markdown 格式。
- **大白话**: “AI 的专业喂料机”。普通爬虫抓下的一堆 HTML 里夹杂着各种网页代码，而 AI 爬虫能瞬间把网页剥得只剩干净的文本和 Markdown，让 AI 读起来又快又准。
- **怎么实现**: 利用异步请求爬取网页，并集成 Puppeteer 或 Playwright 处理动态 JS，最后通过 CSS 选择器或大模型提取有价值的结构化文本。
- **代码**:
  ```python
  # 使用 crawl4ai 进行快速爬取并转换为 Markdown
  import asyncio
  from crawl4ai import AsyncWebCrawler

  async def main():
      async with AsyncWebCrawler() as crawler:
          result = await crawler.arun(url="https://news.ycombinator.com")
          # 直接获取清洗后的 markdown 内容
          print(result.markdown[:500])

  asyncio.run(main())
  ```

### 检索增强生成 | RAG
- **定义**: 将外部知识库（文档、网页、数据库）的检索与大语言模型生成相结合的技术。在回答用户问题时，先去知识库搜出相关文档，然后把文档和问题一起喂给大模型，让大模型“看着参考书写答案”。
- **大白话**: “开卷考试”。AI 不是靠脑子里的死记硬背来回答（容易胡说八道），而是先去公司文档库里搜索“2026年报”，把搜出来的几页纸贴在试卷旁边，看着它们组织语言回答你。
- **怎么实现**: 用户问题 $\rightarrow$ 向量化检索相关文档块 $\rightarrow$ 拼接提示词（Context + Question） $\rightarrow$ 大模型生成回答。
- **代码**:
  ```python
  # 简易 RAG 流程伪代码
  def simple_rag(query, vector_db, llm):
      # 1. 从向量数据库检索最相似的 3 条文档
      retrieved_docs = vector_db.search(query, k=3)
      context = "\n".join(retrieved_docs)
      
      # 2. 拼接成开卷考试 Prompt
      prompt = f"参考以下背景知识回答问题。\n\n背景知识：\n{context}\n\n问题：{query}"
      
      # 3. 大模型基于背景生成答案
      answer = llm.generate(prompt)
      return answer
  ```

### 向量数据库 | Vector Database
- **定义**: 专门用于存储、索引和快速检索高维向量数据的数据库。在 AI 领域，文本、图像等数据被转换成 Embedding 向量后，可以通过向量数据库进行高维空间下的“最近邻搜索”（Similarity Search）。
- **大白话**: “模糊意图搜索引擎”。普通数据库查“西红柿”，查不到“番茄”；而向量数据库知道“西红柿”和“番茄”在语义上是近邻，能直接把语义相近的东西都捞出来。
- **怎么实现**: 使用 HNSW、IVF 等索引算法，在海量高维数据中快速计算余弦距离（Cosine Similarity）或欧氏距离。
- **代码**:
  ```python
  import chromadb

  # 初始化本地向量数据库
  chroma_client = chromadb.Client()
  collection = chroma_client.create_collection(name="ai_glossary")

  # 插入文档（内部会自动生成 Embedding）
  collection.add(
      documents=["智能体能够自主使用工具", "向量数据库用于检索语义数据"],
      metadatas=[{"source": "agent"}, {"source": "db"}],
      ids=["doc1", "doc2"]
  )

  # 查询与 "AI Agent 怎么工作" 最相关的文档
  results = collection.query(
      query_texts=["AI Agent how it works"],
      n_results=1
  )
  print(results['documents'])
  ```

### 嵌入向量 | Embedding
- **定义**: 将高维离散的非结构化数据（如单词、句子、图片）映射到一个低维实数向量空间的技术。在这个空间中，语义越接近的词，它们对应的向量距离就越近。
- **大白话**: “万物的坐标化”。把世间万物变成一串坐标轴上的数字。比如“轻盈”的坐标是 `[0.2, 0.8, -0.1]`，“轻巧”是 `[0.2, 0.7, -0.2]`，“沉重”是 `[-0.9, 0.1, 0.5]`，你看，前两者的坐标距离非常近。
- **怎么实现**: 通过神经网络（如 BERT、Sentence-Transformers）将文本输入编码为固定长度的浮点数数组（如 1536 维）。
- **代码**:
  ```python
  import sentence_transformers
  import numpy as np

  # 加载开源 Embedding 模型
  model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
  
  # 计算两个句子的嵌入向量
  vec1 = model.encode("人工智能改变世界")
  vec2 = model.encode("AI is transforming the world")
  
  # 计算相似度（点积/余弦相似度）
  similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
  print(f"语义相似度: {similarity:.4f}")
  ```

---

## 模型参数与核心概念 | LLM Parameters & Concepts

### 上下文窗口 | Context Window
- **定义**: 大模型在单次对话中能够接收并处理的最大 Token 数量（包括输入的历史消息和模型即将生成的回复）。
- **大白话**: “大模型的运行内存（RAM）”。如果一个大模型的上下文窗口是 128k tokens（约9万字），这就意味着你一次性喂给它一整本小说它也能读懂；但一旦对话内容超出这个窗口，它就会“失忆”或者拒绝回答。
- **怎么实现**: 受限于注意力机制（Attention）的计算复杂度，早期模型窗口很小（如 4k）。目前通过 RoPE 旋转位置编码插值、FlashAttention 等技术，上下文窗口已扩展到 1M（百万级）甚至更多。

### 系统提示词 | System Prompt
- **定义**: 在对话开始前发给大模型的指令，具有最高优先级，用于设定模型的身份背景、工作行为准则、回复格式以及安全底线。
- **大白话**: “大模型的人设与教条”。就像在演员上台前塞给它的剧本大纲：“你现在是一个极其严厉的英语老师，不准用中文，每次回答必须先纠正对方的语法错误。”
- **怎么实现**: 在聊天 API 的 `messages` 列表中，添加一条 `role` 为 `system` 的消息。
- **代码**:
  ```python
  messages = [
      # 设定系统人设
      {"role": "system", "content": "你是一个只使用 JSON 格式回复的技术专家。"},
      # 用户提问
      {"role": "user", "content": "简述什么是 API。"}
  ]
  ```

### 温度与核采样 | Temperature & Top-P
- **定义**: 调节大模型输出随机性与创造力的两个关键采样参数。`Temperature`（温度）控制词表概率分布的平滑度；`Top-P`（核采样/累计概率）控制模型候选词的选择范围。
- **大白话**:
  - **温度**: “AI 喝酒的多少”。0 度表示滴酒不沾，AI 极度清醒，每次都说最稳重、标准的话；1 度表示微醺，AI 脑洞大开，开始写诗、讲段子，但也更容易胡言乱语（幻觉）。
  - **Top-P**: “候选词筛选框”。0.9 表示只从前 90% 最靠谱的词里选，剩下的 10% 奇葩词直接扔掉。
- **代码**:
  ```python
  # API 调用参数配置
  response = openai.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": "写一首关于宇宙的诗"}],
      temperature=0.8, # 稍微带点创意
      top_p=0.95       # 保证用词的基本规范
  )
  ```

### 词元 | Token
- **定义**: 大模型处理和理解文本的基本单位。它可以是一个汉字、一个英文单词的一部分（Subword）或一个标点符号。
- **大白话**: “大模型的识字卡片”。大模型并不直接认识汉字或英文字母，而是将文本切碎成一个个 Token，然后把每个 Token 换成数字 ID 进行计算。通常 100 个英文单词约等于 130 个 Token，而 100 个汉字约等于 100~200 个 Token。
- **代码**:
  ```python
  import tiktoken

  # 获取 gpt-4 的分词器
  enc = tiktoken.encoding_for_model("gpt-4")
  text = "Hello world! 人工智能"
  
  # 分词
  tokens = enc.encode(text)
  print(f"Token IDs: {tokens}")
  print(f"Token 数量: {len(tokens)}")
  ```

---

## 微调、对齐与安全 | Fine-Tuning & Safety

### 模型微调 | Fine-Tuning
- **定义**: 在一个已经预训练好的大基座模型（Base Model）基础上，使用特定的下游任务数据（如客服对话、代码库）进行二次训练，以优化模型在特定领域表现的过程。常见的高效微调技术有 LoRA、QLoRA。
- **大白话**: “岗前培训”。基座模型就像一个刚毕业的大学生，什么都知道一点，但干不好具体的专业活。通过收集几万条公司内部账单和业务问答对他进行微调，他就能变成合格的公司专属客服。

### 人类反馈强化学习 | RLHF & RLAIF
- **定义**: 通过收集人类对模型输出的偏好排序，训练一个奖励模型（Reward Model），然后利用强化学习（如 PPO 算法）引导大模型迎合人类偏好，输出更安全、更有用的回答（RLHF）。RLAIF 则是用更强大、更守规矩的 AI 充当裁判来代替人类评分。
- **大白话**: “大模型的家教与训导”。大模型刚训出来时像个野孩子，满嘴脏话或者胡乱编造。RLHF 就是让家教（人类/AI）对它的多次回答进行打分（“这个回答好，得10分；那个涉嫌歧视，扣10分”），逼它慢慢改掉坏习惯。

### 护栏与安全边界 | Guardrails
- **定义**: 在大模型输入端和输出端设立的拦截、过滤与校验框架（如 Llama Guard、NVIDIA NeMo Guardrails），用于拦截有害的输入（如投毒攻击、越狱提示词）和过滤敏感的输出（如暴力、色情、商业机密泄露）。
- **大白话**: “AI 的保镖兼质检员”。不管大模型自己多想胡说八道，保镖在用户提问时会检查“这是不是越狱攻击？”，在模型回答时会检查“这句是不是敏感言论？”，一旦发现违规，直接用复读机式的提示语替换掉原文。
- **代码**:
  ```python
  # 简易输出端 Guardrails 示例
  def output_guardrail(ai_response):
      sensitive_words = ["涉密密码", "越狱技巧", "制作炸弹"]
      for word in sensitive_words:
          if word in ai_response:
              return "对不起，根据安全策略，我无法提供该内容。"
      return ai_response
  ```
