# rag-doc-agent

企业知识库文档检索智能体 —— 对着芯片数据手册、设计文档、测试报告提问，答案带页码溯源。

面向半导体研发知识库场景：文档格式混杂（PDF 表格、参数列表、公式和正文混排），工程师需要精确数值和页码级溯源。

## 功能特性

- **文档入库** —— PDF / TXT / Markdown → 分块 → ChromaDB 向量库
- **两种分块策略** —— 定长重叠切分（基线）和句子感知打包
- **有据回答** —— 模型严格基于检索到的上下文作答，标注 `[source:page]`
- **工具调用检索** —— 首轮检索不够时，Agent 可自主发起二次检索
- **多轮对话** —— 跨轮维护会话历史
- **溯源追踪** —— 每条回答列出文件、页码、相似度分数
- **HTTP API** —— FastAPI 封装（`/ask`、`/ingest`、`/stats`、`/healthz`）
- **回归评测** —— 210 条标注评测集 + 离线基线 vs 优化对比报告

## 架构

```
PDF/TXT ──► 入库 ──► 分块 ──► ChromaDB (cosine)
                                    │
问题 ──► 检索 top-k ────────────────┤
                                    ▼
                          Anthropic Messages API
                          (系统提示 + 上下文 + 工具)
                                    │
                                    ▼
                          回答 + [source:page] 页码溯源
```

## 安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置
cp .env.example .env
# 编辑 .env：填入 ANTHROPIC_API_KEY

# 3. 放文档
mkdir -p data/docs
# 把 PDF / .txt / .md 文件放进 data/docs/
```

## 使用

```bash
# 入库整个目录
python -m rag_agent.cli ingest --path ./data/docs

# 用句子感知分块入库
python -m rag_agent.cli ingest --path ./data/docs --strategy sentence

# 问一个问题
python -m rag_agent.cli query "输出电压精度是多少？"

# 交互式对话
python -m rag_agent.cli chat

# 查看配置 / 索引统计
python -m rag_agent.cli stats
```

## 示例会话

```
$ python -m rag_agent.cli query "Vout 精度是多少？"

╭─ 回答 ──────────────────────────────────────╮
│ Vout 精度为 ±2%，覆盖全温度范围            │
│ [demo_datasheet.pdf:4]。                     │
╰──────────────────────────────────────────────╯

╭─ 来源 ─────────────────────────────────────╮
│ 文件                │ 页码 │ 分数 │         │
│ demo_datasheet.pdf  │    4 │ 0.823 │        │
│ demo_datasheet.pdf  │    3 │ 0.791 │        │
╰──────────────────────────────────────────────╯
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `ANTHROPIC_API_KEY` | —（必填） | Anthropic API 密钥 |
| `RAG_MODEL` | `claude-sonnet-4-5` | 生成模型 |
| `RAG_CHUNK_SIZE` | `1024` | 每块最大字符数 |
| `RAG_CHUNK_OVERLAP` | `128` | 块间重叠字符数 |
| `RAG_TOP_K` | `5` | 每次查询检索的段落数 |
| `RAG_DB_PATH` | `./data/chroma` | ChromaDB 持久化目录 |
| `RAG_COLLECTION` | `chip_docs` | 集合名称 |

## 项目结构

```
src/rag_agent/
├── config.py       # 环境变量配置、校验
├── chunking.py     # 定长切分和句子感知切分
├── ingest.py       # PDF/TXT 加载与入库流水线
├── vectorstore.py  # ChromaDB 封装（增 / 搜 / 重置）
├── agent.py        # 带工具调用检索闭环的 RAG Agent
├── api.py          # FastAPI HTTP 接口
└── cli.py          # CLI：入库 / 问答 / 对话 / 统计

eval/
├── build_dataset.py  # 重新生成标注评测集
├── run_eval.py       # 离线基线 vs 优化对比报告
└── dataset.jsonl     # 210 条标注数据

tests/
├── test_chunking.py
└── test_agent.py

data/sample_docs/   # 演示用 YC-3300 合成数据手册
```

## HTTP API

```bash
uvicorn rag_agent.api:app --reload
```

| 方法 | 路径 | 请求体 |
|------|------|--------|
| GET | `/healthz` | — |
| POST | `/ask` | `{"question": "...", "history": [{"role":"user","content":"..."}]}` |
| POST | `/ingest` | `{"path": "./data/docs", "strategy": "fixed"}` |
| GET | `/stats` | — |

## 评测

```bash
python eval/build_dataset.py     # 重新生成 dataset.jsonl（210 条）
python eval/run_eval.py          # 基线 vs 优化，离线
```

`run_eval.py` 对比弱配置（定长分块、无重叠、top_k=3）与调优配置（句子分块、overlap=128、top_k=5），
输出 `eval/report.json`，含三项指标：检索命中率、多跳来源完整率、无据拒答正确率。
不需要 API Key —— 只跑检索链路。

## 测试

```bash
pytest tests/ -v
```

单元测试覆盖分块边界（重叠、中文标点、空输入）、配置校验、Agent 输入校验。测试无网络调用。

## 设计说明

- **为什么默认定长分块：** 芯片数据手册混排表格、参数列表和正文。句子切分会把表格单元格拆散；定长加重叠是稳健基线。`--strategy sentence` 适合散文类文档。
- **为什么用 cosine 距离：** 向量按方向比较而非大小 —— 语义检索的标准做法。
- **为什么用工具调用检索：** 首轮 top-k 没命中答案时，模型可以发起精确检索，而不是编造。
- **如何换 embedding：** 只有 `vectorstore.py` 碰 embedding 函数。换 API 型 embedding 只改这一个文件。

## 许可证

MIT
