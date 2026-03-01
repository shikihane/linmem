# linmem — 本地记忆搜索 CLI 工具

## Context

基于前期调研结论：BM25 + LinearRAG（Tri-Graph + PageRank）是本地低成本记忆搜索的最优方案。LinearRAG（ICLR 2026）提供了零 LLM token 的图增强检索能力，但其代码是研究原型，存在硬编码、无错误处理、不支持中文、无 CLI 等问题。

本项目走路线 A：基于 LinearRAG 代码改造封装，GPL-3.0 协议发布。

## 目标

一个本地运行的 CLI 记忆搜索工具：
- `linmem index <目录>` — 索引文档（BM25 + NER + Tri-Graph + Embedding）
- `linmem search <查询>` — BM25 预筛 → LinearRAG 图增强排序 → 输出 Top-K
- `linmem ask <问题>` — search + 送入 LLM 生成答案
- `linmem status` — 查看索引状态

## 架构

```
文档目录
  ↓
[摄入层] PDF/TXT/MD → 分块 → 去重
  ↓                ↓
[索引层] SQLite FTS5   spaCy NER → Tri-Graph (igraph)
         (BM25)        sentence-transformers → Embedding (numpy/Parquet)
  ↓
[检索层] BM25 Top-100 → LinearRAG 两阶段排序
         ① 实体激活（稀疏矩阵语义桥接）
         ② Personalized PageRank
         → Top-K 结果
  ↓
[生成层] (可选) 本地/远程 LLM 生成答案
```

## 项目结构

```
E:/Heyang5/linmem/
├── PLAN.md             # 本文件
├── README.md           # 项目说明
├── LICENSE             # GPL-3.0
├── pyproject.toml      # 项目配置（用 uv 管理）
├── src/
│   └── linmem/
│       ├── __init__.py
│       ├── cli.py          # CLI 入口（argparse）
│       ├── config.py       # 配置（从 LinearRAG config.py 改造）
│       ├── ingest.py       # 文档摄入 + 分块（新增）
│       ├── bm25.py         # SQLite FTS5 BM25（新增）
│       ├── ner.py          # NER（从 LinearRAG ner.py 改造）
│       ├── embedding.py    # Embedding 存储（从 LinearRAG embedding_store.py 改造）
│       ├── graph.py        # Tri-Graph 构建 + 检索核心（从 LinearRAG LinearRAG.py 拆分）
│       ├── retriever.py    # 编排层：BM25 → LinearRAG 两阶段（新增）
│       ├── llm.py          # LLM 调用（从 LinearRAG utils.py 提取）
│       └── utils.py        # 工具函数
└── tests/
    └── ...
```

## 基于 LinearRAG 改造清单

### Phase 1: 项目骨架 + 去硬编码

**从 LinearRAG 代码改造：**

| 源文件 | 改造内容 |
|--------|---------|
| `LinearRAG.py` (~650行) | 拆分为 `graph.py`（核心算法）+ `retriever.py`（编排）。去掉 `CUDA_VISIBLE_DEVICES="4"` 硬编码，`device` 改为自动检测。去掉 `{idx}:` 前缀耦合，段落邻接边用列表顺序。去掉 retrieve 必须有 gold answer 的要求。 |
| `config.py` | 扩展：加入 BM25 配置、文档路径、语言选择、LLM 端点配置 |
| `ner.py` | 修复 `ZeroDivisionError`，修复 O(n²) 的 keys→list 转换，加错误处理 |
| `embedding_store.py` | 改名 `embedding.py`，加 CPU fallback，去掉 `pdb` import |
| `utils.py` | LLM 调用提取到 `llm.py`，加重试逻辑。`normalize_answer()` 改为语言感知 |
| `evaluate.py` | 暂不移植，后续按需 |
| `run.py` | 废弃，用 `cli.py` 替代 |

### Phase 2: 新增功能

1. **文档摄入 (`ingest.py`)**
   - 支持 TXT / MD / PDF（用 `pymupdf` 或 `pdfplumber`）
   - 分块策略：按段落/固定长度 + overlap
   - hash 去重，增量摄入

2. **BM25 索引 (`bm25.py`)**
   - SQLite FTS5，Python 内置 `sqlite3`，零额外依赖
   - `insert(hash_id, text)` / `search(query, top_k)` / `delete(hash_id)`

3. **CLI (`cli.py`)**
   - 用 `argparse`（无额外依赖）
   - 子命令：index / search / ask / status

4. **中文支持**
   - 配置项 `language: zh` 自动选择 `zh_core_web_sm` + `bge-small-zh-v1.5`
   - 配置项 `language: en` 自动选择 `en_core_web_trf` + `all-mpnet-base-v2`

### Phase 3: 增量索引

- Embedding store：已有 hash 去重（继承 LinearRAG）
- NER 缓存：已有增量支持（继承 LinearRAG）
- **Tri-Graph 增量**：新增文档时只 `add_vertices` + `add_edges`，不重建全图
- BM25：SQLite FTS5 天然支持增量 INSERT
- 文档删除：从各存储中按 hash_id 级联删除

## 依赖

```
必需：
  spacy >= 3.6          # NER
  sentence-transformers  # Embedding
  python-igraph         # Tri-Graph + PageRank
  torch                 # Embedding 计算（CPU 模式也需要）
  numpy / pandas / pyarrow  # 数据存储

可选：
  pymupdf 或 pdfplumber  # PDF 支持
  openai                 # LLM 问答（ask 命令）

不需要（LinearRAG 中的废弃依赖）：
  scikit-learn           # 从未使用
  scipy                  # 从未使用
```

## LinearRAG 核心算法参考（路线 A 移植重点）

### Tri-Graph 构建
- 三层节点：实体(Ve) / 句子(Vs) / 段落(Vp)
- 两个稀疏矩阵：Mention(句子↔实体) + Contain(段落↔实体)
- spaCy NER 抽取实体，标点切句，零 LLM 调用

### 两阶段检索
1. **实体激活**：`a_q^t = MAX(σ_q^T · M · a_q^{t-1}, a_q^{t-1})`
   - 稀疏矩阵乘法 + 阈值剪枝，n 跳只需 n 次迭代
2. **PageRank 排序**：`igraph.Graph.personalized_pagerank()`
   - 激活实体作为种子，传播到段落层

### 关键超参
| 参数 | 默认值 | 含义 |
|------|--------|------|
| max_iterations | 3 | 语义桥接最大迭代轮次 |
| iteration_threshold | 0.4 | 实体激活剪枝阈值 |
| passage_ratio (λ) | 0.05 | DPR 分量权重（很小，实体信号为主） |
| damping | 0.85 | PageRank 阻尼系数 |
| retrieval_top_k | 5 | 最终返回段落数 |

## 验证

1. **索引测试**：准备 10 个中文 TXT 文件 → `linmem index ./test-docs/` → 检查 SQLite + Parquet + GraphML 生成
2. **搜索测试**：`linmem search "某个关键词"` → 确认 BM25 + 图增强排序输出合理结果
3. **多跳测试**：准备跨文档关联的内容 → 验证语义桥接能发现跨文档关联
4. **增量测试**：新增文档 → 再次 index → 确认增量更新而非全量重建
5. **ask 测试**：配置 LLM 端点 → `linmem ask "问题"` → 确认生成答案引用了正确段落

## 实施顺序

1. 创建项目目录 + git init + 基础文件（LICENSE, pyproject.toml, README）
2. Phase 1：移植 LinearRAG 核心 + 去硬编码（最小可运行）
3. Phase 2：新增 BM25 + 文档摄入 + CLI
4. Phase 3：增量索引
5. 中文适配 + 测试

## 已知风险

- **中文 NER 精度**：spaCy zh_core_web_sm F1=68%，zh_core_web_trf F1=74%，是最薄弱环节
  - 后续可考虑 HanLP / 百度 LAC 替代
- **内存瓶颈**：全量 embedding 加载到 RAM，100K 段落 ≈ 2.4GB
  - 后续可引入 FAISS 做近似最近邻
- **GPL-3.0 协议**：衍生作品必须同样 GPL，可接受
