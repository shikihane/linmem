# linmem 项目记忆

## 项目概览

linmem 是一个基于 BM25 + LinearRAG（Tri-Graph + PageRank）的本地记忆搜索 CLI 工具。主要面向中文文档的索引和检索。

- 语言: Python 3.10+
- 构建: hatchling
- 入口: `linmem` CLI（`src/linmem/cli.py:main`）
- 许可: GPL-3.0-or-later

## 架构

```
文档 → [Ingest] chunk_text → [Index] BM25 + NER + Embedding + TriGraph → [Retrieve] BM25预筛 → LinearRAG两阶段排序 → [可选] LLM生成
```

11 个源文件在 `src/linmem/`：cli, config, ingest, bm25, ner, embedding, graph, retriever, llm, utils, __init__

## 2026-03-01 测试与部署实施 — 完成报告

### 执行的计划

来源: `docs/plans/2026-03-01-testing-deployment.md`

### 完成的任务（6/6）

| 任务 | 提交 | 内容 |
|------|------|------|
| 1. 修复模型默认值 | `47c8968` | config.py: en_core_web_trf → en_core_web_sm（CPU友好） |
| 2. 添加 pytest 配置 | `9931654` | pyproject.toml: 添加 dev 依赖和 pytest markers |
| 3. 单元测试 | `f672488` | 50 个纯逻辑测试，覆盖 utils/config/bm25/ingest/graph |
| 4. 测试语料 | `ac80e7a` | 5 篇中文文档（虚构公司"星辰科技"，实体交叉覆盖） |
| 5. 集成测试 | `3146af1` | 4 个 @pytest.mark.slow 测试（NER/全流程/增量/CLI） |
| 6. CLI E2E 验证 | — | linmem index/search/status 均正常工作 |

### 发现并修复的 Bug

1. **BM25 FTS5 删除触发器**（`bm25.py`）: 使用了外部内容表的删除语法导致 `OperationalError`，改为 `DELETE FROM chunks_fts WHERE rowid = old.rowid`
2. **BM25 FTS5 分词器**（`bm25.py`）: `unicode61` 无法分词中文，改为 `trigram` 分词器
3. **Windows 路径兼容**: test_config.py 中 index_dir 断言改为 Path 对象比较
4. **CLI 参数顺序**: `--data-dir` 是全局参数，必须放在子命令之前

### 测试结果

- 快速测试: `pytest tests/ -m "not slow"` → **50 passed**
- 慢速测试: `pytest tests/ -m slow` → **4 passed**
- 全部测试: `pytest tests/` → **54 passed**

### 模型依赖

| 模型 | 用途 | 大小 |
|------|------|------|
| `zh_core_web_sm` | spaCy 中文 NER | 75 MB |
| `BAAI/bge-small-zh-v1.5` | 句向量嵌入 | 184 MB（含缓存） |

注意: 在代理环境下 HuggingFace 模型下载需设置 `HF_ENDPOINT=https://hf-mirror.com`

### 常用命令

```bash
# 测试
pytest tests/ -m "not slow"    # 快速测试（无模型依赖）
pytest tests/ -m slow           # 慢速测试（需要模型）
pytest tests/                   # 全部测试

# CLI
linmem index <目录>             # 索引文档
linmem search "<查询>"          # 搜索
linmem status                   # 查看索引状态

# 安装
pip install -e ".[dev]"         # 开发模式安装
```
