# linmem

本地记忆搜索 CLI 工具，基于 BM25 + [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)（Tri-Graph + PageRank）。

[English](README.md)

## 安装

```bash
uv pip install -e .
# PDF 支持
uv pip install -e ".[pdf]"
# LLM 问答
uv pip install -e ".[llm]"
```

## 使用

```bash
# 索引文档目录
linmem index ./docs/

# 搜索
linmem search "关键词"

# 问答（需要配置 LLM）
linmem ask "你的问题"

# 查看索引状态
linmem status
```

## 致谢

本项目的核心检索算法基于 [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) 改造实现，感谢原作者的开源贡献。

> Zhuang, Y., Chen, Z., et al. "LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora." *ICLR 2026*. [arXiv:2510.10114](https://arxiv.org/abs/2510.10114)

## 许可证

本项目基于 LinearRAG 衍生，遵循与原项目一致的 [GPL-3.0](LICENSE) 协议发布。
