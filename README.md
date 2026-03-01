# linmem

Local memory search CLI tool powered by BM25 + [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) (Tri-Graph + PageRank).

[中文说明](README_CN.md)

## Installation

```bash
uv pip install -e .
# PDF support
uv pip install -e ".[pdf]"
# LLM Q&A
uv pip install -e ".[llm]"
```

## Usage

```bash
# Index a document directory
linmem index ./docs/

# Search
linmem search "keyword"

# Q&A (requires LLM configuration)
linmem ask "your question"

# Check index status
linmem status
```

## Acknowledgements

The core retrieval algorithm is based on [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG). Thanks to the original authors for their open-source contribution.

> Zhuang, Y., Chen, Z., et al. "LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora." *ICLR 2026*. [arXiv:2510.10114](https://arxiv.org/abs/2510.10114)

## License

This project is derived from LinearRAG and released under the same [GPL-3.0](LICENSE) license.
