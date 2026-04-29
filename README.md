# DcCA Multi-Modal Benchmark Suite

This repository provides a benchmarking framework to evaluate Deep Canonical Correlation Analysis (DcCA) and other representation learning techniques against traditional indexing and search methods across multiple data modalities.

## Modalities Covered

1. **Tabular**: Compares DcCA (Table2Vec + FAISS) against DuckDB (SQL Scan) and SciPy (KD-Tree).
2. **Temporal**: Compares DcCA (Time2Vec + FAISS) against `tslearn` (DTW Scan) and iSAX (Hamming distance).
3. **Graph**: Compares DcCA (GCN + FAISS) against Neo4j (BFS Search).
4. **Image**: Compares DcCA (ResNet18 + FAISS) against dHash (FastBit) and DuckDB-Blob (Linear Scan).
5. **Text**: Compares DcCA (SBERT + FAISS) against DuckDB-FTS (Keyword) and Whoosh (BM25).
6. **MultiModal**: Late fusion of Tabular, Text, and Image data using DcCA.

## Installation

Ensure you have the necessary dependencies installed:

```bash
pip install torch torchvision numpy faiss-cpu pandas duckdb networkx sentence-transformers tslearn scikit-learn whoosh scipy
```

*Note: Depending on your system setup, you might prefer `faiss-gpu` over `faiss-cpu` for accelerated vector search.*

## Usage

To run the full benchmark suite across all modalities:

```bash
python benchmark_modalities.py
```

This script orchestrates the benchmark execution via `benchmark_worker.py` and saves the consolidated performance metrics to `db_benchmark_results.csv`.

### Running a Specific Modality

You can also run the benchmark for a single modality:

```bash
python benchmark_worker.py [Tabular|Temporal|Graph|Image|Text|MultiModal]
```

## Metrics Evaluated

The framework captures execution times (in milliseconds) for the following phases:
- **Preprocessing (ms)**: Feature extraction and representation learning.
- **Indexing (ms)**: Index construction or database table setup.
- **Search P1 (ms)**: Initial query execution (e.g., top-k k-NN search).
- **Collective P2 (ms)**: Secondary collective search/refinement phase.
