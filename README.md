# DcCA: Data-Centric Counterfactual Analysis

This repository accompanies **Towards Data-Centric Counterfactual Analysis**. It contains timing benchmarks for feature representation, indexing, retrieval, and collective candidate selection across tabular, temporal, graph, image, text, and multimodal data.


## Modalities Covered


| Modality | Representation and vector search | Comparison operations |
| --- | --- | --- |
| Tabular | Randomly initialized multilayer perceptron followed by FAISS search | DuckDB label filtering with `LIMIT 100`; SciPy `cKDTree` nearest-neighbor search |
| Temporal | Randomly initialized linear projection with a sine transform, followed by FAISS search | `tslearn` DTW scan; piecewise averaging and symbolization followed by a NumPy Hamming-distance scan |
| Graph | Adjacency-based feature aggregation with random weights and ReLU, followed by FAISS search | NetworkX traversal to collect nodes within two hops |
| Image | A ResNet18 forward pass is timed; separately generated random vectors are indexed and searched with FAISS | Hamming-distance scan over random binary vectors; binary-file read followed by a NumPy Euclidean-distance scan |
| Text | SentenceTransformer embeddings using `paraphrase-MiniLM-L3-v2`, followed by FAISS search | scikit-learn TF-IDF with cosine similarity; Whoosh search using its default scoring |
| MultiModal | Tabular, text, and random image vectors are projected to 512 dimensions and averaged before FAISS search; a separate ResNet18 forward pass is included in preprocessing time | No separate baseline |

All vector-search branches use CPU FAISS `IndexFlatL2` for exact squared-Euclidean nearest-neighbor search. Each retrieves 100 candidates for the first record, then times greedy farthest-first selection of five candidates using Euclidean distance in the representation space.

### Output Labels

| CSV label | Actual implementation |
| --- | --- |
| `Table2Vec` / `Time2Vec` / `GCN` | The simplified, untrained transformations described above |
| `Neo4j-VEC` / `Neo4j` | FAISS / NetworkX; no Neo4j connection |
| `iSAX` | Symbolization and a NumPy scan; no iSAX index |
| `dHash` / `FastBit` | Random binary vectors and a NumPy scan; no image-derived dHash or FastBit index |
| `DuckDB-Blob` | A local binary-file read and NumPy distance calculation |
| `DuckDB-FTS` | scikit-learn TF-IDF and cosine similarity |
| `BM25` | Whoosh's default scoring configuration |

## Installation

Install the dependencies in a Python environment:

```bash
python -m pip install torch torchvision numpy faiss-cpu pandas duckdb networkx sentence-transformers tslearn scikit-learn whoosh scipy
```

The worker imports these packages for every modality. Text and multimodal runs may download the SentenceTransformer model on first use. The scripts use CPU execution; installing a GPU package alone does not enable GPU search.

## Usage

Run commands from the repository directory. Use a dedicated working copy: the text benchmark recreates and removes `indexdir`, and the image benchmark writes and removes `temp_images.bin`.

To run all six modalities:

```bash
python benchmark_modalities.py
```

The driver launches each modality through `benchmark_worker.py` and writes successful output rows to `db_benchmark_results.csv`. Check the console for worker errors: the driver continues after failures, so a CSV file can contain only a subset of the results.

To run one modality, for example:

```bash
python benchmark_worker.py Tabular
```

Supported names are case-sensitive: `Tabular`, `Temporal`, `Graph`, `Image`, `Text`, and `MultiModal`. A direct worker invocation prints CSV rows to standard output; it does not create the consolidated CSV file.

## Timing Metrics

| Metric | Meaning |
| --- | --- |
| **Preprocessing (ms)** | Timed representation computation or synthetic feature preparation. Model initialization and downloads are outside this timer. |
| **Indexing (ms)** | Timed index construction, database setup, or the corresponding setup operation in that branch. |
| **Search P1 (ms)** | Timed vector retrieval, filtering, traversal, or scan, depending on the method. |
| **Collective P2 (ms)** | Timed greedy farthest-first selection for the vector-search branches. Other rows report `N/A`. |

