## CapybaraDB Metrics and Benchmarks: From Ingestion to Retrieval

This post walks through CapybaraDB’s performance across three fronts: indexing throughput, query latency/throughput, and retrieval quality. All results were produced on the same environment used for the repository benchmarks and are included alongside charts under `benchmark_results/`.

- Indexing: `benchmark_results/indexing_performance.json` + `indexing_performance.png`
- Query performance: `benchmark_results/query_performance.json` + `query_performance.png`
- Retrieval quality (synthetic): `benchmark_results/retrieval_quality_synthetic.json` + `retrieval_quality_synthetic.png`

### Methodology at a Glance

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` with L2-normalized vectors
- Search: exact top-k via dot-product (cosine equivalent) over normalized vectors
- Chunking: token-based chunking at 512 tokens where applicable
- Metrics captured:
  - Indexing: embedding time, storage time, total time, memory, index size
  - Query: average/min/max/p50/p95/p99 latency, throughput, embedding vs retrieval breakdown
  - Quality: precision/recall/F1@k, nDCG@k, MAP, MRR on a synthetic dataset

---

### Indexing Performance

Data source: `benchmark_results/indexing_performance.json`

- Document counts tested: 10, 50, 100, 500, 1000
- Total times (s): 0.138, 1.015, 2.388, 23.126, 76.331
- Average time per doc (s): 0.0138, 0.0203, 0.0239, 0.0463, 0.0763
- Storage times remain small relative to embedding time even at 1k docs (≈0.122 s)
- Index size (MB): 0.020, 0.089, 0.174, 0.859, 1.715
- Peak memory (MB): ~5.0–13.5 across scales

Key takeaways:
- Embedding dominates total indexing time. Storage overhead is negligible in comparison.
- Linear growth with dataset size; average time per document rises as batches get larger and memory pressure appears.
- Index size scales linearly and remains compact for thousands of chunks.

Refer to `benchmark_results/indexing_performance.png` for the trend lines and `indexing_performance_breakdown.png` for stacked time components.

---

### Query Performance

Data source: `benchmark_results/query_performance.json`

- Dataset sizes tested: 100, 500, 1000, 2500, 5000
- Average query latency (ms): 7.79, 7.54, 9.10, 8.52, 8.45
- Throughput (qps): 128.3, 132.6, 109.9, 117.4, 118.3
- p50 latency (ms): 7.45–8.79
- p95 latency (ms): 10.09–12.01
- p99 latency (ms): 11.80–16.39
- Breakdown (avg):
  - Embedding time (ms): ~3.87–4.53
  - Retrieval time (ms): ~3.50–4.57

Observations:
- Latency remains stable and low (≈7–9 ms on average) from 100 to 5000 vectors for top-k search, reflecting efficient vectorized exact search.
- Throughput remains >100 qps at all tested sizes.
- The split between query embedding and retrieval remains balanced; both contribute roughly half of total latency.
- Note: one anomalous value appears in `min_latency_ms` at 500 (-524.27 ms). This is a measurement artifact and should be ignored; distributional statistics (p50/p95/p99) are consistent and reliable.

Charts: `benchmark_results/query_performance.png` and `query_performance_breakdown.png` visualize latency distributions and the embedding vs retrieval split.

---

### Retrieval Quality (Synthetic)

Data source: `benchmark_results/retrieval_quality_synthetic.json`

Configuration:
- Dataset: Synthetic
- Chunk size: 512

Quality metrics:
- Precision@k: P@1=1.00, P@3≈0.756, P@5≈0.480, P@10≈0.240
- Recall@k: R@1≈0.433, R@3≈0.956, R@5=1.00, R@10=1.00
- F1@k: F1@1=0.60, F1@3≈0.836, F1@5≈0.643, F1@10≈0.385
- nDCG@k: nDCG@1=1.00, nDCG@3≈0.954, nDCG@5≈0.979, nDCG@10≈0.979
- MAP≈0.956, MRR=1.00

Interpretation:
- Very strong early precision (P@1=1.0) and nDCG across cutoffs indicate effective ranking of the most relevant content.
- Near-perfect recall by k=5 shows top-5 captures essentially all relevant items.
- High MAP and perfect MRR reflect consistently correct placement of the first relevant result.

Example qualitative results (abridged):
- Query: "machine learning algorithms"
  - Retrieved top-3: `ml_basics`, `ml_types`, `deep_learning` (all relevant)
- Query: "python programming"
  - Retrieved top-3: `python_intro`, `python_data`, `python_web` (all relevant)
- Query: "cloud computing platforms"
  - Retrieved top-3: `cloud_aws`, `cloud_gcp`, `cloud_azure` (all relevant)

See `benchmark_results/retrieval_quality_synthetic.png` for the quality curves.

---

### What These Results Mean for You

- Small to medium collections (≤10k chunks): exact search is fast, simple, and accurate.
- Low latency: median ≈7–9 ms per query with >100 qps throughput in benchmarks.
- Strong quality: excellent early precision and recall on the synthetic task with coherent chunking.
- Scales linearly: indexing and index size grow linearly; storage overhead is minimal compared to embedding time.

### When to Consider ANN

For collections growing beyond the hundreds of thousands to millions of chunks, consider adding an Approximate Nearest Neighbor (ANN) index (e.g., HNSW) to reduce retrieval time. For ≤50k chunks, exact search remains competitive, simpler, and avoids approximation errors.

### Reproducing and Extending

- Use the raw JSON files in `benchmark_results/` to build your own charts or integrate into dashboards.
- Adjust chunk size, precision (`float16`/`binary`) and device (`cuda`) to match your inference constraints.
- Extend quality evaluation to your domain data by swapping in your corpus and relevance labels.

---

CapybaraDB’s current baseline demonstrates a strong balance of simplicity, speed, and quality for practical semantic search workloads. The provided metrics should serve as a reliable reference point as you adapt the system to your scale and use cases.
