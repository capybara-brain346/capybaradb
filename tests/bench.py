import time
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from capybaradb import CapybaraDB


SAMPLE_DOCUMENTS = [
    "Artificial intelligence is transforming how we interact with technology.",
    "Machine learning algorithms can identify patterns in large datasets.",
    "Neural networks are inspired by biological neural networks in the brain.",
    "Deep learning has revolutionized computer vision and natural language processing.",
    "Python is one of the most popular programming languages for data science.",
    "Vector databases enable efficient similarity search over high-dimensional data.",
    "Embeddings capture semantic meaning in a numerical representation.",
    "Transformers have become the dominant architecture in modern NLP.",
    "Retrieval augmented generation combines search with language models.",
    "Semantic search understands the meaning behind queries rather than just keywords.",
    "Cosine similarity measures the angle between two vectors in space.",
    "GPU acceleration can significantly speed up neural network training.",
    "Fine-tuning adapts pre-trained models to specific tasks.",
    "Tokenization is the process of breaking text into smaller units.",
    "Attention mechanisms allow models to focus on relevant parts of input.",
    "Batch processing improves computational efficiency for large datasets.",
    "Normalization helps stabilize training and improve model convergence.",
    "Dimensionality reduction techniques like PCA compress high-dimensional data.",
    "Cross-validation helps evaluate model performance on unseen data.",
    "Hyperparameter tuning optimizes model configuration for better results.",
]

QUERY_SAMPLES = [
    "How does AI change technology?",
    "What are neural networks?",
    "Tell me about vector search",
    "How do transformers work?",
    "What is semantic search?",
]


class BenchmarkRunner:
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.process = psutil.Process()

    def get_memory_usage_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def get_file_size_mb(self, file_path: Path) -> float:
        if file_path.exists():
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0

    def benchmark_indexing(
        self,
        precision: str = "float32",
        device: str = "cpu",
        chunk_size: int = 512,
        num_docs: int = None,
    ) -> Dict[str, Any]:
        collection_name = f"bench_idx_{precision}_{device}_{chunk_size}"
        db = CapybaraDB(
            collection=collection_name,
            chunking=True,
            chunk_size=chunk_size,
            precision=precision,
            device=device,
        )
        db.clear()

        docs_to_index = (
            SAMPLE_DOCUMENTS if num_docs is None else SAMPLE_DOCUMENTS[:num_docs]
        )

        mem_before = self.get_memory_usage_mb()
        start_time = time.perf_counter()

        for doc in docs_to_index:
            db.add_document(doc)

        index_time = time.perf_counter() - start_time
        mem_after = self.get_memory_usage_mb()
        mem_used = mem_after - mem_before

        storage_path = Path("data") / f"{collection_name}.npz"
        disk_size = self.get_file_size_mb(storage_path)

        num_vectors = db.index.total_chunks
        avg_time_per_doc = index_time / len(docs_to_index) * 1000

        db.clear()

        return {
            "num_documents": len(docs_to_index),
            "num_vectors": num_vectors,
            "total_time_sec": round(index_time, 4),
            "avg_time_per_doc_ms": round(avg_time_per_doc, 2),
            "memory_used_mb": round(mem_used, 2),
            "disk_size_mb": round(disk_size, 4),
            "precision": precision,
            "device": device,
            "chunk_size": chunk_size,
        }

    def benchmark_search(
        self,
        precision: str = "float32",
        device: str = "cpu",
        chunk_size: int = 512,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        collection_name = f"bench_search_{precision}_{device}_{chunk_size}"
        db = CapybaraDB(
            collection=collection_name,
            chunking=True,
            chunk_size=chunk_size,
            precision=precision,
            device=device,
        )
        db.clear()

        for doc in SAMPLE_DOCUMENTS:
            db.add_document(doc)

        query_times = []
        similarities = []

        for query in QUERY_SAMPLES:
            start_time = time.perf_counter()
            results = db.search(query, top_k=top_k)
            query_time = time.perf_counter() - start_time

            query_times.append(query_time * 1000)

            if results:
                similarities.extend([r["score"] for r in results])

        db.clear()

        return {
            "num_queries": len(QUERY_SAMPLES),
            "avg_query_latency_ms": round(np.mean(query_times), 2),
            "min_query_latency_ms": round(np.min(query_times), 2),
            "max_query_latency_ms": round(np.max(query_times), 2),
            "std_query_latency_ms": round(np.std(query_times), 2),
            "avg_similarity_score": round(np.mean(similarities), 4)
            if similarities
            else 0,
            "qps": round(len(QUERY_SAMPLES) / (sum(query_times) / 1000), 2),
            "top_k": top_k,
            "precision": precision,
            "device": device,
        }

    def benchmark_serialization(
        self, precision: str = "float32", device: str = "cpu"
    ) -> Dict[str, Any]:
        collection_name = f"bench_serial_{precision}_{device}"
        db = CapybaraDB(
            collection=collection_name,
            chunking=True,
            chunk_size=512,
            precision=precision,
            device=device,
        )
        db.clear()

        for doc in SAMPLE_DOCUMENTS:
            db.add_document(doc)

        start_time = time.perf_counter()
        db.save()
        save_time = time.perf_counter() - start_time

        storage_path = Path("data") / f"{collection_name}.npz"
        file_size = self.get_file_size_mb(storage_path)

        db_load = CapybaraDB(
            collection=collection_name,
            chunking=True,
            chunk_size=512,
            precision=precision,
            device=device,
        )
        db_load.clear()

        start_time = time.perf_counter()
        db_load.load()
        load_time = time.perf_counter() - start_time

        db.clear()
        db_load.clear()

        return {
            "save_time_ms": round(save_time * 1000, 2),
            "load_time_ms": round(load_time * 1000, 2),
            "file_size_mb": round(file_size, 4),
            "num_vectors": len(SAMPLE_DOCUMENTS),
            "precision": precision,
            "device": device,
        }

    def benchmark_scalability(
        self,
        precision: str = "float32",
        device: str = "cpu",
        scale_factors: List[int] = None,
    ) -> List[Dict[str, Any]]:
        if scale_factors is None:
            scale_factors = [100, 500, 1000, 2000, 5000]

        results = []

        for num_docs in scale_factors:
            collection_name = f"bench_scale_{precision}_{device}_{num_docs}"
            db = CapybaraDB(
                collection=collection_name,
                chunking=False,
                precision=precision,
                device=device,
            )
            db.clear()

            docs = [
                SAMPLE_DOCUMENTS[i % len(SAMPLE_DOCUMENTS)] for i in range(num_docs)
            ]

            start_time = time.perf_counter()
            for doc in docs:
                db.add_document(doc)
            index_time = time.perf_counter() - start_time

            query_times = []
            for query in QUERY_SAMPLES[:3]:
                start_time = time.perf_counter()
                db.search(query, top_k=5)
                query_time = time.perf_counter() - start_time
                query_times.append(query_time * 1000)

            storage_path = Path("data") / f"{collection_name}.npz"
            disk_size = self.get_file_size_mb(storage_path)

            results.append(
                {
                    "num_vectors": num_docs,
                    "index_time_sec": round(index_time, 4),
                    "avg_query_latency_ms": round(np.mean(query_times), 2),
                    "disk_size_mb": round(disk_size, 4),
                    "memory_per_1k_vectors_mb": round((disk_size / num_docs) * 1000, 4),
                }
            )

            db.clear()

        return results

    def benchmark_precision_comparison(self) -> Dict[str, Any]:
        precisions = ["float32", "float16", "binary"]
        results = {}

        for precision in precisions:
            try:
                idx_result = self.benchmark_indexing(precision=precision, device="cpu")
                search_result = self.benchmark_search(precision=precision, device="cpu")
                serial_result = self.benchmark_serialization(
                    precision=precision, device="cpu"
                )

                results[precision] = {
                    "indexing": idx_result,
                    "search": search_result,
                    "serialization": serial_result,
                }
            except Exception as e:
                print(f"Error benchmarking precision {precision}: {e}")
                results[precision] = {"error": str(e)}

        return results

    def benchmark_chunk_size_comparison(self) -> Dict[str, Any]:
        chunk_sizes = [256, 512, 1024]
        results = {}

        for chunk_size in chunk_sizes:
            try:
                idx_result = self.benchmark_indexing(
                    precision="float32", device="cpu", chunk_size=chunk_size
                )
                search_result = self.benchmark_search(
                    precision="float32", device="cpu", chunk_size=chunk_size
                )

                results[f"chunk_{chunk_size}"] = {
                    "indexing": idx_result,
                    "search": search_result,
                }
            except Exception as e:
                print(f"Error benchmarking chunk_size {chunk_size}: {e}")
                results[f"chunk_{chunk_size}"] = {"error": str(e)}

        return results

    def benchmark_device_comparison(self) -> Dict[str, Any]:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        results = {}

        for device in devices:
            try:
                idx_result = self.benchmark_indexing(precision="float32", device=device)
                search_result = self.benchmark_search(
                    precision="float32", device=device
                )

                results[device] = {
                    "indexing": idx_result,
                    "search": search_result,
                }
            except Exception as e:
                print(f"Error benchmarking device {device}: {e}")
                results[device] = {"error": str(e)}

        return results

    def run_all_benchmarks(self) -> Dict[str, Any]:
        print("=" * 80)
        print("CAPYBARADB BENCHMARK SUITE")
        print("=" * 80)

        print("\n1. Running Indexing Benchmark...")
        self.results["indexing"] = self.benchmark_indexing()
        print(
            f"   ✓ Indexed {self.results['indexing']['num_documents']} documents in {self.results['indexing']['total_time_sec']}s"
        )

        print("\n2. Running Search Benchmark...")
        self.results["search"] = self.benchmark_search()
        print(
            f"   ✓ Avg query latency: {self.results['search']['avg_query_latency_ms']}ms"
        )

        print("\n3. Running Serialization Benchmark...")
        self.results["serialization"] = self.benchmark_serialization()
        print(
            f"   ✓ Save: {self.results['serialization']['save_time_ms']}ms, Load: {self.results['serialization']['load_time_ms']}ms"
        )

        print("\n4. Running Scalability Benchmark...")
        self.results["scalability"] = self.benchmark_scalability()
        print(f"   ✓ Tested {len(self.results['scalability'])} scale points")

        print("\n5. Running Precision Comparison...")
        self.results["precision_comparison"] = self.benchmark_precision_comparison()
        print(
            f"   ✓ Compared {len(self.results['precision_comparison'])} precision modes"
        )

        print("\n6. Running Chunk Size Comparison...")
        self.results["chunk_size_comparison"] = self.benchmark_chunk_size_comparison()
        print(f"   ✓ Compared {len(self.results['chunk_size_comparison'])} chunk sizes")

        print("\n7. Running Device Comparison...")
        self.results["device_comparison"] = self.benchmark_device_comparison()
        print(f"   ✓ Compared {len(self.results['device_comparison'])} devices")

        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)

        return self.results

    def save_results(self, filename: str = "benchmark_results.json"):
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    def print_summary(self):
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        if "indexing" in self.results:
            print("\n📊 INDEXING METRICS")
            idx = self.results["indexing"]
            print(
                f"  • Total time: {idx['total_time_sec']}s for {idx['num_documents']} documents"
            )
            print(f"  • Avg per document: {idx['avg_time_per_doc_ms']}ms")
            print(f"  • Memory used: {idx['memory_used_mb']}MB")
            print(f"  • Disk size: {idx['disk_size_mb']}MB")

        if "search" in self.results:
            print("\n⚡ SEARCH METRICS")
            search = self.results["search"]
            print(f"  • Avg query latency: {search['avg_query_latency_ms']}ms")
            print(f"  • Throughput: {search['qps']} QPS")
            print(f"  • Avg similarity: {search['avg_similarity_score']}")

        if "serialization" in self.results:
            print("\n💾 STORAGE METRICS")
            serial = self.results["serialization"]
            print(f"  • Save time: {serial['save_time_ms']}ms")
            print(f"  • Load time: {serial['load_time_ms']}ms")
            print(f"  • File size: {serial['file_size_mb']}MB")

        if "precision_comparison" in self.results:
            print("\n🔬 PRECISION COMPARISON")
            for precision, data in self.results["precision_comparison"].items():
                if "error" not in data:
                    idx = data["indexing"]
                    search = data["search"]
                    print(f"  • {precision}:")
                    print(f"    - Memory: {idx['memory_used_mb']}MB")
                    print(f"    - Query latency: {search['avg_query_latency_ms']}ms")
                    print(f"    - Disk size: {data['serialization']['file_size_mb']}MB")

        if "scalability" in self.results:
            print("\n📈 SCALABILITY")
            for point in self.results["scalability"]:
                print(f"  • {point['num_vectors']} vectors:")
                print(f"    - Index time: {point['index_time_sec']}s")
                print(f"    - Query latency: {point['avg_query_latency_ms']}ms")
                print(f"    - Disk size: {point['disk_size_mb']}MB")


def main():
    runner = BenchmarkRunner()

    results = runner.run_all_benchmarks()

    runner.print_summary()

    runner.save_results()


if __name__ == "__main__":
    main()
