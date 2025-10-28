import time
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from capybaradb import CapybaraDB


def get_file_size(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / 1024 / 1024
    return 0


def generate_sample_documents(n_docs, doc_length=100):
    document_templates = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
        "Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and binding, make it very attractive for rapid application development and scripting.",
        "Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, analytics, and intelligence over the Internet to offer faster innovation and flexible resources.",
        "Deep learning is part of machine learning methods based on artificial neural networks with representation learning. It can be supervised, semi-supervised or unsupervised and is used in computer vision and natural language processing.",
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data. It employs techniques from statistics, machine learning and data analysis.",
        "Web development refers to the tasks associated with developing websites and web applications for the internet. It involves web design, content development, client-side scripting, server-side scripting, and network security configuration.",
        "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. These attacks usually aim to access, change, or destroy sensitive information, extort money from users, or interrupt normal business processes.",
        "Blockchain technology is a decentralized, distributed ledger that records transactions across many computers. The record cannot be altered retroactively without altering all subsequent blocks and the consensus of the network.",
        "Natural language processing enables computers to understand, interpret and manipulate human language. It draws from many disciplines including computer science and computational linguistics to bridge the gap between human communication and computer understanding.",
        "Quantum computing harnesses quantum mechanical phenomena to process information. By exploiting superposition and entanglement, quantum computers can solve certain problems exponentially faster than classical computers.",
        "Internet of Things describes the network of physical objects embedded with sensors, software, and other technologies for connecting and exchanging data with other devices and systems over the internet.",
        "Artificial intelligence refers to the simulation of human intelligence in machines programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with learning and problem-solving.",
        "DevOps is a set of practices that combines software development and IT operations. It aims to shorten the systems development life cycle and provide continuous delivery with high software quality through automation and monitoring.",
        "Big data analytics examines large amounts of data to uncover hidden patterns, correlations and insights. It uses advanced analytic techniques against very large, diverse data sets that include structured, semi-structured and unstructured data.",
        "Containerization involves encapsulating an application and its dependencies into a container that can run on any computing environment. This ensures consistency across development, testing, and production environments.",
        "Microservices architecture is an approach to developing a single application as a suite of small services, each running in its own process and communicating with lightweight mechanisms. Each service is built around business capabilities.",
        "Database management systems are software applications that interact with end users, applications, and the database itself to capture and analyze data. They provide facilities to administer and maintain database structures.",
        "Version control systems are tools that help software teams manage changes to source code over time. They keep track of every modification to the code in a special kind of database and enable developers to collaborate efficiently.",
        "API design involves creating interfaces that enable different software applications to communicate with each other. Well-designed APIs are easy to use, reliable, and provide clear documentation for developers.",
        "Software testing is the process of evaluating and verifying that a software product or application does what it is supposed to do. Testing helps identify bugs, gaps, or missing requirements versus actual requirements.",
    ]

    documents = []
    for i in range(n_docs):
        base_template = document_templates[i % len(document_templates)]
        variation = i // len(document_templates)

        if variation > 0:
            doc = f"{base_template} This document (version {variation}) provides additional context and information about the topic, including practical examples and use cases in modern software development."
        else:
            doc = base_template

        documents.append(doc)

    return documents


def generate_queries(n_queries=10):
    queries = [
        "machine learning artificial intelligence",
        "python programming language features",
        "cloud computing services",
        "deep learning neural networks",
        "data science methods and techniques",
        "web development frameworks",
        "cybersecurity protection methods",
        "blockchain distributed ledger",
        "natural language processing applications",
        "quantum computing advantages",
        "internet of things devices",
        "AI problem solving capabilities",
        "DevOps continuous integration",
        "big data analytics insights",
        "containerization benefits",
        "microservices architecture patterns",
        "database management tools",
        "version control collaboration",
        "API design best practices",
        "software testing strategies",
    ]
    return queries[:n_queries]


def benchmark_query_performance():
    print("=" * 80)
    print("QUERY PERFORMANCE BENCHMARK")
    print("=" * 80)

    collection_name = "benchmark_query"
    storage_path = Path("data") / f"{collection_name}.npz"

    if storage_path.exists():
        os.remove(storage_path)

    dataset_sizes = [100, 500, 1000, 2500, 5000]
    n_test_queries = 1000
    top_k = 5

    results = {
        "dataset_sizes": [],
        "avg_query_latency_ms": [],
        "avg_embedding_time_ms": [],
        "avg_retrieval_time_ms": [],
        "throughput_queries_per_sec": [],
        "min_latency_ms": [],
        "max_latency_ms": [],
        "p50_latency_ms": [],
        "p95_latency_ms": [],
        "p99_latency_ms": [],
    }

    for n_docs in dataset_sizes:
        print(f"\n{'=' * 80}")
        print(f"Testing with dataset size: {n_docs} documents")
        print(f"{'=' * 80}")

        if storage_path.exists():
            os.remove(storage_path)

        db = CapybaraDB(collection=collection_name, chunking=False, device="cpu")
        db.clear()

        print(f"  Building index with {n_docs} documents...")
        documents = generate_sample_documents(n_docs)
        for i, doc in enumerate(documents):
            db.add_document(doc)
            if (i + 1) % 500 == 0:
                print(f"    Indexed {i + 1}/{n_docs} documents...")

        db.save()
        index_size = get_file_size(storage_path)
        print(f"  Index built. Size: {index_size:.2f} MB")

        queries = generate_queries(20)
        warmup_queries = queries[:10]

        print(f"  Running warmup phase ({len(warmup_queries)} queries)...")
        for query in warmup_queries:
            db.search(query, top_k=top_k)
        print(f"  Warmup complete. Model and cache are now warm.")

        test_queries = queries * (n_test_queries // len(queries))

        query_latencies = []
        embedding_times = []
        retrieval_times = []

        print(f"  Running {len(test_queries)} test queries (warm cache)...")

        for i, query in enumerate(test_queries):
            query_start = time.time()

            embedding_start = time.time()
            query_embedding = db.model.embed(query)
            embedding_time = time.time() - embedding_start

            retrieval_start = time.time()
            results_list = db.search(query, top_k=top_k)
            retrieval_time = time.time() - retrieval_start

            total_query_time = time.time() - query_start

            query_latencies.append(total_query_time * 1000)
            embedding_times.append(embedding_time * 1000)
            retrieval_times.append(retrieval_time * 1000)

            if (i + 1) % 10 == 0:
                print(f"    Completed {i + 1}/{len(test_queries)} queries...")

        avg_latency = np.mean(query_latencies)
        avg_embedding = np.mean(embedding_times)
        avg_retrieval = np.mean(retrieval_times)
        throughput = 1000 / avg_latency
        min_latency = np.min(query_latencies)
        max_latency = np.max(query_latencies)
        p50_latency = np.percentile(query_latencies, 50)
        p95_latency = np.percentile(query_latencies, 95)
        p99_latency = np.percentile(query_latencies, 99)

        results["dataset_sizes"].append(n_docs)
        results["avg_query_latency_ms"].append(avg_latency)
        results["avg_embedding_time_ms"].append(avg_embedding)
        results["avg_retrieval_time_ms"].append(avg_retrieval)
        results["throughput_queries_per_sec"].append(throughput)
        results["min_latency_ms"].append(min_latency)
        results["max_latency_ms"].append(max_latency)
        results["p50_latency_ms"].append(p50_latency)
        results["p95_latency_ms"].append(p95_latency)
        results["p99_latency_ms"].append(p99_latency)

        print(f"\n  Results for dataset size {n_docs}:")
        print(f"    Average query latency: {avg_latency:.2f} ms")
        print(f"    Average embedding time: {avg_embedding:.2f} ms")
        print(f"    Average retrieval time: {avg_retrieval:.2f} ms")
        print(f"    Throughput: {throughput:.2f} queries/sec")
        print(f"    Min latency: {min_latency:.2f} ms")
        print(f"    Max latency: {max_latency:.2f} ms")
        print(f"    P50 latency: {p50_latency:.2f} ms")
        print(f"    P95 latency: {p95_latency:.2f} ms")
        print(f"    P99 latency: {p99_latency:.2f} ms")

    if storage_path.exists():
        os.remove(storage_path)

    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    json_output = output_dir / "query_performance.json"
    with open(json_output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to {json_output}")
    print(f"{'=' * 80}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Query Performance Benchmark", fontsize=16, fontweight="bold")

    axes[0, 0].plot(
        results["dataset_sizes"],
        results["avg_query_latency_ms"],
        "o-",
        linewidth=2,
        markersize=8,
        color="blue",
    )
    axes[0, 0].set_xlabel("Dataset Size (documents)", fontsize=12)
    axes[0, 0].set_ylabel("Latency (ms)", fontsize=12)
    axes[0, 0].set_title(
        "Average Query Latency vs Dataset Size", fontsize=14, fontweight="bold"
    )
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        results["dataset_sizes"],
        results["avg_retrieval_time_ms"],
        "o-",
        linewidth=2,
        markersize=8,
        color="green",
    )
    axes[0, 1].set_xlabel("Dataset Size (documents)", fontsize=12)
    axes[0, 1].set_ylabel("Time (ms)", fontsize=12)
    axes[0, 1].set_title(
        "Retrieval Time vs Dataset Size", fontsize=14, fontweight="bold"
    )
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(
        results["dataset_sizes"],
        results["throughput_queries_per_sec"],
        "o-",
        linewidth=2,
        markersize=8,
        color="red",
    )
    axes[1, 0].set_xlabel("Dataset Size (documents)", fontsize=12)
    axes[1, 0].set_ylabel("Queries/Second", fontsize=12)
    axes[1, 0].set_title(
        "Query Throughput vs Dataset Size", fontsize=14, fontweight="bold"
    )
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(
        results["dataset_sizes"],
        results["p50_latency_ms"],
        "o-",
        label="P50",
        linewidth=2,
        markersize=8,
    )
    axes[1, 1].plot(
        results["dataset_sizes"],
        results["p95_latency_ms"],
        "o-",
        label="P95",
        linewidth=2,
        markersize=8,
    )
    axes[1, 1].plot(
        results["dataset_sizes"],
        results["p99_latency_ms"],
        "o-",
        label="P99",
        linewidth=2,
        markersize=8,
    )
    axes[1, 1].set_xlabel("Dataset Size (documents)", fontsize=12)
    axes[1, 1].set_ylabel("Latency (ms)", fontsize=12)
    axes[1, 1].set_title(
        "Latency Percentiles vs Dataset Size", fontsize=14, fontweight="bold"
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_output = output_dir / "query_performance.png"
    plt.savefig(plot_output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_output}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Query Performance - Time Breakdown", fontsize=16, fontweight="bold")

    x_pos = np.arange(len(results["dataset_sizes"]))
    width = 0.35

    axes2[0].bar(
        x_pos - width / 2,
        results["avg_embedding_time_ms"],
        width,
        label="Embedding Time",
        alpha=0.8,
    )
    axes2[0].bar(
        x_pos + width / 2,
        results["avg_retrieval_time_ms"],
        width,
        label="Retrieval Time",
        alpha=0.8,
    )
    axes2[0].set_xlabel("Dataset Size", fontsize=12)
    axes2[0].set_ylabel("Time (ms)", fontsize=12)
    axes2[0].set_title("Embedding vs Retrieval Time", fontsize=14, fontweight="bold")
    axes2[0].set_xticks(x_pos)
    axes2[0].set_xticklabels([f"{n}" for n in results["dataset_sizes"]])
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3, axis="y")

    axes2[1].plot(
        results["dataset_sizes"],
        results["min_latency_ms"],
        "o-",
        label="Min",
        linewidth=2,
        markersize=8,
    )
    axes2[1].plot(
        results["dataset_sizes"],
        results["avg_query_latency_ms"],
        "o-",
        label="Average",
        linewidth=2,
        markersize=8,
    )
    axes2[1].plot(
        results["dataset_sizes"],
        results["max_latency_ms"],
        "o-",
        label="Max",
        linewidth=2,
        markersize=8,
    )
    axes2[1].set_xlabel("Dataset Size (documents)", fontsize=12)
    axes2[1].set_ylabel("Latency (ms)", fontsize=12)
    axes2[1].set_title("Latency Min/Avg/Max Comparison", fontsize=14, fontweight="bold")
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_output2 = output_dir / "query_performance_breakdown.png"
    plt.savefig(plot_output2, dpi=300, bbox_inches="tight")
    print(f"Breakdown plot saved to {plot_output2}")

    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    print(
        f"Dataset size range: {min(results['dataset_sizes'])} - {max(results['dataset_sizes'])} documents"
    )
    print(
        f"Latency range: {min(results['avg_query_latency_ms']):.2f}ms - {max(results['avg_query_latency_ms']):.2f}ms"
    )
    print(
        f"Throughput range: {min(results['throughput_queries_per_sec']):.2f} - {max(results['throughput_queries_per_sec']):.2f} queries/sec"
    )
    print(
        f"Retrieval time range: {min(results['avg_retrieval_time_ms']):.2f}ms - {max(results['avg_retrieval_time_ms']):.2f}ms"
    )
    print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    benchmark_query_performance()
