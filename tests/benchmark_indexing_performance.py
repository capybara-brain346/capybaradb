import time
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import psutil
import tracemalloc
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from capybaradb import CapybaraDB


def warmup_model(db, n_warmup_docs=5):
    warmup_docs = generate_sample_documents(n_warmup_docs)
    for doc in warmup_docs:
        db.add_document(doc)
    db.clear()


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


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


def benchmark_indexing_performance():
    print("=" * 80)
    print("INDEXING PERFORMANCE BENCHMARK")
    print("=" * 80)

    collection_name = "benchmark_indexing"
    storage_path = Path("data") / f"{collection_name}.npz"

    if storage_path.exists():
        os.remove(storage_path)

    test_sizes = [10, 50, 100, 500, 1000]
    results = {
        "document_counts": [],
        "embedding_times": [],
        "storage_times": [],
        "total_times": [],
        "memory_usage_mb": [],
        "peak_memory_mb": [],
        "index_size_mb": [],
        "avg_time_per_doc": [],
    }

    print("\nPerforming model warmup...")
    warmup_db = CapybaraDB(collection="warmup_temp", chunking=False, device="cpu")
    warmup_model(warmup_db, n_warmup_docs=10)
    warmup_path = Path("data") / "warmup_temp.npz"
    if warmup_path.exists():
        os.remove(warmup_path)
    print("Warmup complete. Starting benchmarks with warm model.\n")

    for n_docs in test_sizes:
        print(f"\n{'=' * 80}")
        print(f"Testing with {n_docs} documents")
        print(f"{'=' * 80}")

        if storage_path.exists():
            os.remove(storage_path)

        tracemalloc.start()
        initial_memory = get_memory_usage()

        db = CapybaraDB(collection=collection_name, chunking=False, device="cpu")
        db.clear()

        documents = generate_sample_documents(n_docs)

        start_time = time.time()
        embedding_start = time.time()

        for i, doc in enumerate(documents):
            db.add_document(doc)
            if (i + 1) % 10 == 0 or (i + 1) == n_docs:
                print(f"  Processed {i + 1}/{n_docs} documents...")

        embedding_time = time.time() - embedding_start

        storage_start = time.time()
        db.save()
        storage_time = time.time() - storage_start

        total_time = time.time() - start_time

        current_memory = get_memory_usage()
        memory_used = current_memory - initial_memory

        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / 1024 / 1024
        tracemalloc.stop()

        index_size = get_file_size(storage_path)

        avg_time = total_time / n_docs

        results["document_counts"].append(n_docs)
        results["embedding_times"].append(embedding_time)
        results["storage_times"].append(storage_time)
        results["total_times"].append(total_time)
        results["memory_usage_mb"].append(memory_used)
        results["peak_memory_mb"].append(peak_memory_mb)
        results["index_size_mb"].append(index_size)
        results["avg_time_per_doc"].append(avg_time)

        print(f"\n  Results for {n_docs} documents:")
        print(f"    Total time: {total_time:.4f}s")
        print(f"    Embedding time: {embedding_time:.4f}s")
        print(f"    Storage time: {storage_time:.4f}s")
        print(f"    Memory used: {memory_used:.2f} MB")
        print(f"    Peak memory: {peak_memory_mb:.2f} MB")
        print(f"    Index size: {index_size:.2f} MB")
        print(f"    Avg time/doc: {avg_time:.4f}s")

    if storage_path.exists():
        os.remove(storage_path)

    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    json_output = output_dir / "indexing_performance.json"
    with open(json_output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to {json_output}")
    print(f"{'=' * 80}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Indexing Performance Benchmark", fontsize=16, fontweight="bold")

    axes[0, 0].plot(
        results["document_counts"],
        results["total_times"],
        "o-",
        linewidth=2,
        markersize=8,
    )
    axes[0, 0].set_xlabel("Number of Documents", fontsize=12)
    axes[0, 0].set_ylabel("Time (seconds)", fontsize=12)
    axes[0, 0].set_title("Total Indexing Time", fontsize=14, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        results["document_counts"],
        results["peak_memory_mb"],
        "o-",
        color="orange",
        linewidth=2,
        markersize=8,
    )
    axes[0, 1].set_xlabel("Number of Documents", fontsize=12)
    axes[0, 1].set_ylabel("Memory (MB)", fontsize=12)
    axes[0, 1].set_title(
        "Peak Memory Usage During Indexing", fontsize=14, fontweight="bold"
    )
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(
        results["document_counts"],
        results["index_size_mb"],
        "o-",
        color="green",
        linewidth=2,
        markersize=8,
    )
    axes[1, 0].set_xlabel("Number of Documents", fontsize=12)
    axes[1, 0].set_ylabel("Size (MB)", fontsize=12)
    axes[1, 0].set_title("Serialized Index Size", fontsize=14, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(
        results["document_counts"],
        results["avg_time_per_doc"],
        "o-",
        color="red",
        linewidth=2,
        markersize=8,
    )
    axes[1, 1].set_xlabel("Number of Documents", fontsize=12)
    axes[1, 1].set_ylabel("Time (seconds)", fontsize=12)
    axes[1, 1].set_title("Average Time per Document", fontsize=14, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_output = output_dir / "indexing_performance.png"
    plt.savefig(plot_output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_output}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(
        "Indexing Performance - Time Breakdown", fontsize=16, fontweight="bold"
    )

    x_pos = np.arange(len(results["document_counts"]))
    width = 0.35

    axes2[0].bar(
        x_pos - width / 2,
        results["embedding_times"],
        width,
        label="Embedding Time",
        alpha=0.8,
    )
    axes2[0].bar(
        x_pos + width / 2,
        results["storage_times"],
        width,
        label="Storage Time",
        alpha=0.8,
    )
    axes2[0].set_xlabel("Test Size", fontsize=12)
    axes2[0].set_ylabel("Time (seconds)", fontsize=12)
    axes2[0].set_title("Embedding vs Storage Time", fontsize=14, fontweight="bold")
    axes2[0].set_xticks(x_pos)
    axes2[0].set_xticklabels([f"{n} docs" for n in results["document_counts"]])
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3, axis="y")

    axes2[1].plot(
        results["document_counts"],
        results["memory_usage_mb"],
        "o-",
        label="Memory Used",
        linewidth=2,
        markersize=8,
    )
    axes2[1].plot(
        results["document_counts"],
        results["peak_memory_mb"],
        "o-",
        label="Peak Memory",
        linewidth=2,
        markersize=8,
    )
    axes2[1].set_xlabel("Number of Documents", fontsize=12)
    axes2[1].set_ylabel("Memory (MB)", fontsize=12)
    axes2[1].set_title("Memory Usage Comparison", fontsize=14, fontweight="bold")
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_output2 = output_dir / "indexing_performance_breakdown.png"
    plt.savefig(plot_output2, dpi=300, bbox_inches="tight")
    print(f"Breakdown plot saved to {plot_output2}")

    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    print(
        f"Document range tested: {min(results['document_counts'])} - {max(results['document_counts'])}"
    )
    print(
        f"Time range: {min(results['total_times']):.4f}s - {max(results['total_times']):.4f}s"
    )
    print(
        f"Peak memory range: {min(results['peak_memory_mb']):.2f} MB - {max(results['peak_memory_mb']):.2f} MB"
    )
    print(
        f"Index size range: {min(results['index_size_mb']):.2f} MB - {max(results['index_size_mb']):.2f} MB"
    )
    print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    benchmark_indexing_performance()
