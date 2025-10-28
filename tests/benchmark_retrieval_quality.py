import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from capybaradb import CapybaraDB


def create_test_corpus():
    documents = {
        "ml_basics": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Machine learning algorithms are trained on data sets that contain examples of the desired output.",
        "ml_types": "There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to learn the mapping between input and output. Unsupervised learning finds hidden patterns in unlabeled data. Reinforcement learning trains agents to make sequences of decisions by rewarding desired behaviors.",
        "deep_learning": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. These deep neural networks can automatically learn hierarchical representations of data. Deep learning has revolutionized fields like computer vision, natural language processing, and speech recognition by achieving state-of-the-art results on many benchmark tasks.",
        "python_intro": "Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its notable use of significant whitespace. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "python_data": "Python has become the dominant language for data science and machine learning. Libraries like NumPy, Pandas, and Scikit-learn provide powerful tools for data manipulation and analysis. The Python ecosystem also includes deep learning frameworks such as TensorFlow and PyTorch, making it the go-to language for AI development.",
        "python_web": "Python is widely used for web development through frameworks like Django and Flask. Django is a high-level web framework that encourages rapid development and clean design. Flask is a lightweight microframework that gives developers more control and flexibility. Both frameworks are used by major companies to build scalable web applications.",
        "cloud_aws": "Amazon Web Services (AWS) is a comprehensive cloud computing platform offering over 200 services including computing power, storage, and databases. AWS pioneered cloud computing and remains the market leader. Services like EC2 for virtual servers, S3 for object storage, and Lambda for serverless computing are widely used by organizations of all sizes.",
        "cloud_azure": "Microsoft Azure is a cloud computing platform that provides services for building, deploying, and managing applications. Azure integrates well with Microsoft's existing enterprise tools and offers hybrid cloud capabilities. It provides services across compute, storage, networking, databases, analytics, and AI, competing directly with AWS and Google Cloud.",
        "cloud_gcp": "Google Cloud Platform (GCP) is Google's suite of cloud computing services running on the same infrastructure that Google uses internally. GCP is particularly strong in data analytics, machine learning, and containerization with Kubernetes. BigQuery for data warehousing and TensorFlow for ML are popular GCP offerings.",
        "docker_intro": "Docker is a platform for developing, shipping, and running applications in containers. Containers package an application with all its dependencies, ensuring consistency across different environments. Docker has revolutionized software deployment by making it easy to create, deploy, and run applications in isolated, reproducible environments.",
        "kubernetes": "Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications. Originally developed by Google, Kubernetes has become the industry standard for container orchestration. It provides features like automated rollouts, service discovery, load balancing, and self-healing capabilities.",
        "devops_cicd": "DevOps is a set of practices that combines software development and IT operations to shorten the development lifecycle. Continuous Integration and Continuous Deployment (CI/CD) are core DevOps practices that automate the process of testing and deploying code. Popular CI/CD tools include Jenkins, GitLab CI, GitHub Actions, and CircleCI.",
        "security_basics": "Cybersecurity involves protecting computer systems, networks, and data from digital attacks, unauthorized access, and damage. Key security principles include confidentiality, integrity, and availability (CIA triad). Common security measures include firewalls, encryption, access controls, regular security audits, and employee training on security best practices.",
        "security_crypto": "Cryptography is the practice of securing information through encryption. Modern cryptography uses mathematical algorithms to encrypt and decrypt data. Public-key cryptography, introduced with RSA, allows secure communication without sharing secret keys. Hash functions like SHA-256 ensure data integrity, while digital signatures provide authentication and non-repudiation.",
        "database_sql": "SQL (Structured Query Language) databases are relational databases that store data in tables with predefined schemas. Popular SQL databases include PostgreSQL, MySQL, and Oracle. They use ACID properties (Atomicity, Consistency, Isolation, Durability) to ensure reliable transaction processing. SQL is excellent for complex queries and maintaining data integrity.",
        "database_nosql": "NoSQL databases are non-relational databases designed for distributed data storage and horizontal scaling. Types include document stores (MongoDB), key-value stores (Redis), column-family stores (Cassandra), and graph databases (Neo4j). NoSQL databases sacrifice some consistency guarantees for improved performance and scalability, following the BASE model instead of ACID.",
        "api_rest": "REST (Representational State Transfer) is an architectural style for designing networked applications. RESTful APIs use HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources identified by URLs. REST APIs are stateless, cacheable, and provide a uniform interface. JSON is the most common data format for REST API responses.",
        "api_graphql": "GraphQL is a query language for APIs developed by Facebook. Unlike REST, GraphQL allows clients to request exactly the data they need, reducing over-fetching and under-fetching. GraphQL uses a strongly-typed schema and supports real-time updates through subscriptions. It provides a more efficient and flexible alternative to REST for complex data requirements.",
        "testing_unit": "Unit testing involves testing individual components or functions in isolation to ensure they work correctly. Good unit tests are fast, independent, and repeatable. Popular testing frameworks include JUnit for Java, pytest for Python, and Jest for JavaScript. Unit tests should cover edge cases and validate both expected behavior and error handling.",
        "testing_integration": "Integration testing verifies that different modules or services work together correctly. While unit tests focus on individual components, integration tests check the interactions between components. Integration tests often involve databases, external APIs, and other dependencies. They are slower than unit tests but catch issues that unit tests might miss.",
        "blockchain_basics": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records called blocks. Each block contains a cryptographic hash of the previous block, creating an immutable chain. Blockchain provides transparency, security, and decentralization. While originally created for Bitcoin, blockchain has applications beyond cryptocurrency.",
        "blockchain_ethereum": "Ethereum is a blockchain platform that enables smart contracts and decentralized applications (dApps). Unlike Bitcoin which focuses on peer-to-peer payments, Ethereum provides a Turing-complete programming environment. Smart contracts are self-executing contracts with terms directly written in code. Ethereum's native cryptocurrency is Ether (ETH).",
        "nlp_basics": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. NLP enables computers to read, understand, and generate human language. Applications include machine translation, sentiment analysis, text summarization, and chatbots. Modern NLP heavily relies on deep learning and transformer models.",
        "nlp_transformers": "Transformers are a neural network architecture that has revolutionized NLP. Introduced in the paper 'Attention is All You Need', transformers use self-attention mechanisms to process sequential data. Models like BERT, GPT, and T5 are based on transformer architecture. Transformers enable transfer learning where models pre-trained on large corpora can be fine-tuned for specific tasks.",
        "data_engineering": "Data engineering involves designing and building systems for collecting, storing, and analyzing data at scale. Data engineers create data pipelines that move data from sources to destinations, transforming it along the way. They work with technologies like Apache Spark, Kafka, Airflow, and various database systems to enable data-driven decision making.",
        "data_visualization": "Data visualization is the graphical representation of information and data. Effective visualizations help people understand trends, patterns, and outliers in data. Common visualization types include line charts, bar charts, scatter plots, and heatmaps. Tools like Tableau, PowerBI, and Python libraries (Matplotlib, Seaborn, Plotly) are used to create interactive and insightful visualizations.",
    }
    return documents


def create_ground_truth():
    ground_truth = {
        "machine learning algorithms": ["ml_basics", "ml_types", "deep_learning"],
        "python programming": ["python_intro", "python_data", "python_web"],
        "cloud computing platforms": ["cloud_aws", "cloud_azure", "cloud_gcp"],
        "container technology": ["docker_intro", "kubernetes", "devops_cicd"],
        "cybersecurity and encryption": ["security_basics", "security_crypto"],
        "database systems": ["database_sql", "database_nosql"],
        "API design": ["api_rest", "api_graphql"],
        "software testing": ["testing_unit", "testing_integration"],
        "blockchain technology": ["blockchain_basics", "blockchain_ethereum"],
        "natural language processing": ["nlp_basics", "nlp_transformers"],
        "supervised and unsupervised learning": ["ml_types", "ml_basics"],
        "neural networks and deep learning": ["deep_learning", "nlp_transformers"],
        "data science with python": [
            "python_data",
            "data_engineering",
            "data_visualization",
        ],
        "AWS cloud services": ["cloud_aws", "cloud_gcp", "cloud_azure"],
        "smart contracts": ["blockchain_ethereum", "blockchain_basics"],
    }
    return ground_truth


def calculate_metrics(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_k)

    true_positives = len(retrieved_set.intersection(relevant_set))

    precision_at_k = true_positives / k if k > 0 else 0
    recall_at_k = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0

    return precision_at_k, recall_at_k


def benchmark_retrieval_quality():
    print("=" * 80)
    print("RETRIEVAL QUALITY BENCHMARK")
    print("=" * 80)

    documents = create_test_corpus()
    ground_truth = create_ground_truth()

    chunk_sizes = [None, 256, 512, 1024]
    chunk_size_labels = {
        None: "No Chunking",
        256: "256 tokens",
        512: "512 tokens",
        1024: "1024 tokens",
    }
    k_values = [1, 3, 5, 10]

    results = {
        "chunk_sizes": [],
        "chunk_labels": [],
        "precision_at_k": defaultdict(list),
        "recall_at_k": defaultdict(list),
        "examples": [],
    }

    for chunk_size in chunk_sizes:
        collection_name = f"benchmark_quality_{chunk_size if chunk_size else 'none'}"
        storage_path = Path("data") / f"{collection_name}.npz"

        if storage_path.exists():
            os.remove(storage_path)

        chunk_label = chunk_size_labels[chunk_size]
        print(f"\n{'=' * 80}")
        print(f"Testing with chunk size: {chunk_label}")
        print(f"{'=' * 80}")

        use_chunking = chunk_size is not None
        db = CapybaraDB(
            collection=collection_name,
            chunking=use_chunking,
            chunk_size=chunk_size if chunk_size else 512,
            device="cpu",
        )
        db.clear()

        doc_id_map = {}
        for doc_id, text in documents.items():
            added_id = db.add_document(text, doc_id=doc_id)
            doc_id_map[added_id] = doc_id

        print(f"  Indexed {len(documents)} documents")

        all_precisions = defaultdict(list)
        all_recalls = defaultdict(list)

        for query_idx, (query, relevant_ids) in enumerate(ground_truth.items()):
            search_results = db.search(query, top_k=max(k_values))

            retrieved_doc_ids = []
            for result in search_results:
                doc_id = result["doc_id"]
                if doc_id in retrieved_doc_ids:
                    continue
                retrieved_doc_ids.append(doc_id)

            for k in k_values:
                precision, recall = calculate_metrics(
                    retrieved_doc_ids, relevant_ids, k
                )
                all_precisions[k].append(precision)
                all_recalls[k].append(recall)

            if chunk_size is None and query_idx < 3:
                print(f"\n  Query: '{query}'")
                print(f"  Relevant documents: {relevant_ids}")
                print(f"  Retrieved (top 5):")
                for i, result in enumerate(search_results[:5]):
                    doc_id = result["doc_id"]
                    score = result["score"]
                    is_relevant = (
                        "✓ RELEVANT" if doc_id in relevant_ids else "✗ Not relevant"
                    )
                    text_preview = (
                        result["text"][:100] + "..."
                        if len(result["text"]) > 100
                        else result["text"]
                    )
                    print(f"    {i + 1}. [{doc_id}] (score: {score:.4f}) {is_relevant}")
                    print(f"       {text_preview}")

                example = {
                    "chunk_size": chunk_label,
                    "query": query,
                    "relevant_ids": relevant_ids,
                    "retrieved_ids": retrieved_doc_ids[:5],
                    "results": [
                        {
                            "doc_id": r["doc_id"],
                            "score": float(r["score"]),
                            "is_relevant": r["doc_id"] in relevant_ids,
                            "text_preview": r["text"][:150],
                        }
                        for r in search_results[:5]
                    ],
                }
                results["examples"].append(example)

        results["chunk_sizes"].append(chunk_size if chunk_size else 0)
        results["chunk_labels"].append(chunk_label)

        for k in k_values:
            avg_precision = np.mean(all_precisions[k])
            avg_recall = np.mean(all_recalls[k])
            results["precision_at_k"][k].append(avg_precision)
            results["recall_at_k"][k].append(avg_recall)

            print(f"\n  Metrics for K={k}:")
            print(f"    Average Precision@{k}: {avg_precision:.4f}")
            print(f"    Average Recall@{k}: {avg_recall:.4f}")

        if storage_path.exists():
            os.remove(storage_path)

    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    json_output = output_dir / "retrieval_quality.json"
    results_serializable = {
        "chunk_sizes": results["chunk_sizes"],
        "chunk_labels": results["chunk_labels"],
        "precision_at_k": {k: v for k, v in results["precision_at_k"].items()},
        "recall_at_k": {k: v for k, v in results["recall_at_k"].items()},
        "examples": results["examples"],
    }
    with open(json_output, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to {json_output}")
    print(f"{'=' * 80}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Retrieval Quality Benchmark", fontsize=16, fontweight="bold")

    x_pos = np.arange(len(results["chunk_labels"]))
    width = 0.2

    for i, k in enumerate([1, 3, 5, 10]):
        ax = axes[i // 2, i % 2]

        precision_values = results["precision_at_k"][k]
        recall_values = results["recall_at_k"][k]

        ax.bar(
            x_pos - width / 2,
            precision_values,
            width,
            label=f"Precision@{k}",
            alpha=0.8,
        )
        ax.bar(x_pos + width / 2, recall_values, width, label=f"Recall@{k}", alpha=0.8)

        ax.set_xlabel("Chunk Size", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"Precision and Recall @ K={k}", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results["chunk_labels"], rotation=15, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.0])

    plt.tight_layout()

    plot_output = output_dir / "retrieval_quality.png"
    plt.savefig(plot_output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_output}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(
        "Retrieval Quality - Effect of Chunk Size", fontsize=16, fontweight="bold"
    )

    for k in k_values:
        axes2[0].plot(
            x_pos,
            results["precision_at_k"][k],
            "o-",
            label=f"K={k}",
            linewidth=2,
            markersize=8,
        )
    axes2[0].set_xlabel("Chunk Size", fontsize=12)
    axes2[0].set_ylabel("Precision", fontsize=12)
    axes2[0].set_title("Precision@K vs Chunk Size", fontsize=14, fontweight="bold")
    axes2[0].set_xticks(x_pos)
    axes2[0].set_xticklabels(results["chunk_labels"], rotation=15, ha="right")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    axes2[0].set_ylim([0, 1.0])

    for k in k_values:
        axes2[1].plot(
            x_pos,
            results["recall_at_k"][k],
            "o-",
            label=f"K={k}",
            linewidth=2,
            markersize=8,
        )
    axes2[1].set_xlabel("Chunk Size", fontsize=12)
    axes2[1].set_ylabel("Recall", fontsize=12)
    axes2[1].set_title("Recall@K vs Chunk Size", fontsize=14, fontweight="bold")
    axes2[1].set_xticks(x_pos)
    axes2[1].set_xticklabels(results["chunk_labels"], rotation=15, ha="right")
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    axes2[1].set_ylim([0, 1.0])

    plt.tight_layout()

    plot_output2 = output_dir / "retrieval_quality_chunk_effect.png"
    plt.savefig(plot_output2, dpi=300, bbox_inches="tight")
    print(f"Chunk size effect plot saved to {plot_output2}")

    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    for i, chunk_label in enumerate(results["chunk_labels"]):
        print(f"\n{chunk_label}:")
        for k in k_values:
            precision = results["precision_at_k"][k][i]
            recall = results["recall_at_k"][k][i]
            print(f"  K={k:2d} - Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    benchmark_retrieval_quality()
