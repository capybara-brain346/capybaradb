import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse

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


def load_cisi_documents(cisi_all_path):
    documents = {}
    current_id = None
    current_field = None
    current_text = []

    with open(cisi_all_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith(".I "):
                if current_id is not None and current_text:
                    documents[current_id] = " ".join(current_text).strip()

                current_id = line.split()[1]
                current_field = None
                current_text = []

            elif line.startswith(".T") or line.startswith(".W"):
                current_field = "text"
            elif (
                line.startswith(".A") or line.startswith(".B") or line.startswith(".X")
            ):
                current_field = "skip"
            elif current_field == "text" and line.strip():
                current_text.append(line.strip())

        if current_id is not None and current_text:
            documents[current_id] = " ".join(current_text).strip()

    return documents


def load_cisi_queries(cisi_qry_path):
    queries = {}
    current_id = None
    current_field = None
    current_text = []

    with open(cisi_qry_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith(".I "):
                if current_id is not None and current_text:
                    queries[current_id] = " ".join(current_text).strip()

                current_id = line.split()[1]
                current_field = None
                current_text = []

            elif line.startswith(".W"):
                current_field = "text"
            elif (
                line.startswith(".T") or line.startswith(".A") or line.startswith(".B")
            ):
                current_field = "skip"
            elif current_field == "text" and line.strip():
                current_text.append(line.strip())

        if current_id is not None and current_text:
            queries[current_id] = " ".join(current_text).strip()

    return queries


def load_cisi_relevance(cisi_rel_path):
    relevance = defaultdict(list)

    with open(cisi_rel_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                query_id = parts[0]
                doc_id = parts[1]
                relevance[query_id].append(doc_id)

    return relevance


def check_cisi_dataset(cisi_dir):
    cisi_path = Path(cisi_dir)
    required_files = ["CISI.ALL", "CISI.QRY", "CISI.REL"]

    if not cisi_path.exists():
        return False, f"Directory {cisi_dir} does not exist"

    missing_files = []
    for file in required_files:
        if not (cisi_path / file).exists():
            missing_files.append(file)

    if missing_files:
        return False, f"Missing required files: {', '.join(missing_files)}"

    return True, "CISI dataset found"


def print_cisi_instructions():
    instructions = """
    ================================================================================
    CISI DATASET NOT FOUND
    ================================================================================
    
    To use the CISI dataset for evaluation, please follow these steps:
    
    1. Download the CISI dataset from Kaggle:
       https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval
    
    2. Extract the dataset to a directory (e.g., ./cisi_dataset/)
    
    3. Ensure the following files are present:
       - CISI.ALL (1,460 documents)
       - CISI.QRY (112 queries)
       - CISI.REL (relevance judgments)
    
    4. Run the benchmark with:
       python benchmark_retrieval_quality.py --dataset cisi --cisi-dir ./cisi_dataset/
    
    Alternatively, you can run with the synthetic dataset (default):
       python benchmark_retrieval_quality.py --dataset synthetic
    
    ================================================================================
    """
    print(instructions)


def calculate_metrics(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_k)

    true_positives = len(retrieved_set.intersection(relevant_set))

    precision_at_k = true_positives / k if k > 0 else 0
    recall_at_k = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0

    f1_at_k = 0
    if precision_at_k + recall_at_k > 0:
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

    return precision_at_k, recall_at_k, f1_at_k


def calculate_average_precision(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    num_relevant = len(relevant_set)

    precision_sum = 0.0
    num_relevant_retrieved = 0

    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            num_relevant_retrieved += 1
            precision_at_i = num_relevant_retrieved / i
            precision_sum += precision_at_i

    if num_relevant_retrieved == 0:
        return 0.0

    return precision_sum / num_relevant


def calculate_reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)

    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            return 1.0 / i

    return 0.0


def calculate_dcg(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k, 1):
        if doc_id in relevant_set:
            rel = 1
            dcg += rel / np.log2(i + 1)

    return dcg


def calculate_ndcg(retrieved_ids, relevant_ids, k):
    dcg = calculate_dcg(retrieved_ids, relevant_ids, k)

    ideal_retrieved = list(relevant_ids)[:k]
    idcg = calculate_dcg(ideal_retrieved, relevant_ids, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def benchmark_retrieval_quality(dataset_type="synthetic", cisi_dir=None):
    print("=" * 80)
    print("RETRIEVAL QUALITY BENCHMARK")
    print("=" * 80)

    if dataset_type == "cisi":
        if cisi_dir is None:
            print("\nError: CISI directory not specified")
            print_cisi_instructions()
            return None

        exists, message = check_cisi_dataset(cisi_dir)
        if not exists:
            print(f"\nError: {message}")
            print_cisi_instructions()
            return None

        print(f"\nLoading CISI dataset from {cisi_dir}...")
        cisi_path = Path(cisi_dir)

        documents_dict = load_cisi_documents(cisi_path / "CISI.ALL")
        queries_dict = load_cisi_queries(cisi_path / "CISI.QRY")
        ground_truth = load_cisi_relevance(cisi_path / "CISI.REL")

        documents = documents_dict

        print(f"  Loaded {len(documents)} documents")
        print(f"  Loaded {len(queries_dict)} queries")
        print(f"  Loaded relevance judgments for {len(ground_truth)} queries")

        dataset_label = "CISI"
    else:
        print("\nUsing synthetic dataset...")
        documents = create_test_corpus()
        queries_dict = {
            str(i): query for i, query in enumerate(create_ground_truth().keys())
        }
        ground_truth = create_ground_truth()

        ground_truth_with_str_keys = {}
        for i, (query, relevant_ids) in enumerate(create_ground_truth().items()):
            ground_truth_with_str_keys[str(i)] = relevant_ids
        ground_truth = ground_truth_with_str_keys

        dataset_label = "Synthetic"

    chunk_size = 512
    k_values = [1, 3, 5, 10]

    results = {
        "dataset": dataset_label,
        "chunk_size": chunk_size,
        "precision_at_k": {},
        "recall_at_k": {},
        "f1_at_k": {},
        "ndcg_at_k": {},
        "map": 0.0,
        "mrr": 0.0,
        "examples": [],
    }

    collection_name = f"benchmark_quality_{dataset_label.lower()}"
    storage_path = Path("data") / f"{collection_name}.npz"

    if storage_path.exists():
        os.remove(storage_path)

    print(f"\n{'=' * 80}")
    print(f"Testing with chunk size: {chunk_size} tokens")
    print(f"{'=' * 80}")

    db = CapybaraDB(
        collection=collection_name,
        chunking=True,
        chunk_size=chunk_size,
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
    all_f1s = defaultdict(list)
    all_ndcgs = defaultdict(list)
    all_aps = []
    all_rrs = []

    for query_id, relevant_ids in ground_truth.items():
        if dataset_type == "cisi":
            query = queries_dict.get(query_id, "")
        else:
            query = queries_dict.get(query_id, "")

        if not query:
            continue

        search_results = db.search(query, top_k=max(k_values))

        retrieved_doc_ids = []
        for result in search_results:
            doc_id = result["doc_id"]
            if doc_id in retrieved_doc_ids:
                continue
            retrieved_doc_ids.append(doc_id)

        ap = calculate_average_precision(retrieved_doc_ids, relevant_ids)
        all_aps.append(ap)

        rr = calculate_reciprocal_rank(retrieved_doc_ids, relevant_ids)
        all_rrs.append(rr)

        for k in k_values:
            precision, recall, f1 = calculate_metrics(
                retrieved_doc_ids, relevant_ids, k
            )
            all_precisions[k].append(precision)
            all_recalls[k].append(recall)
            all_f1s[k].append(f1)

            ndcg = calculate_ndcg(retrieved_doc_ids, relevant_ids, k)
            all_ndcgs[k].append(ndcg)

    num_examples = 3 if dataset_type == "synthetic" else 5
    query_ids_for_examples = list(ground_truth.keys())[:num_examples]

    for query_id in query_ids_for_examples:
        relevant_ids = ground_truth[query_id]
        if dataset_type == "cisi":
            query = queries_dict.get(query_id, "")
        else:
            query = queries_dict.get(query_id, "")

        if not query:
            continue

        search_results = db.search(query, top_k=5)
        retrieved_doc_ids = []
        for result in search_results:
            doc_id = result["doc_id"]
            if doc_id not in retrieved_doc_ids:
                retrieved_doc_ids.append(doc_id)

        print(f"\n  Query ID {query_id}: '{query[:80]}...'")
        print(f"  Relevant documents: {relevant_ids[:10]}")
        print("  Retrieved (top 5):")
        for i, result in enumerate(search_results[:5]):
            doc_id = result["doc_id"]
            score = result["score"]
            is_relevant = "✓ RELEVANT" if doc_id in relevant_ids else "✗ Not relevant"
            text_preview = (
                result["text"][:80] + "..."
                if len(result["text"]) > 80
                else result["text"]
            )
            print(f"    {i + 1}. [{doc_id}] (score: {score:.4f}) {is_relevant}")
            print(f"       {text_preview}")

        example = {
            "chunk_size": chunk_size,
            "query_id": query_id,
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

    mean_ap = np.mean(all_aps) if all_aps else 0.0
    mean_rr = np.mean(all_rrs) if all_rrs else 0.0
    results["map"] = mean_ap
    results["mrr"] = mean_rr

    print("\n  Overall Metrics:")
    print(f"    MAP (Mean Average Precision): {mean_ap:.4f}")
    print(f"    MRR (Mean Reciprocal Rank): {mean_rr:.4f}")

    for k in k_values:
        avg_precision = np.mean(all_precisions[k])
        avg_recall = np.mean(all_recalls[k])
        avg_f1 = np.mean(all_f1s[k])
        avg_ndcg = np.mean(all_ndcgs[k])

        results["precision_at_k"][k] = avg_precision
        results["recall_at_k"][k] = avg_recall
        results["f1_at_k"][k] = avg_f1
        results["ndcg_at_k"][k] = avg_ndcg

        print(f"\n  Metrics for K={k}:")
        print(f"    Precision@{k}: {avg_precision:.4f}")
        print(f"    Recall@{k}: {avg_recall:.4f}")
        print(f"    F1@{k}: {avg_f1:.4f}")
        print(f"    NDCG@{k}: {avg_ndcg:.4f}")

    if storage_path.exists():
        os.remove(storage_path)

    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    json_output = output_dir / f"retrieval_quality_{dataset_label.lower()}.json"
    with open(json_output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to {json_output}")
    print(f"{'=' * 80}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Retrieval Quality Benchmark - {dataset_label} Dataset (Chunk Size: {chunk_size})",
        fontsize=16,
        fontweight="bold",
    )

    k_list = list(k_values)
    x_pos = np.arange(len(k_list))
    width = 0.2

    precision_values = [results["precision_at_k"][k] for k in k_list]
    recall_values = [results["recall_at_k"][k] for k in k_list]
    f1_values = [results["f1_at_k"][k] for k in k_list]
    ndcg_values = [results["ndcg_at_k"][k] for k in k_list]

    ax = axes[0, 0]
    ax.bar(
        x_pos - 1.5 * width,
        precision_values,
        width,
        label="Precision",
        alpha=0.8,
    )
    ax.bar(x_pos - 0.5 * width, recall_values, width, label="Recall", alpha=0.8)
    ax.bar(x_pos + 0.5 * width, f1_values, width, label="F1", alpha=0.8)
    ax.bar(x_pos + 1.5 * width, ndcg_values, width, label="NDCG", alpha=0.8)
    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("All Metrics @K", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"K={k}" for k in k_list])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.0])

    ax = axes[0, 1]
    ax.bar(x_pos, precision_values, width=0.6, alpha=0.8, color="steelblue")
    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision@K", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"K={k}" for k in k_list])
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.0])

    ax = axes[1, 0]
    ax.bar(x_pos, recall_values, width=0.6, alpha=0.8, color="darkorange")
    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Recall@K", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"K={k}" for k in k_list])
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.0])

    ax = axes[1, 1]
    ax.bar(x_pos, ndcg_values, width=0.6, alpha=0.8, color="forestgreen")
    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("NDCG", fontsize=12)
    ax.set_title("NDCG@K", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"K={k}" for k in k_list])
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.0])

    plt.tight_layout()

    plot_output = output_dir / f"retrieval_quality_{dataset_label.lower()}.png"
    plt.savefig(plot_output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_output}")
    plt.close()

    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    print(f"\nDataset: {dataset_label}")
    print(f"Chunk Size: {chunk_size} tokens")
    print("\nOverall Metrics:")
    print(f"  MAP: {results['map']:.4f}")
    print(f"  MRR: {results['mrr']:.4f}")
    print("\nMetrics at Different K Values:")
    for k in k_values:
        precision = results["precision_at_k"][k]
        recall = results["recall_at_k"][k]
        f1 = results["f1_at_k"][k]
        ndcg = results["ndcg_at_k"][k]
        print(
            f"  K={k:2d} - P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, NDCG: {ndcg:.4f}"
        )
    print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval quality using synthetic or CISI dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["synthetic", "cisi"],
        default="synthetic",
        help="Dataset to use for evaluation (default: synthetic)",
    )
    args = parser.parse_args()

    benchmark_retrieval_quality(dataset_type=args.dataset, cisi_dir="./cisi")
