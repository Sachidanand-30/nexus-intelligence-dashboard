 Nexus Intelligence — Review Intelligence Platform

Nexus Intelligence — Review Intelligence Platform is an enterprise-grade AI analytics system designed to process, denoise, cluster, and resolve unstructured e-commerce customer feedback. Powered by **LangGraph** for deterministic workflow state management, **LangChain** and **Fireworks AI** for intelligent transformation and extraction, **ChromaDB** for vector storage, and **Scikit-learn (HDBSCAN)** for spatial anomaly clustering, the platform automatically isolates zero-day defects and generates engineering strategies cross-referenced against technical specifications.

---

## Frameworks & Technology Stack Mapping

| Framework / Tool | Specific Component | Role & Function in the System |
| --- | --- | --- |
| **FastAPI & Uvicorn** | REST API & Static Mount | Serves application endpoints, provides CORS middleware, and hosts the static dashboard and storefront clients. |
| **LangGraph** | `StateGraph`, `END` | Orchestrates the multi-node review ingestion pipeline with conditional branching, error gates, and state isolation. |
| **LangChain Core** | `ChatPromptTemplate`, LCEL (`prompt | llm`) | Formulates structured prompt chains for data denoising, Hinglish-to-English translation, JSON-enforced ABSA, and RAG strategy synthesis. |
| **Fireworks AI** | `accounts/fireworks/models/deepseek-v3p1` | High-speed LLM inference utilized for translation, aspect-based sentiment parsing, and corporate consultant analysis. |
| **Fireworks AI** | `nomic-ai/nomic-embed-text-v1.5` | Generates 768-dimensional dense vector embeddings for deduplication/spam detection and semantic clustering. |
| **ChromaDB** | `PersistentClient` (`nexus_reviews`) | Embedded vector database storing high-confidence review embeddings along with string documents and parsed ABSA JSON metadata. |
| **Scikit-learn** | `HDBSCAN` | Density-based unsupervised clustering applied over vector embeddings to discover recurring defect clusters without manual labeling. |
| **Scikit-learn** | `cosine_similarity` | Rapid similarity matrix computation over an in-memory rolling buffer to identify duplicate or bot-generated spam reviews. |
| **Pydantic** | `BaseModel` | Request schema validation for single-review, batch ingestion, and human-in-the-loop approval endpoints. |
| **Chart.js** | CDN Distribution | Frontend visualization engine powering real-time risk heatmaps, sentiment distributions, stacked feature bars, and timelines. |
| **Marked.js** | CDN Distribution | Client-side Markdown rendering and export engine for downloaded executive strategy briefs. |

---

## Backend Architecture & AI Orchestration

The backend heavily relies on `langgraph` to create a directed, stateful processing pipeline, and `langchain_core` to interact with the Fireworks AI models (`deepseek-v3p1` for text generation and `nomic-embed-text-v1.5` for embeddings).

### 1. LangGraph State Machine & Pipeline Nodes

The ingestion lifecycle is managed as a directed cyclic graph running on a mutable `ReviewState` typed dictionary.

```text
               [ Raw Customer Review ]
                          │
                          ▼
                 ┌─────────────────┐
                 │   spam_filter   │
                 └────────┬────────┘
                          │
                 [ Is Cosine > 0.92? ]
                    /           \
           Yes     /             \   No
                  ▼               ▼
             ┌─────────┐   ┌──────────────┐
             │   END   │   │ text_cleaner │
             │ (Audit) │   └──────┬───────┘
             └─────────┘          │
                                  ▼
                           ┌─────────────┐
                           │ absa_engine │
                           └──────┬──────┘
                                  │
                       [ Confidence < 0.60? ]
                          /              \
                 Yes     /                \   No
                        ▼                  ▼
              ┌──────────────────┐   ┌───────────────┐
              │  sarcasm_queue   │   │ store_db_node │
              │  (Manual Audit)  │   └───────┬───────┘
              └──────────────────┘           │
                                             ▼
                                          [ END ]
```

#### Graph State Definition (`ReviewState`)

Every review execution instantiates a state with the following contract (acting as the shared memory for a single execution):

* `raw_text` (*str*): The unprocessed customer input text.
* `is_spam` (*bool*): Boolean flag indicating whether the review is an adversarial or duplicate submission.
* `clean_text` (*str*): Denoised, emoji-stripped, and translated professional English representation.
* `status` (*str*): Human-readable tracking message describing pipeline progression.
* `absa_result` (*dict*): Aspect-Based Sentiment Analysis output containing extracted product entities, sentiments, and confidence scores.
* `translate` (*bool*): Toggle instructing whether non-English review handling is applied.

#### LangGraph Node Implementations

1. **`spam_filter` (`check_spam_node`):**
* Bypasses short reviews (<30 characters).
* Generates an embedding vector using `FireworksEmbeddings`.
* Computes cosine similarity against a rolling FIFO memory cache (`MAX_CACHE_SIZE = 500`).
* Flags the review as spam if similarity exceeds `0.92`.

2. **`text_cleaner` (`clean_text_node`):**
* Constructs a LangChain prompt template specifying e-commerce denoising.
* Calls the `deepseek-v3p1` model via LCEL (`prompt | llm`) to convert Hinglish/slang into neutral English while preserving core sentiment. This node only executes heavily if the `translate` state is active.

3. **`absa_engine` (`absa_node`):**
* Invokes the LLM to identify specific features (e.g., *Battery, Display, Thermals*), classify sentiment (*Positive, Negative, Neutral*), and assign a confidence coefficient between `0.0` and `1.0`.
* Forces the LLM to output a clean JSON structure:
```json
{
  "extractions": [
    {"feature": "Battery", "sentiment": "Negative", "confidence": 0.52}
  ]
}
```
* If confidence falls below `0.60`, the node dynamically assigns a status sending the review to the **Sarcasm / Uncertainty Queue**.

4. **`store_db_node`:**
* Executes a safety check: if the state contains `"Sarcasm"` or `"Low Confidence"`, vector insertion is aborted to avoid poisoning vector space.
* For valid reviews, generates the final vector and persists it to ChromaDB, attaching the ABSA JSON results as metadata.

#### Conditional Routing

A conditional edge evaluates the state directly after `spam_filter`:

```python
def route_spam(state: ReviewState):
    if state["is_spam"]:
        return "end"
    return "clean"
```

If `is_spam` is `True`, execution jumps directly to `END`, bypassing text cleaning and model extraction. Otherwise, it proceeds sequentially: `text_cleaner` $\rightarrow$ `absa_engine` $\rightarrow$ `store_db_node` $\rightarrow$ `END`.

---

### 2. ChromaDB: Persistent Vector Storage & Reset Protocol

The vector database handles semantic search and serves as the data retrieval source for anomaly detection:

* **Initialization & State Parity:**
```python
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_client.delete_collection("nexus_reviews")
collection = chroma_client.get_or_create_collection("nexus_reviews")
```

On server restart, the persistent collection is intentionally cleared and re-instantiated. This ensures ChromaDB vectors remain 1:1 synchronized with backend in-memory tracking arrays (`global_history`, `review_registry`).
* **Vector Ingestion Contract:** Reviews entering `store_db_node` are indexed with:
* `ids`: Unique UUID v4 string.
* `embeddings`: 768-dimensional dense vector from `nomic-embed-text-v1.5`.
* `documents`: Normalized `clean_text`.
* `metadatas`: Serialized JSON payload containing the complete ABSA extraction object (`{"extraction": json.dumps(...)}`).

---

### 3. HDBSCAN: Unsupervised Defect Clustering & Risk Scoring

Rather than relying on pre-labeled categories, the `/detect-anomalies` endpoint executes **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)** across the dense vector space.

#### Clustering Pipeline

1. **Vector Retrieval:** Pulls all stored embeddings, document texts, and metadata from ChromaDB:
```python
data = collection.get(include=["embeddings", "documents", "metadatas"])
```

A minimum threshold of at least 3 reviews is required before clustering can execute.
2. **Spatial Partitioning:**
```python
clusterer = HDBSCAN(min_cluster_size=3, copy=True)
labels = clusterer.fit_predict(embeddings)
```

3. **Noise Isolation:** Data points marked with label `-1` are discarded as background noise, isolating dense complaint patterns.
4. **Feature Extraction & Risk Formula:**
For each valid cluster, metadata is decoded to aggregate sentiment distribution. The primary defect topic is extracted using `Counter(features).most_common(1)`.
The backend computes a **Predictive Risk Score** to rank anomalies based on defect severity and complaint volume:
$$\text{Urgency} = \left(\frac{\text{Negative Sentiment Count}}{\text{Total Cluster Count}}\right) + 1$$

$$\text{Predictive Risk Score} = \text{Total Cluster Count} \times \text{Urgency}$$

5. **Cluster Caching:** Valid clusters are cached in `cluster_cache[cluster_id]` to support instant retrieval during RAG strategy synthesis.

---

### 4. Industry-Aware Gated Retrieval RAG

When a high-risk defect cluster is identified, administrators can trigger the `/generate-strategy/{cluster_id}` endpoint:

```text
 ┌─────────────────────────┐     ┌────────────────────────┐
 │   Target Cluster Data   │     │   industry_specs.txt   │
 │ (Aggregated Complaints) │     │ (Engineering Limits)   │
 └────────────┬────────────┘     └───────────┬────────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │      LangChain Prompt        │
              │ (Consultant Decision Matrix) │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  DeepSeek-V3.1 (Fireworks)   │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │    Structured Strategy       │
              │  - Zero-Day Classification   │
              │  - Engineering Root-Cause    │
              │  - Customer Support Auto-Rep │
              │  - Roadmap Priority (P0-P2)  │
              └──────────────────────────────┘
```

The LLM is provided with two conflicting contexts to prevent hallucinations:

1. **Source of Truth:** Real customer complaints extracted from the cluster.
2. **Engineering Limits:** Official product tolerances from `industry_specs.txt`.

The chain classifies the issue as:

* **Documented Technical Defect:** The problem describes a metric directly violating specified hardware tolerances (e.g., surface temperature exceeds 40°C).
* **Zero-Day Defect:** The problem reflects a genuine technical defect not documented anywhere in the official specifications.
* **Market Sentiment:** The feedback reflects subjective dissatisfaction rather than an engineering violation.

---

## Complete Data Flow Lifecycle

1. **Ingestion:** A user or batch process submits a review via the FastAPI endpoints (`/stream-review` or `/batch-ingest`).
2. **LangGraph Execution:** The review enters the `ReviewState` and triggers the `spam_filter`.
3. **Spam Gate:** Duplicates are blocked via vector similarity. Clean reviews proceed.
4. **Transformation:** The text is denoised and features/sentiments are extracted via LangChain LLM invocations.
5. **Storage:** Processed reviews are embedded and stored in ChromaDB, while the FastAPI in-memory registries cache the data for live frontend syncing.
6. **Clustering (HDBSCAN):** The `/detect-anomalies` endpoint groups the vector space into risk clusters.
7. **Resolution & HITL:** An admin uses Gated RAG to synthesize a response strategy for a cluster. The admin approves the AI-drafted reply, which is pushed directly to the public mock storefront.

---

## Human-in-the-Loop (HITL) Storefront & Live Telemetry

* **Storefront Integration (`storefront.html` & `storefront.js`):** Simulates an e-commerce customer portal. Customers submit reviews that route directly through the LangGraph ingestion pipeline.
* **HITL Verification Gate:** AI-generated marketing responses generated by the RAG system are saved as drafts. They are suppressed from public storefront endpoints until an administrator approves them via `/approve-reply/{review_id}` or `/approve-cluster-replies/{cluster_id}`.
* **Live Dashboard Telemetry (`index.html` & `app.js`):**
  * **Bubble Heatmap:** Plots review volume ($x$) against predictive risk score ($y$), where bubble size indicates cluster magnitude.
  * **Sentiment Doughnut:** Renders global sentiment distribution (Positive, Negative, Neutral).
  * **Stacked Feature Bar:** Deconstructs sentiment performance across extracted hardware features.
  * **Timeline:** Tracks accumulating positive and negative signals across successive reviews.

---

## Project Structure

```text
nexus-intelligence/
│
├── main.py                  # FastAPI server, LangGraph DAG, ChromaDB & HDBSCAN logic, LangChain logic
├── .env                     # Stores FIREWORKS_API_KEY
├── industry_specs.txt       # Hardware specifications & local knowledge base for Gated RAG
│
└── static/                  # Mounted automatically at application root ("/")
    ├── index.html           # Executive analytics dashboard
    ├── index1.html          # Alternative dashboard view
    ├── app.js               # Dashboard controller, Chart.js managers, state logic & polling loops
    ├── storefront.html      # Mock Amazon-style customer product page with verified AI replies
    ├── storefront.js        # Storefront API polling and review submission (3s polling)
    └── style.css            # Light-mode enterprise design system
```

---

## API Reference

### Review Ingestion

* `POST /stream-review`: Processes a single review through LangGraph, running spam filtering, translation, ABSA extraction, and ChromaDB insertion.
  * *Payload:* `{"text": "string", "translate": true}`
* `POST /batch-ingest`: Sequentially processes an array of reviews.
  * *Payload:* `{"reviews": ["string", "string"], "translate": true}`

### Analytics & Clustering

* `GET /detect-anomalies`: Queries ChromaDB embeddings, executes HDBSCAN clustering, and returns risk scores and isolated complaint clusters.
* `GET /metrics`: Aggregates the Global Sentiment Index (GSI), sentiment timelines, feature performance maps, and queue metrics.
* `GET /sarcasm-queue`: Lists ambiguous reviews quarantined due to low ABSA confidence (< 0.60).
* `GET /flagged`: Returns the audit trail of reviews flagged by the cosine similarity spam filter.
* `GET /review-history`: Returns the sequential history of all processed clean reviews.

### Strategy & HITL Workflows

* `POST /generate-strategy/{cluster_id}`: Runs Gated RAG cross-referencing cluster reviews against `industry_specs.txt` to output structured root-cause analyses and support replies.
* `GET /storefront-reviews`: Returns all non-spam reviews, displaying verified AI support responses only if approved.
* `POST /approve-reply/{review_id}`: Approves an AI response for a single review.
* `POST /approve-cluster-replies/{cluster_id}`: Bulk-approves and publishes an AI response across all reviews associated with an anomaly cluster.

---

## Setup & Installation

1. **Install Python Dependencies:**
```bash
pip install fastapi uvicorn pydantic python-dotenv chromadb scikit-learn langchain-fireworks langchain-core langgraph numpy
```

2. **Configure API Keys:**
Create a `.env` file in the root directory:
```env
FIREWORKS_API_KEY=your_fireworks_api_key_here
```

3. **Establish RAG Knowledge Base / Technical Specifications:**
Create `industry_specs.txt` in the root directory to define the RAG decision space:
```text
Nexus Pro Smartphone Technical Specs:
- Operating Temperature: 0°C to 40°C. Surface temp exceeding 45°C indicates thermal runaway.
- Display: 120Hz AMOLED. Screen flickering at low brightness is resolved by disabling DC Dimming.
- Battery: 5000mAh. Degradation exceeding 15% in first 90 days indicates defective batch cell.
```

4. **Run the Server:**
```bash
uvicorn main:app --reload
```

## Usage Guide

1. Navigate to `http://localhost:8000/` to access the primary analytics dashboard. The frontend polls the backend every 10 seconds to update Chart.js visualizations.
2. Open a separate tab for `http://localhost:8000/storefront.html` to view the live customer storefront, which polls for approved AI replies every 3 seconds.
3. Submit a review containing specific product features (e.g., "The battery drains too fast but the screen is nice") to trigger the LangGraph pipeline.
4. Submit identical reviews rapidly to test the vector cosine similarity spam filter.
5. When the dashboard flags an Anomaly Cluster (requires at least 3 similar negative reviews), click "Resolve Trend" to execute the LangChain Gated RAG prompt. Approve the drafted reply to push it to the storefront view.

---

*Built for Hack Malenadu.*
