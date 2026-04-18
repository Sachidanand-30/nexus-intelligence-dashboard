# Nexus Intelligence — Review Intelligence Platform

Nexus Intelligence is an enterprise-grade, fully autonomous product review intelligence platform. It transforms chaotic, noisy customer feedback (including Hinglish and heavy emojis) into structured, actionable business strategies in under 5 seconds.

Instead of passive dashboards, Nexus uses a 4-layer AI architecture to detect unseen "Zero-Day Defects" and automatically synthesize engineering technical briefs and customer support responses using **Agentic Gated Retrieval (RAG)**.

Perfect for the modern enterprise, it features a complete **Human-in-the-Loop (HITL)** approval flow linking a mock Amazon-storefront directly to an executive command center.

---

## 🏗️ The 4-Layer Architecture

### Layer 1: Data Gateway & Pre-Processing (The "Clean Room")
Real-world data is messy. Layer 1 ensures only high-signal human insights enter the system.
- **Vector-Based Spam Filter**: Defends against LLM-bot campaigns by embedding incoming text and calculating Cosine Similarity against recent history (Threshold > 0.92 = Flagged as Spam).
- **Denoising Orchestrator**: Uses a **LangGraph** pipeline and **Fireworks AI** to translate local dialects (like Hinglish) into clean English, stripping emojis and non-semantic noise.

### Layer 2: The Intelligence Core (The "Brain")
Moving beyond basic positive/negative scoring to extract granular component-level insights.
- **Track A (The Knowns): Joint ABSA**: Aspect-Based Sentiment Analysis extracts specific features and sentiment (e.g., "Battery: Negative, Screen: Positive"). If ambiguity or sarcasm is detected, it flags the review into an uncertainty queue.
- **Track B (The Unknowns): HDBSCAN Clustering**: To detect entirely new "Zero-Day Anomalies", semantic vectors are stored in **ChromaDB** and clustered using HDBSCAN. Unlike K-Means, HDBSCAN doesn't require knowing cluster counts in advance, finding organic shapes in defect data.

### Layer 3: The Digital Twin Cockpit (The "Face")
Translating complex machine learning math into a clean, professional B2B SaaS dashboard.
- **Executive Dashboard**: A sleek, light-mode interface (inspired by Vercel/Linear) showing live Key Performance Indicators (KPIs).
- **Anomaly Heatmap**: Visually alerts product managers when a cluster of zero-day defects unexpectedly spikes.

### Layer 4: Action Synthesis (The "Hands")
Turning passive alerts into active business strategy.
- **Industry-Aware RAG Engine**: When an anomaly is detected, the system pulls the exact problem cluster from ChromaDB and cross-references it against internal **Industry Specifications** (operating temps, IP68 limits, battery cycles).
- **Dual-Document Generation**: Synthesizes a P0 Technical Brief for Engineering AND a customized, empathetic reply for Customer Support.
- **Human-in-the-Loop Gate**: A manager clicks "Approve & Post", instantly pushing the AI-drafted reply to the live storefront.

---

## 💻 Tech Stack

- **Backend**: FastAPI, Python 3
- **AI Orchestration**: LangGraph, LangChain
- **LLM & Embeddings**: Fireworks AI (`accounts/fireworks/models/deepseek-v3p1`, `nomic-ai/nomic-embed-text-v1.5`)
- **Vector Database**: ChromaDB (Persistent)
- **Machine Learning**: Scikit-Learn (HDBSCAN, Cosine Similarity), NumPy
- **Frontend**: HTML5, Vanilla CSS, JavaScript, Chart.js

---

## 🚀 How to Run Locally

### Prerequisites
1. Python 3.9+
2. A Fireworks AI API Key

### Installation

1. Clone the repository and navigate into the folder:
   ```bash
   cd hackmalenadu
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your API key:
   ```env
   FIREWORKS_API_KEY="your_api_key_here"
   ```

### Running the Platform

Boot the FastAPI server with hot-reload enabled:
```bash
uvicorn main:app --reload --port 8000
```

The system will start and reset the local Vector database to ensure a clean slate.

### The Live Demo Loop
1. **View A (The Customer)**: Open `http://127.0.0.1:8000/storefront.html`. Submit a highly specific review (e.g., "The phone gets scorching hot at 55°C when I run the camera").
2. **View B (The Executive)**: Open `http://127.0.0.1:8000/` (Dashboard). Watch the LangGraph pipeline process the review and the HDBSCAN algorithm flag it on the Heatmap.
3. **The AI Generation**: Click "Resolve Trend" on the dashboard. The RAG engine will cross-reference the `industry_specs.txt` and draft a response.
4. **The HITL Loop**: Click "✓ Approve & Post". Return to the storefront to see the AI reply magically appear live beneath the customer's review!

---
*Built for Hack Malenadu.*
