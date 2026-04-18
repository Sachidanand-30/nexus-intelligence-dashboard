import os
import json
import uuid
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
# ML & AI Imports
from sklearn.metrics.pairwise import cosine_similarity
from langchain_fireworks import ChatFireworks, FireworksEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

# Load Environment Variables
load_dotenv()
if not os.getenv("FIREWORKS_API_KEY"):
    raise ValueError("FIREWORKS_API_KEY is missing in .env file")

# Initialize FastAPI
app = FastAPI(title="Nexus Intelligence Cockpit")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- IN-MEMORY STATE ---
global_history = []       # Successfully analyzed reviews
sarcasm_queue = []         # Low-confidence / sarcasm reviews
flagged_reviews = []       # Spam audit trail (never deleted)
sentiment_timeline = []    # Timestamped sentiment snapshots
cluster_cache = {}         # Stateful mapping: cluster_id -> {reviews, feature, metadatas}
review_registry = {}       # UUID -> full review data (for storefront + HITL)

# --- INDUSTRY KNOWLEDGE BASE ---
_specs_path = os.path.join(os.path.dirname(__file__), "industry_specs.txt")
if os.path.exists(_specs_path):
    with open(_specs_path, "r", encoding="utf-8") as f:
        INDUSTRY_SPECS = f.read()
else:
    INDUSTRY_SPECS = "No industry specifications loaded."

# --- INITIALIZE MODELS (100% CLOUD/API - ZERO LOCAL DOWNLOADS) ---
print("[*] Connecting to Fireworks AI...")
# 1. Fireworks API for Vector Embeddings
embedding_model = FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")

# 2. Fireworks API for the LLM
llm = ChatFireworks(
    model="accounts/fireworks/models/deepseek-v3p1", 
    temperature=0.1
)
print("[OK] Nexus Intelligence Engine Online.")

# --- 1. THE VECTOR CACHE ---
recent_vectors = []
MAX_CACHE_SIZE = 500

# --- CHROMADB SETUP (fresh on each restart to stay in sync with in-memory state) ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Clear stale data so clusters match the in-memory history
chroma_client.delete_collection("nexus_reviews")
collection = chroma_client.get_or_create_collection("nexus_reviews")
print("[OK] ChromaDB reset — clean slate.")

# --- 2. LANGGRAPH STATE DEFINITION ---
class ReviewState(TypedDict):
    raw_text: str
    is_spam: bool
    clean_text: str
    status: str
    absa_result: dict  # Layer 2 data
    translate: bool    # Whether to run translation

# --- 3. LANGGRAPH NODES ---

def check_spam_node(state: ReviewState):
    """Gate 2: Vector Similarity Check using Fireworks API"""
    text = state["raw_text"]
    
    if len(text) < 30:
        return {"is_spam": False, "status": "Passed (Short)"}
    
    # Generate vector instantly via API
    vector = embedding_model.embed_query(text)
    
    is_spam = False
    if len(recent_vectors) > 0:
        similarities = cosine_similarity([vector], recent_vectors)[0]
        max_sim = np.max(similarities)
        
        # Threshold matches prompt spec: > 0.92 = spam
        if max_sim > 0.92:
            is_spam = True
            return {"is_spam": True, "status": f"Flagged Spam (Cosine: {max_sim:.2f})"}
    
    recent_vectors.append(vector)
    if len(recent_vectors) > MAX_CACHE_SIZE:
        recent_vectors.pop(0)
        
    return {"is_spam": False, "status": "Clean Human Data"}

def clean_text_node(state: ReviewState):
    """Layer 1: The Denoising & Translation Orchestrator"""
    text = state["raw_text"]
    
    # If translate is disabled and text is likely English, skip heavy LLM call
    if not state.get("translate", True):
        return {"clean_text": text}
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an e-commerce data cleaner. Your job is to take noisy, unstructured reviews (often containing Hinglish, emojis, or bad grammar) and output a clean, professional English translation. Strip excessive emojis. DO NOT summarize or alter the core sentiment. ONLY output the clean text, nothing else."),
        ("user", "{review}")
    ])
    
    chain = prompt | llm
    clean_result = chain.invoke({"review": text})
    
    return {"clean_text": clean_result.content}

def absa_node(state: ReviewState):
    """Layer 2: Aspect-Based Sentiment Analysis & Sarcasm Check"""
    clean_text = state["clean_text"]
    
    absa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an Aspect-Based Sentiment Analysis engine. 
        Read the clean review and extract ALL specific product features mentioned (e.g., Battery, Screen, Delivery).
        For each feature, determine the sentiment (Positive, Negative, Neutral).
        Assign a confidence score between 0.0 and 1.0. If the text is sarcastic, lower the confidence score below 0.6.
        Respond ONLY with a valid JSON in this exact format, with an array of extractions to support multiple features:
        {{"extractions": [
            {{"feature": "feature_name", "sentiment": "Positive", "confidence": 0.95}}
        ]}}"""),
        ("user", "{review}")
    ])
    
    chain = absa_prompt | llm
    result = chain.invoke({"review": clean_text})
    
    # Parse the JSON response securely
    try:
        raw_json = result.content.replace("```json", "").replace("```", "").strip()
        absa_data = json.loads(raw_json)
        
        has_low_confidence = False
        if "extractions" in absa_data:
            for item in absa_data["extractions"]:
                if item.get("confidence", 1.0) < 0.60:
                    has_low_confidence = True
        elif absa_data.get("confidence", 1.0) < 0.60:
            has_low_confidence = True
            
        if has_low_confidence:
            state["status"] = "Sent to Sarcasm Queue (Low Confidence)"
            
    except json.JSONDecodeError:
        absa_data = {"error": "Failed to parse JSON", "raw": result.content}
        
    return {"absa_result": absa_data, "status": state.get("status", "Analyzed Successfully")}

def store_db_node(state: ReviewState):
    """Layer 2.5: Store to Vector DB (Skip if Sarcasm)"""
    if "Sarcasm" in state.get("status", "") or "Low Confidence" in state.get("status", ""):
        return {"status": state.get("status")}
    
    text = state["clean_text"]
    try:
        vector = embedding_model.embed_query(text)
        doc_id = str(uuid.uuid4())
        
        collection.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[{"extraction": json.dumps(state.get("absa_result", {}))}]
        )
        state["status"] = "Stored in Vector DB"
    except Exception as e:
        state["status"] = f"DB Error: {str(e)}"
        
    return {"status": state.get("status")}

# --- 4. LANGGRAPH ROUTING LOGIC ---
def route_spam(state: ReviewState):
    if state["is_spam"]:
        return "end"
    return "clean"

# --- 5. BUILD THE GRAPH ---
workflow = StateGraph(ReviewState)

workflow.add_node("spam_filter", check_spam_node)
workflow.add_node("text_cleaner", clean_text_node)
workflow.add_node("absa_engine", absa_node)
workflow.add_node("store_db_node", store_db_node)

workflow.set_entry_point("spam_filter")

# Routing Logic: If spam -> End. If clean -> send to cleaner.
workflow.add_conditional_edges(
    "spam_filter",
    route_spam,
    {"end": END, "clean": "text_cleaner"}
)

# Linear flow: Cleaner output goes straight to ABSA extraction
workflow.add_edge("text_cleaner", "absa_engine")
workflow.add_edge("absa_engine", "store_db_node")
workflow.add_edge("store_db_node", END)

nexus_app = workflow.compile()

# ============================================================
#  HELPER: Process a single review through the full pipeline
# ============================================================
def _run_pipeline(text: str, translate: bool = True) -> dict:
    """Runs one review through the LangGraph pipeline and updates in-memory state."""
    initial_state = {
        "raw_text": text,
        "is_spam": False,
        "clean_text": "",
        "status": "",
        "absa_result": {},
        "translate": translate,
    }
    result = nexus_app.invoke(initial_state)
    
    ts = time.time()
    review_id = str(uuid.uuid4())
    
    # Route to the correct in-memory store
    if result.get("is_spam"):
        flagged_reviews.append({
            "raw_text": result["raw_text"],
            "status": result["status"],
            "timestamp": ts,
        })
    elif "Sarcasm" in result.get("status", "") or "Low Confidence" in result.get("status", ""):
        sarcasm_queue.append(result)
    else:
        global_history.append(result)
        # Track sentiment over time
        pos = neg = neu = 0
        ext = result.get("absa_result", {})
        if "extractions" in ext:
            for f in ext["extractions"]:
                s = str(f.get("sentiment", "")).lower()
                if s == "positive": pos += 1
                elif s == "negative": neg += 1
                else: neu += 1
        sentiment_timeline.append({"ts": ts, "pos": pos, "neg": neg, "neu": neu})
    
    # Register in storefront registry (every review, including spam/sarcasm)
    review_registry[review_id] = {
        "review_id": review_id,
        "original_text": result["raw_text"],
        "clean_text": result.get("clean_text", ""),
        "extraction": result.get("absa_result", {}),
        "status": result.get("status", ""),
        "is_spam": result.get("is_spam", False),
        "reply_draft": None,
        "is_approved": False,
        "timestamp": ts,
    }
    
    result["review_id"] = review_id
    return result


# ============================================================
#  6. FASTAPI ENDPOINTS
# ============================================================

class ReviewRequest(BaseModel):
    text: str
    translate: Optional[bool] = True

class BatchReviewRequest(BaseModel):
    reviews: List[str]
    translate: Optional[bool] = True

# --- Single Review Ingestion ---
@app.post("/stream-review")
def process_single_review(request: ReviewRequest):
    result = _run_pipeline(request.text, request.translate)
    return {
        "original_text": result["raw_text"],
        "is_spam": result["is_spam"],
        "pipeline_status": result["status"],
        "cleaned_text": result.get("clean_text", "N/A"),
        "extraction": result.get("absa_result", {}),
    }

# --- Batch Review Ingestion ---
@app.post("/batch-ingest")
def batch_ingest(request: BatchReviewRequest):
    total = len(request.reviews)
    results = []
    for i, text in enumerate(request.reviews):
        print(f"[BATCH] Processing {i+1}/{total}: {text[:50]}...")
        try:
            r = _run_pipeline(text.strip(), request.translate)
            results.append({
                "original": r["raw_text"],
                "status": r["status"],
                "is_spam": r["is_spam"],
            })
        except Exception as e:
            print(f"[BATCH] Error on review {i+1}: {e}")
            results.append({
                "original": text[:80],
                "status": f"Error: {str(e)}",
                "is_spam": False,
            })
    print(f"[BATCH] Complete: {len(results)}/{total} processed")
    return {"processed": len(results), "results": results}

# --- Layer 4: Industry-Aware Gated Retrieval RAG ---
@app.post("/generate-strategy/{cluster_id}")
def generate_strategy(cluster_id: int):
    """Layer 4: Industry-Aware Gated Retrieval RAG"""
    # 1. Context Retrieval from cached cluster mapping
    if cluster_id not in cluster_cache:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found. Run /detect-anomalies first.")
    
    cluster = cluster_cache[cluster_id]
    reviews = cluster["reviews"]
    feature = cluster["feature"]
    
    if len(reviews) < 2:
        raise HTTPException(status_code=400, detail="Not enough context in this cluster for reliable synthesis.")
    
    # 2. Aggregate into a single context string (Source of Truth)
    context_block = "\n".join([f"- Review {i+1}: {r}" for i, r in enumerate(reviews)])
    
    # 3. Industry-Aware Strategy Synthesis
    strategy_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Corporate Consultant performing Industry-Aware Gated Retrieval Analysis.
You are given TWO context sources:
1. CUSTOMER REVIEWS — real complaints about a specific product defect cluster.
2. INDUSTRY SPECIFICATIONS — the official technical limits, standards, and troubleshooting playbooks for this product.

Your task:
- Cross-reference the customer complaints against the industry specs.
- If the complaint describes a failure that VIOLATES a documented spec (e.g., device hit 50°C but spec says max 40°C), classify it as "Documented Technical Defect".
- If the complaint describes an issue NOT mentioned anywhere in the specs (e.g., screen flickers when opening a specific app), classify it as "Zero-Day Defect".
- If the complaint is purely about subjective experience with no technical violation, classify it as "Market Sentiment".

Do NOT hallucinate. Base your analysis strictly on the provided contexts.

Respond ONLY with a valid JSON object in this exact format:
{{
  "defect_type": "Zero-Day Defect" or "Documented Technical Defect" or "Market Sentiment",
  "industry_benchmark_gap": "A sentence explaining how far the product deviates from the spec, or stating no spec exists for this issue.",
  "technical_report": "A detailed defect brief for the Engineering team identifying the root cause.",
  "marketing_response": "A brand-safe, empathetic reply for customer support. MUST include a specific troubleshooting step or next action pulled directly from the [TROUBLESHOOTING & RESOLUTION PLAYBOOKS] section based on the defect type.",
  "roadmap_recommendation": {{
    "priority": "P0 or P1 or P2",
    "next_step": "A suggested next step for the product roadmap."
  }}
}}

Do NOT wrap in markdown code blocks. Output raw JSON only."""),
        ("user", "Defect Cluster Feature: {feature}\nNumber of affected reports: {count}\n\n=== CUSTOMER REVIEWS (Source of Truth) ===\n{context}\n\n=== INDUSTRY SPECIFICATIONS ===\n{specs}")
    ])
    
    chain = strategy_prompt | llm
    result = chain.invoke({
        "feature": feature,
        "count": str(len(reviews)),
        "context": context_block,
        "specs": INDUSTRY_SPECS
    })
    
    # 4. Parse structured JSON securely
    try:
        raw = result.content.replace("```json", "").replace("```", "").strip()
        strategy_data = json.loads(raw)
    except json.JSONDecodeError:
        strategy_data = {
            "defect_type": "Unknown",
            "industry_benchmark_gap": "Unable to parse LLM response.",
            "technical_report": result.content,
            "marketing_response": "We are aware of this issue and our team is actively investigating a resolution.",
            "roadmap_recommendation": {"priority": "P1", "next_step": "Investigate further."}
        }
    
    # Find review_ids matching this cluster's reviews
    cluster_review_ids = []
    for rid, rdata in review_registry.items():
        if rdata["original_text"] in reviews or rdata.get("clean_text", "") in reviews:
            cluster_review_ids.append(rid)
    
    return {
        "cluster_id": cluster_id,
        "feature": feature,
        "reviews_analyzed": len(reviews),
        "strategy": strategy_data,
        "review_ids": cluster_review_ids
    }

# --- Dashboard Metrics ---
@app.get("/metrics")
async def get_metrics():
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    feature_map = {}  # feature -> {positive, negative, neutral}
    
    for item in global_history:
        ext = item.get("absa_result", {})
        if "extractions" in ext:
            for f in ext["extractions"]:
                feat = f.get("feature", "Unknown")
                s = str(f.get("sentiment", "")).lower()
                
                if feat not in feature_map:
                    feature_map[feat] = {"positive": 0, "negative": 0, "neutral": 0}
                
                if s == "positive":
                    positive_count += 1
                    feature_map[feat]["positive"] += 1
                elif s == "negative":
                    negative_count += 1
                    feature_map[feat]["negative"] += 1
                else:
                    neutral_count += 1
                    feature_map[feat]["neutral"] += 1
                
    total_analyzed = len(global_history) + len(sarcasm_queue) + len(flagged_reviews)
    total_sentiments = positive_count + negative_count
    gsi = int((positive_count / total_sentiments) * 100) if total_sentiments > 0 else 0
        
    return {
        "total_processed": total_analyzed,
        "clean_count": len(global_history),
        "spam_count": len(flagged_reviews),
        "sarcasm_count": len(sarcasm_queue),
        "gsi": gsi,
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count,
        "feature_map": feature_map,
        "sentiment_timeline": sentiment_timeline[-20:],  # Last 20 data points
        "history": global_history,
    }

# --- Sarcasm Queue ---
@app.get("/sarcasm-queue")
async def get_sarcasm_queue():
    return {"queue": sarcasm_queue}

# --- Spam Audit Trail ---
@app.get("/flagged")
async def get_flagged():
    return {"flagged": flagged_reviews}

# --- Storefront Reviews (for mock Amazon page) ---
class ApproveReplyRequest(BaseModel):
    reply_text: str

@app.get("/storefront-reviews")
async def get_storefront_reviews():
    """Returns all reviews for the storefront. Only shows reply_draft if approved."""
    reviews = []
    for rid, rdata in review_registry.items():
        if rdata.get("is_spam"):
            continue  # Don't show spam on storefront
        entry = {
            "review_id": rid,
            "text": rdata["original_text"],
            "clean_text": rdata.get("clean_text", ""),
            "extraction": rdata.get("extraction", {}),
            "timestamp": rdata.get("timestamp", 0),
            "reply": None,
        }
        if rdata.get("is_approved") and rdata.get("reply_draft"):
            entry["reply"] = rdata["reply_draft"]
        reviews.append(entry)
    # Sort by newest first
    reviews.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"reviews": reviews}

@app.post("/approve-reply/{review_id}")
def approve_reply(review_id: str, request: ApproveReplyRequest):
    """Human-in-the-Loop: Approve an AI-drafted reply for a specific review."""
    if review_id not in review_registry:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found.")
    review_registry[review_id]["reply_draft"] = request.reply_text
    review_registry[review_id]["is_approved"] = True
    return {"status": "approved", "review_id": review_id}

@app.post("/approve-cluster-replies/{cluster_id}")
def approve_cluster_replies(cluster_id: int, request: ApproveReplyRequest):
    """Approve reply for all reviews in a cluster at once."""
    if cluster_id not in cluster_cache:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found.")
    
    cluster = cluster_cache[cluster_id]
    approved_count = 0
    for rid, rdata in review_registry.items():
        if rdata["original_text"] in cluster["reviews"] or rdata.get("clean_text", "") in cluster["reviews"]:
            rdata["reply_draft"] = request.reply_text
            rdata["is_approved"] = True
            approved_count += 1
    
    return {"status": "approved", "cluster_id": cluster_id, "reviews_updated": approved_count}

# --- Review History (timeline-friendly) ---
@app.get("/review-history")
async def get_review_history():
    history = []
    for i, item in enumerate(global_history):
        history.append({
            "id": i + 1,
            "raw": item.get("raw_text", ""),
            "clean": item.get("clean_text", ""),
            "status": item.get("status", ""),
            "absa": item.get("absa_result", {}),
        })
    return {"reviews": history}

# --- Anomaly Detection (HDBSCAN) ---
@app.get("/detect-anomalies")
def detect_anomalies():
    try:
        # 1. Fetch vectors from Chroma
        data = collection.get(include=["embeddings", "documents", "metadatas"])
        
        if data.get("embeddings") is None or len(data["embeddings"]) < 3:
            return {"status": "Not enough data for clustering (minimum 3 required).", "anomalies_detected": 0, "clusters": []}
        
        # 2. Run HDBSCAN
        from sklearn.cluster import HDBSCAN
        
        embeddings = np.array(data["embeddings"])
        clusterer = HDBSCAN(min_cluster_size=3, copy=True)
        labels = clusterer.fit_predict(embeddings)
        
        # 3. Anomaly Rules
        clusters_info = {}
        
        for i, label in enumerate(labels):
            if label == -1:
                continue
                
            if label not in clusters_info:
                clusters_info[label] = {
                    "reviews": [],
                    "metadatas": [],
                    "count": 0
                }
                
            clusters_info[label]["reviews"].append(data["documents"][i])
            clusters_info[label]["metadatas"].append(data["metadatas"][i])
            clusters_info[label]["count"] += 1
            
        responses = []
        from collections import Counter
        
        for lbl, info in clusters_info.items():
            features = []
            negatives = 0
            for meta in info["metadatas"]:
                try:
                    extraction = json.loads(meta.get("extraction", "{}"))
                    if "extractions" in extraction:
                        for ext in extraction["extractions"]:
                            features.append(ext.get("feature", "Unknown"))
                            if ext.get("sentiment", "").lower() == "negative":
                                negatives += 1
                    elif "feature" in extraction:
                        features.append(extraction["feature"])
                        if extraction.get("sentiment", "").lower() == "negative":
                            negatives += 1
                except:
                    pass
                    
            if features:
                main_feature = Counter(features).most_common(1)[0][0]
            else:
                main_feature = "Unknown"
                
            urgency = (negatives / info["count"]) + 1 if info["count"] > 0 else 1
            risk_score = info["count"] * urgency
            
            # Cache the full cluster data for Layer 4 Gated Retrieval
            cluster_cache[int(lbl)] = {
                "reviews": info["reviews"],
                "feature": main_feature,
                "metadatas": info["metadatas"]
            }
            
            responses.append({
                "cluster_id": int(lbl),
                "main_feature": main_feature,
                "review_count": info["count"],
                "negative_count": negatives,
                "predictive_risk_score": round(risk_score, 2),
                "sample_reviews": info["reviews"][:3]
            })
            
        return {"anomalies_detected": len(responses), "clusters": responses}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "anomalies_detected": 0, "clusters": []}

# Mount Static Front-End
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/", StaticFiles(directory="static", html=True), name="static")