// ============================================================
//  NEXUS INTELLIGENCE — Dashboard Controller v2.0
// ============================================================

const API = "http://127.0.0.1:8000";

// --- Chart Instances ---
let heatmapChart = null;
let sentimentChart = null;
let featureChart = null;
let timelineChart = null;

// --- State ---
let currentBrief = "";
let pendingReplyText = "";  // Stores the marketing response for HITL approval

// --- DOM Selectors ---
const $ = id => document.getElementById(id);

const reviewInput     = $("review-input");
const ingestBtn       = $("ingest-btn");
const batchBtn        = $("batch-btn");
const batchFile       = $("batch-file");
const ingestStatus    = $("ingest-status");
const translateToggle = $("translate-toggle");

const kpiTotal     = $("kpi-total");
const kpiClean     = $("kpi-clean");
const kpiSpam      = $("kpi-spam");
const kpiAnomalies = $("kpi-anomalies");
const kpiGsi       = $("kpi-gsi");

const actionCenter = $("action-center");
const sarcasmQueue = $("sarcasm-queue");
const reviewFeed   = $("review-feed");

const modal         = $("strategy-modal");
const modalContent  = $("modal-content");
const modalSubtitle = $("modal-subtitle");
const closeModal    = $("close-modal");
const downloadBtn   = $("download-btn");
const toastContainer = $("toast-container");

// --- Color Palette (Light Mode) ---
const C = {
    emerald:     "#16a34a",
    emeraldDim:  "rgba(22, 163, 74, 0.12)",
    rose:        "#ef4444",
    roseDim:     "rgba(239, 68, 68, 0.12)",
    amber:       "#ea580c",
    amberDim:    "rgba(234, 88, 12, 0.1)",
    purple:      "#7c3aed",
    purpleDim:   "rgba(124, 58, 237, 0.1)",
    blue:        "#2563eb",
    blueDim:     "rgba(37, 99, 235, 0.1)",
    gray:        "rgba(226, 232, 240, 0.8)",
    gridLine:    "rgba(226, 232, 240, 0.6)",
    text:        "#64748b",
};

// ============================================================
//  TOAST NOTIFICATIONS
// ============================================================
function toast(message, type = "success") {
    const icons = { success: "✅", error: "❌", warn: "⚠️", info: "ℹ️" };
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.innerHTML = `<span>${icons[type] || "ℹ️"}</span><span>${message}</span>`;
    toastContainer.appendChild(el);
    setTimeout(() => {
        el.style.animation = "toast-out 0.3s ease forwards";
        setTimeout(() => el.remove(), 300);
    }, 3500);
}

// ============================================================
//  ANIMATED KPI COUNTER
// ============================================================
function animateValue(el, target) {
    const current = parseInt(el.textContent) || 0;
    if (current === target) return;
    
    const duration = 500;
    const start = performance.now();
    
    function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const value = Math.round(current + (target - current) * ease);
        el.textContent = value;
        if (progress < 1) requestAnimationFrame(tick);
        else el.classList.add("count-animated");
    }
    requestAnimationFrame(tick);
}

// ============================================================
//  CHART INITIALIZATION
// ============================================================
function initCharts() {
    Chart.defaults.color = C.text;
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 11;

    // Global tooltip styling for light mode
    Chart.defaults.plugins.tooltip.titleColor = "#0f172a";
    Chart.defaults.plugins.tooltip.bodyColor = "#334155";
    Chart.defaults.plugins.tooltip.backgroundColor = "#fff";
    Chart.defaults.plugins.tooltip.borderColor = "#e2e8f0";
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.boxPadding = 4;

    // 1. Anomaly Heatmap (Bubble)
    heatmapChart = new Chart($("heatmapChart").getContext("2d"), {
        type: "bubble",
        data: { datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: "Review Volume", color: C.text },
                    grid: { color: C.gridLine },
                    border: { color: C.gridLine },
                },
                y: {
                    title: { display: true, text: "Risk Score", color: C.text },
                    grid: { color: C.gridLine },
                    border: { color: C.gridLine },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#fff",
                    borderColor: C.rose,
                    borderWidth: 1,
                    titleFont: { weight: "700" },
                    callbacks: {
                        label: ctx => `${ctx.raw.label} — Risk: ${ctx.raw.y.toFixed(1)} | Vol: ${ctx.raw.x}`,
                    },
                },
            },
        },
    });

    // 2. Sentiment Doughnut
    sentimentChart = new Chart($("sentimentChart").getContext("2d"), {
        type: "doughnut",
        data: {
            labels: ["Positive", "Negative", "Neutral"],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [C.emerald, C.rose, C.gray],
                borderColor: "transparent",
                borderWidth: 0,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "68%",
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { padding: 16, usePointStyle: true, pointStyleWidth: 8, font: { size: 11 } },
                },
                tooltip: {
                    backgroundColor: "#fff",
                    borderColor: C.emerald,
                    borderWidth: 1,
                },
            },
        },
    });

    // 3. Feature × Sentiment (Stacked Bar — Sunburst alternative)
    featureChart = new Chart($("featureChart").getContext("2d"), {
        type: "bar",
        data: {
            labels: [],
            datasets: [
                { label: "Positive", data: [], backgroundColor: C.emerald, borderRadius: 4 },
                { label: "Negative", data: [], backgroundColor: C.rose, borderRadius: 4 },
                { label: "Neutral",  data: [], backgroundColor: C.gray, borderRadius: 4 },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
                x: {
                    stacked: true,
                    grid: { color: C.gridLine },
                    border: { color: C.gridLine },
                },
                y: {
                    stacked: true,
                    grid: { display: false },
                    border: { color: C.gridLine },
                },
            },
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { padding: 14, usePointStyle: true, pointStyleWidth: 8, font: { size: 11 } },
                },
                tooltip: {
                    backgroundColor: "#fff",
                    borderColor: C.purple,
                    borderWidth: 1,
                },
            },
        },
    });

    // 4. Sentiment Timeline (Area Line)
    timelineChart = new Chart($("timelineChart").getContext("2d"), {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Positive",
                    data: [],
                    borderColor: C.emerald,
                    backgroundColor: C.emeraldDim,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 3,
                    pointBackgroundColor: C.emerald,
                },
                {
                    label: "Negative",
                    data: [],
                    borderColor: C.rose,
                    backgroundColor: C.roseDim,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 3,
                    pointBackgroundColor: C.rose,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false }, border: { color: C.gridLine } },
                y: { grid: { color: C.gridLine }, border: { color: C.gridLine }, beginAtZero: true },
            },
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { padding: 14, usePointStyle: true, pointStyleWidth: 8, font: { size: 11 } },
                },
                tooltip: {
                    backgroundColor: "#fff",
                    borderColor: C.blue,
                    borderWidth: 1,
                },
            },
        },
    });
}

// ============================================================
//  DASHBOARD UPDATE LOOP
// ============================================================
async function updateDashboard() {
    try {
        // --- Metrics ---
        const metricsRes = await fetch(`${API}/metrics`);
        const m = await metricsRes.json();

        animateValue(kpiTotal, m.total_processed);
        animateValue(kpiClean, m.clean_count || 0);
        animateValue(kpiSpam, m.spam_count || 0);

        // GSI
        if (m.gsi > 0) {
            kpiGsi.textContent = `${m.gsi}%`;
            kpiGsi.className = m.gsi >= 60
                ? "kpi-value emerald"
                : "kpi-value rose";
        } else {
            kpiGsi.textContent = "—";
        }

        // Sentiment Doughnut (real data!)
        sentimentChart.data.datasets[0].data = [m.positive || 0, m.negative || 0, m.neutral || 0];
        sentimentChart.update("none");

        // Feature × Sentiment Breakdown
        const fMap = m.feature_map || {};
        const features = Object.keys(fMap);
        if (features.length > 0) {
            featureChart.data.labels = features;
            featureChart.data.datasets[0].data = features.map(f => fMap[f].positive);
            featureChart.data.datasets[1].data = features.map(f => fMap[f].negative);
            featureChart.data.datasets[2].data = features.map(f => fMap[f].neutral);
            featureChart.update("none");
        }

        // Sentiment Timeline
        const timeline = m.sentiment_timeline || [];
        if (timeline.length > 0) {
            const labels = timeline.map((_, i) => `#${i + 1}`);
            let posAcc = 0, negAcc = 0;
            const posData = timeline.map(t => { posAcc += t.pos; return posAcc; });
            const negData = timeline.map(t => { negAcc += t.neg; return negAcc; });
            timelineChart.data.labels = labels;
            timelineChart.data.datasets[0].data = posData;
            timelineChart.data.datasets[1].data = negData;
            timelineChart.update("none");
        }

        // --- Review Feed ---
        renderReviewFeed(m.history || []);

        // --- Sarcasm Queue ---
        const sqRes = await fetch(`${API}/sarcasm-queue`);
        const sqData = await sqRes.json();
        renderSarcasmQueue(sqData.queue);

        // --- Anomalies ---
        const anomRes = await fetch(`${API}/detect-anomalies`);
        const anomalies = await anomRes.json();

        if (anomalies.anomalies_detected !== undefined) {
            animateValue(kpiAnomalies, anomalies.anomalies_detected);
            renderActionCenter(anomalies.clusters);
            updateHeatmap(anomalies.clusters);
        }

    } catch (err) {
        console.error("Dashboard sync error:", err);
    }
}

// ============================================================
//  HEATMAP UPDATE
// ============================================================
function updateHeatmap(clusters) {
    if (!clusters || clusters.length === 0) {
        heatmapChart.data.datasets = [];
        heatmapChart.update("none");
        return;
    }

    const pts = clusters.map(c => ({
        x: c.review_count,
        y: c.predictive_risk_score,
        r: Math.min(Math.max(c.review_count * 5, 8), 28),
        label: c.main_feature,
    }));

    heatmapChart.data.datasets = [{
        data: pts,
        backgroundColor: C.roseDim,
        borderColor: C.rose,
        borderWidth: 2,
        hoverBackgroundColor: C.rose,
    }];
    heatmapChart.update("none");
}

// ============================================================
//  ACTION CENTER (Anomaly Clusters)
// ============================================================
function renderActionCenter(clusters) {
    if (!clusters || clusters.length === 0) {
        actionCenter.innerHTML = `<div class="empty-state">System stable. No risk clusters detected.</div>`;
        return;
    }

    actionCenter.innerHTML = "";
    clusters.forEach(c => {
        const card = document.createElement("div");
        card.className = "alert-card";
        card.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span class="alert-feature">▲ ${c.main_feature}</span>
                <span class="risk-badge">RISK ${c.predictive_risk_score}</span>
            </div>
            <div class="alert-meta">${c.review_count} reports · ${c.negative_count || 0} negative signals</div>
            <button class="btn-resolve" onclick="generateStrategy(${c.cluster_id}, '${c.main_feature.replace(/'/g, "\\'")}')">
                ⚡ RESOLVE TREND — Gated RAG
            </button>
        `;
        actionCenter.appendChild(card);
    });
}

// ============================================================
//  SARCASM / UNCERTAINTY QUEUE
// ============================================================
function renderSarcasmQueue(queue) {
    if (!queue || queue.length === 0) {
        sarcasmQueue.innerHTML = `<div class="empty-state">Queue empty — no ambiguous reviews.</div>`;
        return;
    }

    sarcasmQueue.innerHTML = "";
    [...queue].reverse().forEach(q => {
        const card = document.createElement("div");
        card.className = "queue-card";
        card.innerHTML = `
            <div class="queue-status">${q.status}</div>
            <div class="queue-text">"${truncate(q.raw_text, 120)}"</div>
        `;
        sarcasmQueue.appendChild(card);
    });
}

// ============================================================
//  LIVE REVIEW FEED
// ============================================================
function renderReviewFeed(history) {
    if (!history || history.length === 0) {
        reviewFeed.innerHTML = `<div class="empty-state">No reviews ingested yet.</div>`;
        return;
    }

    reviewFeed.innerHTML = "";
    [...history].reverse().slice(0, 20).forEach(item => {
        const card = document.createElement("div");
        card.className = "feed-card";

        let statusClass = "clean";
        if (item.is_spam) statusClass = "spam";
        else if ((item.status || "").includes("Sarcasm")) statusClass = "sarcasm";

        let featureTags = "";
        const ext = item.absa_result || {};
        if (ext.extractions) {
            featureTags = ext.extractions.map(e => {
                const cls = (e.sentiment || "").toLowerCase() === "negative" ? "negative" : "";
                return `<span class="feature-tag ${cls}">${e.feature} · ${e.sentiment}</span>`;
            }).join("");
        }

        card.innerHTML = `
            <span class="feed-status ${statusClass}">${item.status || "Processed"}</span>
            <div class="feed-text">${truncate(item.clean_text || item.raw_text, 140)}</div>
            ${featureTags ? `<div class="feed-features">${featureTags}</div>` : ""}
        `;
        reviewFeed.appendChild(card);
    });
}

// ============================================================
//  STRATEGY MODAL (Layer 4 RAG)
// ============================================================
window.generateStrategy = async function(clusterId, featureName) {
    // Open modal
    modal.classList.add("active");
    modalSubtitle.textContent = `Resolving: ${featureName} (Cluster #${clusterId})`;
    modalContent.innerHTML = `
        <div class="spinner-container">
            <div class="spinner"></div>
            <span class="spinner-text">Cross-referencing reviews against Industry Specs…</span>
        </div>`;
    downloadBtn.style.display = "none";
    currentBrief = "";

    try {
        const res = await fetch(`${API}/generate-strategy/${clusterId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Strategy generation failed.");
        }

        const data = await res.json();
        const s = data.strategy;

        // Defect type badge
        const defectType = s.defect_type || "Unknown";
        let badgeClass = "badge badge-sentiment";
        if (defectType.includes("Zero-Day")) badgeClass = "badge badge-zero-day";
        else if (defectType.includes("Documented")) badgeClass = "badge badge-documented";

        // Priority badge
        const priority = s.roadmap_recommendation?.priority || "P1";
        let priClass = "badge-priority badge-p1";
        if (priority === "P0") priClass = "badge-priority badge-p0";
        else if (priority === "P2") priClass = "badge-priority badge-p2";

        modalContent.innerHTML = `
            <!-- Classification Row -->
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px">
                <span class="${badgeClass}">${defectType}</span>
                <span class="${priClass}">${priority}</span>
            </div>

            <!-- Reviews Analyzed -->
            <div style="font-size:0.72rem;color:var(--text-muted);font-family:var(--font-mono);margin-bottom:16px">
                ${data.reviews_analyzed} reviews cross-referenced against industry specifications
            </div>

            <!-- Industry Benchmark Gap -->
            <div class="strategy-section" style="border-color:var(--border-purple)">
                <div class="strategy-label purple">📐 Industry Benchmark Gap</div>
                <div class="strategy-text">${s.industry_benchmark_gap || "N/A"}</div>
            </div>

            <!-- Technical Report -->
            <div class="strategy-section">
                <div class="strategy-label emerald">🛠️ Engineering Brief</div>
                <div class="strategy-text">${s.technical_report || "N/A"}</div>
            </div>

            <!-- Marketing Response -->
            <div class="strategy-section">
                <div class="strategy-label blue">📢 Customer Support Auto-Reply</div>
                <div class="strategy-text" style="font-style:italic">"${s.marketing_response || "N/A"}"</div>
            </div>

            <!-- Roadmap -->
            <div class="strategy-section">
                <div class="strategy-label amber">🗺️ Roadmap Recommendation</div>
                <div style="display:flex;align-items:start;gap:12px">
                    <span class="${priClass}" style="flex-shrink:0">${priority}</span>
                    <div class="strategy-text">${s.roadmap_recommendation?.next_step || "N/A"}</div>
                </div>
            </div>

            <!-- HITL: Approve & Post to Storefront -->
            <div style="margin-top:18px;padding-top:18px;border-top:1px solid var(--border)">
                <button id="approve-post-btn"
                    style="width:100%;padding:12px 20px;background:var(--success);color:#fff;border:none;border-radius:var(--radius);font-family:var(--font);font-size:0.88rem;font-weight:700;cursor:pointer;transition:all 0.15s"
                    onclick="approveAndPost(${clusterId})">
                    ✓ Approve & Post Reply to Storefront
                </button>
                <div style="font-size:0.7rem;color:var(--text-muted);text-align:center;margin-top:6px">
                    Human-in-the-Loop gate — reply will appear on the storefront once approved
                </div>
            </div>
        `;

        // Build downloadable markdown
        currentBrief = [
            `# Strategy Brief: ${featureName} (Cluster #${clusterId})`,
            ``,
            `**Classification:** ${defectType}`,
            `**Priority:** ${priority}`,
            `**Reviews Analyzed:** ${data.reviews_analyzed}`,
            ``,
            `## 📐 Industry Benchmark Gap`,
            s.industry_benchmark_gap || "N/A",
            ``,
            `## 🛠️ Engineering Brief`,
            s.technical_report || "N/A",
            ``,
            `## 📢 Customer Support Auto-Reply`,
            `> ${s.marketing_response || "N/A"}`,
            ``,
            `## 🗺️ Roadmap Recommendation`,
            `- **Priority:** ${priority}`,
            `- **Next Step:** ${s.roadmap_recommendation?.next_step || "N/A"}`,
        ].join("\n");

        pendingReplyText = s.marketing_response || "";
        downloadBtn.style.display = "block";
        toast(`Strategy synthesized for ${featureName}`, "success");

    } catch (err) {
        modalContent.innerHTML = `
            <div class="strategy-section" style="border-color:var(--border-danger)">
                <div class="strategy-label rose">❌ Generation Failed</div>
                <div class="strategy-text">${err.message}</div>
            </div>`;
        toast(err.message, "error");
    }
};

// ============================================================
//  APPROVE & POST (Human-in-the-Loop)
// ============================================================
async function approveAndPost(clusterId) {
    const btn = document.getElementById("approve-post-btn");
    if (!btn || !pendingReplyText) return;

    btn.disabled = true;
    btn.textContent = "Posting...";
    btn.style.opacity = "0.7";

    try {
        const res = await fetch(`${API}/approve-cluster-replies/${clusterId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reply_text: pendingReplyText }),
        });

        const data = await res.json();

        btn.style.background = "var(--success)";
        btn.style.opacity = "1";
        btn.textContent = `✓ Approved — ${data.reviews_updated || 0} replies posted to storefront`;
        btn.disabled = true;

        toast(`Reply approved! ${data.reviews_updated || 0} reviews updated on storefront.`, "success");

    } catch (err) {
        btn.textContent = "✗ Failed — try again";
        btn.style.background = "var(--danger)";
        btn.style.opacity = "1";
        btn.disabled = false;
        toast(`Approval failed: ${err.message}`, "error");
    }
}

// ============================================================
//  MODAL CONTROLS
// ============================================================
closeModal.addEventListener("click", () => {
    modal.classList.remove("active");
});

modal.addEventListener("click", e => {
    if (e.target === modal) modal.classList.remove("active");
});

downloadBtn.addEventListener("click", () => {
    if (!currentBrief) return;
    const blob = new Blob([currentBrief], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "Strategy_Brief.md";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast("Brief downloaded!", "success");
});

// ============================================================
//  REVIEW INGESTION
// ============================================================
ingestBtn.addEventListener("click", async () => {
    const text = reviewInput.value.trim();
    if (!text) return;

    showStatus("⏳ Transmitting to Layer 1…", "success");
    ingestBtn.disabled = true;

    try {
        const res = await fetch(`${API}/stream-review`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text,
                translate: translateToggle.checked,
            }),
        });
        const data = await res.json();

        if (data.is_spam) {
            showStatus(`🛡️ ${data.pipeline_status}`, "danger");
            toast("Spam detected — routed to audit trail", "warn");
        } else if ((data.pipeline_status || "").includes("Sarcasm")) {
            showStatus(`⚠️ ${data.pipeline_status}`, "warn");
            toast("Low confidence — sent to Uncertainty Queue", "warn");
        } else {
            showStatus(`✅ ${data.pipeline_status}`, "success");
            toast("Review processed successfully", "success");
        }

        reviewInput.value = "";
        updateDashboard();

    } catch (err) {
        showStatus("❌ Connection to engine failed", "danger");
        toast("Connection failed", "error");
    } finally {
        ingestBtn.disabled = false;
    }
});

// ============================================================
//  BATCH UPLOAD — Sequential with Live Progress
// ============================================================
batchBtn.addEventListener("click", () => batchFile.click());

batchFile.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    ingestBtn.disabled = true;
    batchBtn.disabled = true;

    try {
        const content = await file.text();
        const parsed = JSON.parse(content);

        // Accept either an array of strings or { reviews: [...] }
        let reviews = [];
        if (Array.isArray(parsed)) reviews = parsed;
        else if (parsed.reviews && Array.isArray(parsed.reviews)) reviews = parsed.reviews;
        else throw new Error("JSON must be an array of strings or { reviews: [...] }");

        const total = reviews.length;
        let processed = 0;
        let errors = 0;

        showStatus(`⏳ Batch: 0/${total} processed…`, "success");
        toast(`Starting batch: ${total} reviews`, "info");

        for (const review of reviews) {
            try {
                await fetch(`${API}/stream-review`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        text: review,
                        translate: translateToggle.checked,
                    }),
                });
                processed++;
            } catch {
                errors++;
                processed++;
            }

            // Update progress every review
            showStatus(`⏳ Batch: ${processed}/${total} processed…`, "success");

            // Refresh dashboard every 5 reviews
            if (processed % 5 === 0) {
                updateDashboard();
            }
        }

        showStatus(`✅ Batch complete: ${processed}/${total} reviews (${errors} errors)`, "success");
        toast(`Batch done: ${processed} reviews ingested!`, "success");
        updateDashboard();

    } catch (err) {
        showStatus(`❌ ${err.message}`, "danger");
        toast(`Batch failed: ${err.message}`, "error");
    } finally {
        ingestBtn.disabled = false;
        batchBtn.disabled = false;
        batchFile.value = "";
    }
});

// ============================================================
//  HELPERS
// ============================================================
function showStatus(msg, type) {
    ingestStatus.textContent = msg;
    ingestStatus.className = `status-msg show ${type}`;
    setTimeout(() => {
        ingestStatus.className = "status-msg";
    }, 5000);
}

function truncate(str, len) {
    if (!str) return "";
    return str.length > len ? str.slice(0, len) + "…" : str;
}

// ============================================================
//  BOOT SEQUENCE
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    updateDashboard();

    // Auto-sync every 10 seconds
    setInterval(updateDashboard, 10000);

    console.log("%c⚡ Nexus Intelligence v2.0 — Online", "color: #34d399; font-weight: bold; font-size: 14px;");
});
