// ============================================================
//  NEXUS STOREFRONT — Mock Amazon Review Interface
// ============================================================

const API = "http://127.0.0.1:8000";

// DOM
const reviewInput = document.getElementById("review-input");
const submitBtn = document.getElementById("submit-btn");
const formStatus = document.getElementById("form-status");
const reviewsList = document.getElementById("reviews-list");
const reviewCount = document.getElementById("review-count");

// ============================================================
//  SUBMIT REVIEW
// ============================================================
submitBtn.addEventListener("click", async () => {
    const text = reviewInput.value.trim();
    if (!text) return;

    submitBtn.disabled = true;
    showStatus("Submitting your review...", "success");

    try {
        const res = await fetch(`${API}/stream-review`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, translate: true }),
        });

        const data = await res.json();

        if (data.is_spam) {
            showStatus("Your review was flagged by our spam filter.", "error");
        } else {
            showStatus("Review submitted successfully! It will appear below shortly.", "success");
            reviewInput.value = "";
        }

        // Immediately refresh feed
        fetchReviews();

    } catch (err) {
        showStatus("Error submitting review. Please try again.", "error");
    } finally {
        submitBtn.disabled = false;
    }
});

// ============================================================
//  FETCH & RENDER REVIEWS (Polls every 3s)
// ============================================================
async function fetchReviews() {
    try {
        const res = await fetch(`${API}/storefront-reviews`);
        const data = await res.json();
        renderReviews(data.reviews || []);
    } catch (err) {
        console.error("Failed to fetch reviews:", err);
    }
}

function renderReviews(reviews) {
    reviewCount.textContent = reviews.length;

    if (reviews.length === 0) {
        reviewsList.innerHTML = `<div class="empty-reviews">No reviews yet. Be the first to review this product!</div>`;
        return;
    }

    reviewsList.innerHTML = "";
    reviews.forEach(r => {
        const item = document.createElement("div");
        item.className = "review-item";

        // Generate random-looking star rating based on sentiment
        const ext = r.extraction || {};
        let sentiment = "neutral";
        let stars = 3;
        if (ext.extractions && ext.extractions.length > 0) {
            const sentiments = ext.extractions.map(e => (e.sentiment || "").toLowerCase());
            const hasNeg = sentiments.includes("negative");
            const hasPos = sentiments.includes("positive");
            if (hasNeg && !hasPos) { stars = 2; sentiment = "negative"; }
            else if (hasPos && !hasNeg) { stars = 5; sentiment = "positive"; }
            else if (hasPos && hasNeg) { stars = 3; sentiment = "mixed"; }
        }

        const starStr = "★".repeat(stars) + "☆".repeat(5 - stars);

        // Time ago
        const timeAgo = getTimeAgo(r.timestamp);

        // Feature tags
        let tags = "";
        if (ext.extractions && ext.extractions.length > 0) {
            tags = ext.extractions.map(e => {
                const s = (e.sentiment || "").toLowerCase();
                return `<span class="review-tag ${s}">${e.feature} · ${e.sentiment}</span>`;
            }).join("");
        }

        // Reviewer initials
        const initial = r.text.charAt(0).toUpperCase();

        // AI Reply
        let replyHTML = "";
        if (r.reply) {
            replyHTML = `
                <div class="ai-reply">
                    <div class="ai-reply-header">
                        <span class="ai-reply-badge">✓ Verified</span>
                        <span class="ai-reply-label">Nexus Support</span>
                    </div>
                    <div class="ai-reply-text">${escapeHtml(r.reply)}</div>
                </div>
            `;
        }

        item.innerHTML = `
            <div class="review-meta">
                <div class="reviewer-avatar">${initial}</div>
                <div class="reviewer-name">Customer</div>
                <div class="review-time">${timeAgo}</div>
            </div>
            <div class="review-stars">${starStr}</div>
            <div class="review-text">${escapeHtml(r.text)}</div>
            ${tags ? `<div class="review-tags">${tags}</div>` : ""}
            ${replyHTML}
        `;

        reviewsList.appendChild(item);
    });
}

// ============================================================
//  HELPERS
// ============================================================
function showStatus(msg, type) {
    formStatus.textContent = msg;
    formStatus.className = `store-status show ${type}`;
    setTimeout(() => formStatus.classList.remove("show"), 4000);
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function getTimeAgo(ts) {
    if (!ts) return "";
    const diff = Math.floor(Date.now() / 1000 - ts);
    if (diff < 60) return "Just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

// ============================================================
//  INIT — Start polling
// ============================================================
fetchReviews();
setInterval(fetchReviews, 3000);
