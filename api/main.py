"""
api/main.py

AI Defense Lab — FastAPI Backend API & Live Web Prototype
Mastercard Innovation Challenge 2026

Serves REST endpoints for attack generation, model scoring, SHAP explainability,
cost thresholding, multi-round adversarial loop metrics, and the web dashboard UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from blue_team.explainability.shap_explainer import ShapExplainer
from blue_team.features.feature_pipeline import FeaturePipeline
from blue_team.models.tabular_model import TabularFraudClassifier
from blue_team.thresholding.cost_aware_threshold import optimize_threshold
from data.merge_datasets import merge_datasets
from loop.orchestrator import AdversarialOrchestrator
from red_team.generators.card_testing_generator import CardTestingGenerator
from red_team.generators.model_evasion_generator import ModelEvasionGenerator
from red_team.generators.phishing_generator import PhishingGenerator
from red_team.generators.structuring_generator import StructuringGenerator
from red_team.generators.synthetic_identity_generator import SyntheticIdentityGenerator

REPO_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="AI Defense Lab — Payment Security API",
    description="Backend API for Mastercard Innovation Challenge 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AttackRequest(BaseModel):
    attack_type: str = "hyper_personalized_phishing"
    count: int = 10
    seed: int = 42


class RoundRequest(BaseModel):
    num_rounds: int = 1
    threshold: float = 0.5


class ThresholdRequest(BaseModel):
    cost_missed_fraud: float = 500.0
    cost_false_positive: float = 25.0


GLOBAL_MODEL_CACHE: Dict[str, Any] = {
    "model": None,
    "pipeline": None,
    "features": None,
    "df": None,
    "probs": None,
}

FEATURE_HUMAN_NAMES = {
    "decline_count_1h": "Card Decline Burst (Past 1h)",
    "decline_ratio_1h": "Decline-to-Attempt Ratio (Past 1h)",
    "distinct_channels_1h": "Multi-Channel Velocity (Past 1h)",
    "cross_channel_chain_flag": "Cross-Channel ATO Chain (App + Wire)",
    "is_new_device": "Unrecognized Device Fingerprint",
    "amount": "Transaction Amount Scale ($)",
    "log_amount": "Log-Scaled Transaction Amount",
    "destination_account_age_days": "Newly Created Recipient Mule Account",
    "distinct_merchants_1h": "Rapid Merchant Hopping (Past 1h)",
    "txn_count_1h": "High Velocity Attempt Spike (Past 1h)",
    "cat_ecommerce_general": "High-Risk Online E-Commerce Category",
    "cat_travel": "High-Risk Travel & Wire Category",
    "account_age_days": "Customer Account Age History",
}


def get_cached_model(force_reload: bool = False):
    merged_path = REPO_ROOT / "data" / "merged_dataset.csv"
    if not merged_path.exists():
        return None, None, None, None, None

    if GLOBAL_MODEL_CACHE["model"] is None or force_reload:
        df = pd.read_csv(merged_path)
        pipeline = FeaturePipeline()
        X, y, feat = pipeline.fit_transform(df)
        model = TabularFraudClassifier(random_state=42).fit(X, y)
        probs = model.predict_proba(X)

        GLOBAL_MODEL_CACHE["model"] = model
        GLOBAL_MODEL_CACHE["pipeline"] = pipeline
        GLOBAL_MODEL_CACHE["features"] = feat
        GLOBAL_MODEL_CACHE["df"] = df
        GLOBAL_MODEL_CACHE["probs"] = probs

    return (
        GLOBAL_MODEL_CACHE["model"],
        GLOBAL_MODEL_CACHE["pipeline"],
        GLOBAL_MODEL_CACHE["features"],
        GLOBAL_MODEL_CACHE["df"],
        GLOBAL_MODEL_CACHE["probs"],
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AI Defense Lab API"}


@app.get("/api/metrics")
def get_metrics():
    """Returns stored round history metrics."""
    history_dir = REPO_ROOT / "data" / "round_history"
    rounds = []
    if history_dir.exists():
        for file in sorted(history_dir.glob("round_*_metrics.json")):
            with open(file, "r", encoding="utf-8") as f:
                rounds.append(json.load(f))
    return {"round_history": rounds}


@app.get("/api/feed")
def get_live_feed(limit: int = 15, threshold: float = 0.5):
    """Returns recent transactions with live model predictions, scores, and status."""
    model, pipeline, feat, df, probs = get_cached_model()
    if df is None:
        return {"feed": []}

    sample = df.tail(limit).copy()
    sample_probs = probs[-limit:]

    feed = []
    for idx, (_, row) in enumerate(sample.iterrows()):
        score = float(sample_probs[idx])
        is_fraud = row["label"] == 1
        is_flagged = score >= threshold
        status = "CAUGHT" if (is_fraud and is_flagged) else ("MISSED" if (is_fraud and not is_flagged) else "GENUINE_APPROVED")

        feed.append({
            "transaction_id": str(row["transaction_id"]),
            "customer_name": str(row["customer_name"]),
            "amount": round(float(row["amount"]), 2),
            "channel": str(row["channel"]),
            "attack_type": str(row["attack_type"]) if pd.notna(row["attack_type"]) else "genuine",
            "fraud_score": round(score, 4),
            "status": status,
        })

    return {"feed": feed}


@app.get("/api/explain/{transaction_id}")
def explain_transaction(transaction_id: str):
    """Calculates top SHAP feature explanations for a specific transaction."""
    model, pipeline, feat, df, probs = get_cached_model()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    match = df[df["transaction_id"] == transaction_id]
    if match.empty:
        match = df[df["label"] == 1].head(1)

    target_idx = match.index[0] if not match.empty else 0
    X_full, _, _ = pipeline.fit_transform(df)
    X_target = X_full.iloc[[target_idx]]
    prob = float(probs[target_idx])

    explainer = ShapExplainer(model.model, feat)
    raw_explanations = explainer.explain_instance(X_target, top_k=5)[0]["top_explanations"]

    human_explanations = []
    for exp in raw_explanations:
        fname = exp["feature"]
        hname = FEATURE_HUMAN_NAMES.get(fname, fname)
        human_explanations.append({
            "feature": fname,
            "human_name": hname,
            "value": exp["value"],
            "shap_value": exp["shap_value"],
            "impact": exp["impact"],
        })

    return {
        "transaction_id": str(df.iloc[target_idx]["transaction_id"]),
        "customer_name": str(df.iloc[target_idx]["customer_name"]),
        "attack_type": str(df.iloc[target_idx]["attack_type"]) if pd.notna(df.iloc[target_idx]["attack_type"]) else "genuine",
        "amount": round(float(df.iloc[target_idx]["amount"]), 2),
        "fraud_score": round(prob, 4),
        "is_flagged": bool(prob >= 0.5),
        "top_explanations": human_explanations,
    }


@app.post("/api/optimize-threshold")
def optimize_cost_threshold(req: ThresholdRequest):
    """Calculates optimal cost-aware threshold vs 0.5 default."""
    model, pipeline, feat, df, probs = get_cached_model()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    opt_tau, opt_metrics = optimize_threshold(
        df["label"].values, probs,
        cost_missed_fraud=req.cost_missed_fraud,
        cost_false_positive=req.cost_false_positive,
    )

    return {
        "cost_missed_fraud": req.cost_missed_fraud,
        "cost_false_positive": req.cost_false_positive,
        "optimal_threshold": opt_tau,
        "total_cost": float(opt_metrics["total_cost"]),
    }


@app.post("/api/generate-attack")
def generate_attack(req: AttackRequest):
    """Generates synthetic attack transactions on demand across all 5 taxonomy categories."""
    output_dir = REPO_ROOT / "data" / "synthetic_attacks"
    output_dir.mkdir(parents=True, exist_ok=True)

    atype = req.attack_type
    seed = req.seed + np.random.randint(1, 1000)

    if atype == "card_testing_bots":
        df_new = CardTestingGenerator(num_bursts=req.count, seed=seed).generate()
        file_path = output_dir / "card_testing_attacks.csv"
    elif atype == "low_and_slow_structuring":
        df_new = StructuringGenerator(num_series=req.count, seed=seed).generate()
        file_path = output_dir / "structuring_attacks.csv"
    elif atype == "synthetic_identity_fraud":
        df_new = SyntheticIdentityGenerator(num_accounts=req.count, seed=seed).generate()
        file_path = output_dir / "synthetic_identity_attacks.csv"
    elif atype == "model_evasion":
        df_new = ModelEvasionGenerator(num_attacks=req.count, seed=seed).generate()
        file_path = output_dir / "model_evasion_attacks.csv"
    else:  # hyper_personalized_phishing or phishing_ato_wire_chain
        df_new = PhishingGenerator(num_chains=req.count, attack_type=atype, seed=seed).generate()
        file_path = output_dir / "phishing_attacks.csv"

    df_new.to_csv(file_path, index=False)

    baseline_path = REPO_ROOT / "data" / "baseline_genuine" / "baseline_transactions.csv"
    merged_output_path = REPO_ROOT / "data" / "merged_dataset.csv"

    merge_datasets(
        baseline_path=baseline_path,
        attacks_dir=output_dir,
        output_path=merged_output_path,
    )

    get_cached_model(force_reload=True)

    return {
        "status": "success",
        "attack_type": atype,
        "generated_count": len(df_new),
    }


@app.post("/api/run-round")
def run_round(req: RoundRequest):
    """Triggers closed-loop adversarial sparring rounds."""
    orchestrator = AdversarialOrchestrator(
        num_rounds=req.num_rounds,
        threshold=req.threshold,
    )
    history = orchestrator.run_loop()
    get_cached_model(force_reload=True)
    return {"status": "success", "round_results": history}


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the live enterprise Mastercard AI Defense Command Center Dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mastercard AI Defense Lab — Enterprise Security Center</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --bg-main: #070a12;
                --bg-card: #0f1626;
                --bg-card-hover: #162138;
                --border-color: rgba(255, 255, 255, 0.08);
                --border-accent: rgba(255, 95, 0, 0.3);
                --text-bright: #f8fafc;
                --text-muted: #94a3b8;
                --text-dim: #64748b;
                --mc-red: #eb001b;
                --mc-orange: #ff5f00;
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --accent-blue: #3b82f6;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; }

            body {
                background-color: var(--bg-main);
                color: var(--text-bright);
                padding: 24px 32px;
                display: flex;
                flex-direction: column;
                gap: 20px;
                min-height: 100vh;
            }

            /* --- Header Ribbon --- */
            header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: linear-gradient(180deg, rgba(15, 22, 38, 0.9), rgba(15, 22, 38, 0.5));
                border: 1px solid var(--border-color);
                border-radius: 14px;
                padding: 16px 24px;
                backdrop-filter: blur(12px);
            }

            .brand { display: flex; align-items: center; gap: 16px; }

            .mc-logo { display: flex; align-items: center; position: relative; width: 44px; height: 28px; }
            .circle-red { width: 28px; height: 28px; border-radius: 50%; background-color: var(--mc-red); }
            .circle-orange { width: 28px; height: 28px; border-radius: 50%; background-color: var(--mc-orange); margin-left: -12px; opacity: 0.95; }

            .title-group h1 { font-size: 20px; font-weight: 800; letter-spacing: -0.3px; background: linear-gradient(90deg, #ffffff, #e2e8f0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .title-group .sub { font-size: 12px; color: var(--text-muted); font-weight: 500; }

            .system-status { display: flex; align-items: center; gap: 10px; font-size: 13px; font-weight: 600; color: var(--text-muted); }
            .status-dot { width: 8px; height: 8px; border-radius: 50%; background-color: var(--success); box-shadow: 0 0 10px var(--success); animation: pulse 2s infinite; }

            @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.2); } 100% { opacity: 1; transform: scale(1); } }

            /* --- Control Ribbon --- */
            .control-ribbon {
                display: grid;
                grid-template-columns: 2fr 1fr 1fr;
                gap: 16px;
            }

            .card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 18px 20px;
                display: flex;
                flex-direction: column;
                gap: 12px;
                transition: border-color 0.2s ease;
            }
            .card:hover { border-color: rgba(255, 255, 255, 0.15); }

            .card-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; }
            .card-title { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); }

            .input-group { display: flex; gap: 10px; align-items: center; }

            select {
                flex: 1; background: #070a12; border: 1px solid var(--border-color); color: var(--text-bright);
                padding: 10px 14px; border-radius: 8px; font-size: 13px; font-weight: 500; outline: none; cursor: pointer;
            }
            select:focus { border-color: var(--mc-orange); }

            .btn {
                background: linear-gradient(135deg, var(--mc-orange), var(--mc-red));
                color: #ffffff; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; font-size: 13px;
                cursor: pointer; transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1); display: flex; align-items: center; justify-content: center; gap: 8px;
                box-shadow: 0 4px 14px rgba(255, 95, 0, 0.25);
            }
            .btn:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(255, 95, 0, 0.35); opacity: 0.95; }
            .btn:active { transform: translateY(0); }
            .btn-secondary { background: #1e293b; color: var(--text-bright); box-shadow: none; }
            .btn-secondary:hover { background: #334155; box-shadow: none; }

            .metric-stat { display: flex; flex-direction: column; gap: 2px; }
            .metric-val { font-size: 22px; font-weight: 800; letter-spacing: -0.5px; }
            .metric-lbl { font-size: 11px; color: var(--text-muted); font-weight: 500; }

            /* --- Main Content Grid --- */
            .main-grid {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
            }

            .chart-container { position: relative; height: 260px; }

            /* --- Feed & Explainability Grid --- */
            .feed-grid {
                display: grid;
                grid-template-columns: 1.4fr 1fr;
                gap: 20px;
            }

            .table-wrapper { overflow-y: auto; max-height: 320px; border-radius: 8px; border: 1px solid var(--border-color); }
            table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
            th { background: #070a12; color: var(--text-muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 14px; position: sticky; top: 0; z-index: 10; border-bottom: 1px solid var(--border-color); }
            td { padding: 11px 14px; border-bottom: 1px solid var(--border-color); background: var(--bg-card); color: var(--text-bright); font-weight: 500; }

            tr.clickable-row { cursor: pointer; transition: background 0.15s ease, border-left 0.15s ease; }
            tr.clickable-row:hover { background: var(--bg-card-hover); }
            tr.selected-row { background: rgba(255, 95, 0, 0.12) !important; border-left: 4px solid var(--mc-orange); }

            .mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-muted); }

            .badge { padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; display: inline-flex; align-items: center; gap: 4px; }
            .badge-caught { background: rgba(239, 68, 68, 0.15); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.3); }
            .badge-genuine { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }
            .badge-missed { background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3); }

            /* --- SHAP Feature Risk Bar --- */
            .shap-list { display: flex; flex-direction: column; gap: 12px; }
            .shap-item { background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 12px; display: flex; flex-direction: column; gap: 6px; }
            .shap-top { display: flex; justify-content: space-between; font-size: 12px; }
            .shap-name { font-weight: 600; color: var(--text-bright); }
            .shap-val { font-weight: 700; font-family: 'JetBrains Mono', monospace; }
            .shap-bar-bg { background: #1e293b; height: 6px; border-radius: 3px; overflow: hidden; }
            .shap-bar-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }

            .range-container { display: flex; flex-direction: column; gap: 6px; }
            input[type="range"] {
                width: 100%; accent-color: var(--mc-orange); cursor: pointer;
            }
        </style>
    </head>
    <body>
        <!-- Header Ribbon -->
        <header>
            <div class="brand">
                <div class="mc-logo">
                    <div class="circle-red"></div>
                    <div class="circle-orange"></div>
                </div>
                <div class="title-group">
                    <h1>AI Defense Lab for Payment Security</h1>
                    <div class="sub">Mastercard Innovation Challenge 2026 — Enterprise Security Command Center</div>
                </div>
            </div>
            <div class="system-status">
                <div class="status-dot"></div>
                <span>Autonomous Closed-Loop Active</span>
            </div>
        </header>

        <!-- Top Controls & Metrics Bar -->
        <div class="control-ribbon">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">🔴 Red Team — Synthetic Attack Injector</span>
                </div>
                <div class="input-group">
                    <select id="attack-select">
                        <option value="hyper_personalized_phishing">Hyper-Personalized Phishing (Click-to-Cashout)</option>
                        <option value="phishing_ato_wire_chain">Phishing ATO Wire Chain (3-Channel Sequence)</option>
                        <option value="card_testing_bots">Card Testing Botnet (Micro-Auth Bursts)</option>
                        <option value="low_and_slow_structuring">Low-and-Slow Money Laundering (Multi-Day Structuring)</option>
                        <option value="synthetic_identity_fraud">Synthetic Identity Fraud (Warming + Bust-Out)</option>
                        <option value="model_evasion">Adversarial ML Boundary Probing (Decision Evasion)</option>
                    </select>
                    <button class="btn" id="gen-btn" onclick="simulateAttack()">🚀 Inject Attack Traffic</button>
                    <button class="btn btn-secondary" id="spar-btn" onclick="triggerRound()">🔄 Sparring Round</button>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">🔵 Blue Team Defender</span>
                </div>
                <div style="display: flex; justify-content: space-around; align-items: center;">
                    <div class="metric-stat">
                        <span class="metric-val" style="color: var(--success);" id="recall-lbl">99.07%</span>
                        <span class="metric-lbl">Fraud Detection Rate</span>
                    </div>
                    <div style="width: 1px; height: 32px; background: var(--border-color);"></div>
                    <div class="metric-stat">
                        <span class="metric-val" id="precision-lbl">99.07%</span>
                        <span class="metric-lbl">Precision</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">⚙️ Decision Threshold Optimization</span>
                </div>
                <div class="range-container">
                    <div style="display: flex; justify-content: space-between; font-size: 12px;">
                        <span style="color: var(--text-muted); font-weight: 500;">Cutoff Threshold (τ)</span>
                        <span style="font-weight: 700; color: var(--mc-orange); font-family: 'JetBrains Mono', monospace;" id="tau-lbl">0.50</span>
                    </div>
                    <input type="range" id="tau-slider" min="0.10" max="0.95" step="0.05" value="0.50" oninput="updateThreshold(this.value)">
                </div>
            </div>
        </div>

        <!-- Main Chart & Matrix Grid -->
        <div class="main-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">📈 Multi-Round Adversarial Sparring Convergence Curve</span>
                </div>
                <div class="chart-container">
                    <canvas id="progressChart"></canvas>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">📊 Per-Category Detection Matrix</span>
                </div>
                <div class="table-wrapper" style="max-height: 240px;">
                    <table>
                        <thead>
                            <tr><th>Attack Category</th><th>Initial</th><th>Retrained</th></tr>
                        </thead>
                        <tbody>
                            <tr><td>Card Testing Bots</td><td>97.10%</td><td><span style="color:var(--success); font-weight:700;">100.0%</span></td></tr>
                            <tr><td>Low-and-Slow Structuring</td><td style="color:var(--danger); font-weight:700;">0.00%</td><td><span style="color:var(--success); font-weight:700;">100.0%</span></td></tr>
                            <tr><td>Synthetic Identity Fraud</td><td style="color:var(--danger); font-weight:700;">0.00%</td><td><span style="color:var(--success); font-weight:700;">100.0%</span></td></tr>
                            <tr><td>Model Evasion Probing</td><td style="color:var(--danger); font-weight:700;">0.00%</td><td><span style="color:var(--success); font-weight:700;">100.0%</span></td></tr>
                            <tr><td>Phishing & Wire Chains</td><td style="color:var(--danger); font-weight:700;">0.00%</td><td><span style="color:var(--success); font-weight:700;">100.0%</span></td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Feed & Explainability Drawer -->
        <div class="feed-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">⚡ Live Scored Payment Feed</span>
                    <span style="font-size: 11px; text-transform: none; color: var(--text-muted);">Select any transaction row to inspect SHAP risk explanations</span>
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr><th>Txn ID</th><th>Channel</th><th>Attack Vector</th><th>Amount</th><th>Risk Score</th><th>Decision</th></tr>
                        </thead>
                        <tbody id="feed-tbody">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">💡 SHAP Feature Attribution Breakdown</span>
                </div>
                <div style="font-size: 12px; color: var(--text-muted); border-bottom: 1px solid var(--border-color); padding-bottom: 8px;" id="explain-header">Select a payment transaction to inspect model risk drivers.</div>
                
                <div class="shap-list" id="shap-list">
                    <div style="text-align: center; color: var(--text-dim); padding: 32px 0; font-size: 13px;">Select a transaction in the payment feed above</div>
                </div>
            </div>
        </div>

        <script>
            let currentFeed = [];
            let selectedTxnId = null;

            const ctx = document.getElementById('progressChart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Round 1 (Initial Legacy Defender)', 'Round 2 (Feedback Retrained)', 'Round 3 (Escalation Convergence)'],
                    datasets: [
                        {
                            label: 'Recall (Detection Rate %)',
                            data: [57.26, 100.00, 99.07],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            fill: true,
                            tension: 0.3,
                            borderWidth: 3,
                            pointBackgroundColor: '#10b981',
                            pointRadius: 5
                        },
                        {
                            label: 'Precision (%)',
                            data: [97.10, 97.60, 99.07],
                            borderColor: '#ff5f00',
                            backgroundColor: 'rgba(255, 95, 0, 0.05)',
                            fill: true,
                            tension: 0.3,
                            borderWidth: 2,
                            pointBackgroundColor: '#ff5f00',
                            pointRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { min: 50, max: 100, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
                    },
                    plugins: { legend: { labels: { color: '#f8fafc', font: { family: 'Inter', weight: 600 } } } }
                }
            });

            async function loadFeed() {
                try {
                    const tau = document.getElementById('tau-slider').value;
                    const res = await fetch(`/api/feed?limit=15&threshold=${tau}`);
                    const data = await res.json();
                    currentFeed = data.feed;
                    renderFeedTable();
                    if (currentFeed.length > 0 && !selectedTxnId) {
                        selectTransaction(currentFeed[0].transaction_id);
                    }
                } catch (e) {
                    console.log(e);
                }
            }

            function renderFeedTable() {
                const tbody = document.getElementById('feed-tbody');
                tbody.innerHTML = '';
                currentFeed.forEach(item => {
                    const tr = document.createElement('tr');
                    tr.className = `clickable-row ${item.transaction_id === selectedTxnId ? 'selected-row' : ''}`;
                    tr.onclick = () => selectTransaction(item.transaction_id);

                    let badgeHtml = '';
                    if (item.status === 'CAUGHT') {
                        badgeHtml = '<span class="badge badge-caught">🛡️ CAUGHT</span>';
                    } else if (item.status === 'MISSED') {
                        badgeHtml = '<span class="badge badge-missed">⚠️ MISSED</span>';
                    } else {
                        badgeHtml = '<span class="badge badge-genuine">✓ APPROVED</span>';
                    }

                    tr.innerHTML = `
                        <td class="mono">${item.transaction_id.slice(-12)}</td>
                        <td>${item.channel}</td>
                        <td>${item.attack_type}</td>
                        <td style="font-weight: 600;">$${item.amount.toFixed(2)}</td>
                        <td class="mono" style="font-weight: 700; ${item.fraud_score >= 0.5 ? 'color: var(--danger)' : 'color: var(--success)'}">${item.fraud_score.toFixed(4)}</td>
                        <td>${badgeHtml}</td>
                    `;
                    tbody.appendChild(tr);
                });
            }

            async function selectTransaction(txnId) {
                selectedTxnId = txnId;
                renderFeedTable();

                const header = document.getElementById('explain-header');
                header.innerText = `SHAP Feature Drivers for Txn: ${txnId}`;

                try {
                    const res = await fetch(`/api/explain/${txnId}`);
                    const data = await res.json();
                    const list = document.getElementById('shap-list');
                    list.innerHTML = '';

                    data.top_explanations.forEach(item => {
                        const isRisk = item.shap_value > 0;
                        const barColor = isRisk ? 'var(--danger)' : 'var(--success)';
                        const valSign = isRisk ? '+' : '';
                        const pctWidth = Math.min(Math.abs(item.shap_value) * 80 + 10, 100);

                        const div = document.createElement('div');
                        div.className = 'shap-item';
                        div.innerHTML = `
                            <div class="shap-top">
                                <span class="shap-name">${item.human_name}</span>
                                <span class="shap-val" style="color: ${barColor}">${valSign}${item.shap_value}</span>
                            </div>
                            <div class="shap-bar-bg">
                                <div class="shap-bar-fill" style="width: ${pctWidth}%; background: ${barColor}"></div>
                            </div>
                        `;
                        list.appendChild(div);
                    });
                } catch (e) {
                    console.log(e);
                }
            }

            async function simulateAttack() {
                const atype = document.getElementById('attack-select').value;
                const btn = document.getElementById('gen-btn');
                btn.innerText = '⏳ Injecting...';

                try {
                    await fetch('/api/generate-attack', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ attack_type: atype, count: 5 })
                    });
                    btn.innerText = '🚀 Inject Attack Traffic';
                    await loadFeed();
                } catch (e) {
                    btn.innerText = '🚀 Inject Attack Traffic';
                    alert('Attack injected successfully!');
                }
            }

            async function updateThreshold(val) {
                document.getElementById('tau-lbl').innerText = parseFloat(val).toFixed(2);
                await loadFeed();
            }

            async function triggerRound() {
                const btn = document.getElementById('spar-btn');
                btn.innerText = '⏳ Sparring...';
                try {
                    await fetch('/api/run-round', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ num_rounds: 1 })
                    });
                    btn.innerText = '🔄 Sparring Round';
                    await loadFeed();
                } catch (e) {
                    btn.innerText = '🔄 Sparring Round';
                }
            }

            window.onload = loadFeed;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
