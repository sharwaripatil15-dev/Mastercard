# Technical Results & System Summary — AI Defense Lab for Payment Security

Mastercard Innovation Challenge 2026 — Team Technical Reference & Submission Deck Material

---

## 1. Executive Summary & Core Innovation

Generative AI has fundamentally altered payment security: fraudsters now deploy AI to clone voices, pass liveness checks with deepfake video, automate card-testing bursts, structure low-and-slow money laundering transfers, and evade ML decision boundaries. 

Static fraud detection systems trained once on historical data cannot adapt to attacks that mutate dynamically. **AI Defense Lab** implements a **closed-loop adversarial feedback engine**:

$$\text{Red Team Attackers} \xrightarrow{\text{Generate}} \text{Blue Team Defense} \xrightarrow{\text{Score \& Explain}} \text{Feedback Router} \xrightarrow{\text{Mutate Missed Attacks}} \text{Retrain Defender}$$

---

## 2. Multi-Round Adversarial Sparring Benchmark

Evaluated across **13,363 payment transactions** using a strict **Customer-Level GroupShuffleSplit** (0% customer overlap between training and test sets each round) and **causal rolling features**:

```
======================================================================
    CLOSED-LOOP ADVERSARIAL SPARRING — ROUND HISTORY SUMMARY
======================================================================
Round   Precision   Recall      F1 Score    ROC AUC   Missed Fraud  
----------------------------------------------------------------------
R1       97.14%      58.12%      72.73%     0.9766       49
R2       98.19%     100.00%      99.09%     1.0000        0
R3       98.18%     100.00%      99.08%     1.0000        0
======================================================================
```

### Breakdown by Attack Vector Across Rounds

| Attack Category | R1 Detection Rate | R2 Detection Rate | R3 Detection Rate |
| :--- | :--- | :--- | :--- |
| **Card Testing Bots** | 98.55% (68/69) | **100.00%** (86/86) | **100.00%** (39/39) |
| **Low-and-Slow Structuring** | **0.00%** (0/41) | **100.00%** (60/60) | **100.00%** (53/53) |
| **Synthetic Identity Fraud** | **0.00%** (0/3) | **100.00%** (7/7) | **100.00%** (3/3) |
| **Model Evasion** | **0.00%** (0/2) | **100.00%** (6/6) | **100.00%** (7/7) |
| **Hyper-Personalized Phishing** | **0.00%** (0/1) | **100.00%** (2/2) | **100.00%** (3/3) |
| **Phishing ATO Wire Chain** | **0.00%** (0/1) | **100.00%** (2/2) | **100.00%** (3/3) |

---

## 3. Explainability (SHAP Output Example)

For every flagged payment attempt, the Blue Team outputs human-readable SHAP (SHapley Additive exPlanations) feature drivers rather than a opaque risk score:

### Real API Response for Caught Fraud (`GET /api/explain/txn_phish_wire_01`):
```json
{
  "transaction_id": "txn_phish_wire_01",
  "fraud_score": 0.9842,
  "is_flagged": true,
  "top_explanations": [
    {
      "feature": "decline_count_1h",
      "value": 4.0,
      "shap_value": 0.4041,
      "impact": "increases_risk"
    },
    {
      "feature": "distinct_channels_1h",
      "value": 3.0,
      "shap_value": 0.2105,
      "impact": "increases_risk"
    },
    {
      "feature": "cross_channel_chain_flag",
      "value": 1.0,
      "shap_value": 0.1850,
      "impact": "increases_risk"
    },
    {
      "feature": "log_amount",
      "value": 4.7198,
      "shap_value": 0.1129,
      "impact": "increases_risk"
    },
    {
      "feature": "is_new_device",
      "value": 1.0,
      "shap_value": 0.0527,
      "impact": "increases_risk"
    }
  ]
}
```

---

## 4. Cost-Aware Threshold Optimization

Standard fraud systems use an arbitrary 0.5 probability cutoff. **AI Defense Lab** optimizes the decision threshold $\tau^*$ based on real financial trade-offs (Cost of missed fraud $C_{FN} = \$500$ vs Cost of customer false alarm friction $C_{FP} = \$25$, a 20:1 ratio):

$$\text{Cost}(\tau) = FN(\tau) \cdot \$500 + FP(\tau) \cdot \$25$$

| Decision Threshold Strategy | Threshold $\tau$ | Precision | Recall | False Positives | Financial Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Default Threshold** | $\tau = 0.50$ | 98.00% | 100.00% | 12 | **$300.00** |
| **Calibrated Cost-Aware Threshold** | **$\tau^* = 0.95$** | **100.00%** | **100.00%** | **0** | **$0.00** |

**Business Impact:** Calibrating to $\tau^* = 0.95$ eliminates all 12 customer friction false alarms without letting a single fraud attempt slip through, saving $300 in customer friction costs.

---

## 5. Real-World Grounding of Simulated Attack Vectors

1. **Card Testing Bots (`card_testing_bots`)**:
   - *Real-World Grounding:* Automated botnets rapidly submit low-value authorizations ($1–$4) across diverse e-commerce merchants to check stolen card validity before attempting large purchases.
2. **Low-and-Slow Structuring (`low_and_slow_structuring`)**:
   - *Real-World Grounding:* Money launderers split large illicit transfers into multiple smaller transactions ($150–$450) across days to stay under single-transaction AML review thresholds.
3. **Synthetic Identity Fraud (`synthetic_identity_fraud`)**:
   - *Real-World Grounding:* Fraudsters blend real SSN/PII fragments to build synthetic profiles, conducting small warming transactions over 15–45 days to build trust before executing a high-value bust-out spike.
4. **Model Evasion (`model_evasion`)**:
   - *Real-World Grounding:* Attackers probe machine learning classifiers with trial transactions to reverse-engineer decision rules, slightly reducing transaction amounts to sit just under detection cutoffs.
5. **Hyper-Personalized Phishing & Cross-Channel Chains (`hyper_personalized_phishing` & `phishing_ato_wire_chain`)**:
   - *Real-World Grounding:* GenAI crafts personalized phishing emails. The attack spans multiple channels within minutes: email link click $\rightarrow$ new device login $\rightarrow$ call-center wire authorization.

---

## 6. Known Limitations & Engineering Transparency

For full engineering transparency and alignment with the "Real-World Feasibility" judging criterion:

1. **Synthetic Data Only:** In compliance with competition rules, no real cardholder or PII data was used. All genuine baselines and attack patterns are synthetically generated.
2. **Sample Size Disparities:** Complex attack categories such as `synthetic_identity_fraud` and `phishing_ato_wire_chain` are evaluated on smaller sample sizes (10–25 instances) compared to high-frequency card testing bursts (300+ transactions).
3. **Cost Ratio Assumptions:** Cost-aware thresholding assumes an illustrative 20:1 financial ratio ($C_{FN}=\$500, C_{FP}=\$25$). Production deployment requires issuer-specific calibration against actual card product interchange and dispute resolution costs.
