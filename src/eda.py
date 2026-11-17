"""Exploratory data analysis: saves figures to reports/figures/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

train = pd.read_csv("data/train.csv")
train["is_fraud"] = train["is_fraud"].astype(int)
base = train["is_fraud"].mean()

# 1. Fraud rate by categorical feature.
cats = ["merchant_category", "auth_method", "channel", "card_type"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, c in zip(axes.ravel(), cats):
    rate = train.groupby(c)["is_fraud"].mean().sort_values()
    ax.barh(rate.index, rate.values * 100, color="#4C72B0")
    ax.axvline(base * 100, color="#C44E52", ls="--", label=f"overall {base:.2%}")
    ax.set_title(f"Fraud rate by {c}")
    ax.set_xlabel("% fraud")
    ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fraud_rate_by_category.png", dpi=120)

# 2. Fraud rate by binary red-flag features.
flags = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
rates = pd.DataFrame({f: train.groupby(train[f].astype(str))["is_fraud"].mean() for f in flags}).T * 100
ax = rates.plot.barh(figsize=(9, 5), color=["#8C8C8C", "#C44E52"])
ax.set_xlabel("% fraud")
ax.set_title("Fraud rate when red flag is False vs True")
plt.tight_layout()
plt.savefig(FIG / "fraud_rate_by_flag.png", dpi=120)

# 3. Fraud rate across deciles of the strongest numeric features.
nums = ["merchant_risk_score", "velocity_score", "customer_age", "amount_usd"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, c in zip(axes, nums):
    bins = pd.qcut(train[c], 10, duplicates="drop")
    rate = train.groupby(bins, observed=True)["is_fraud"].mean() * 100
    ax.plot(range(len(rate)), rate.values, marker="o", color="#4C72B0")
    ax.set_title(c)
    ax.set_xlabel("decile")
    ax.set_ylabel("% fraud")
fig.tight_layout()
fig.savefig(FIG / "fraud_rate_by_decile.png", dpi=120)
print("figures written to", FIG)
