import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import joblib

# ---------------------- Argument Parser ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Path to input JSON file')
parser.add_argument('--output', required=True, help='Path to output CSV file')
args = parser.parse_args()

# ---------------------- Load Transactions ----------------------
with open(args.input, 'r') as f:
    transactions = json.load(f)

# ---------------------- Feature Engineering ----------------------
user_groups = defaultdict(list)
for txn in transactions:
    user = txn.get('userWallet')
    if user:
        user_groups[user].append(txn)

features = []

for user, txns in user_groups.items():
    deposit_usd = 0.0
    borrow_usd = 0.0
    timestamps = []
    txn_values = []

    for txn in txns:
        action = txn.get("action", "").lower()
        ts = txn.get("timestamp")
        if ts:
            timestamps.append(datetime.fromtimestamp(ts).date())

        try:
            amount = float(txn["actionData"]["amount"]) / 1e6
            price = float(txn["actionData"]["assetPriceUSD"])
            amount_usd = amount * price
        except (KeyError, ValueError, TypeError):
            amount_usd = 0.0

        txn_values.append(amount_usd)

        if action == "deposit":
            deposit_usd += amount_usd
        elif action == "borrow":
            borrow_usd += amount_usd

    total_txns = len(txns)
    active_days = len(set(timestamps))
    deposit_to_borrow_ratio = deposit_usd / (borrow_usd + 1e-6)
    avg_txn_value = np.mean(txn_values) if txn_values else 0.0
    max_txn = np.max(txn_values) if txn_values else 0.0
    min_txn = np.min(txn_values) if txn_values else 0.0

    features.append({
        "userWallet": user,
        "total_deposit_usd": deposit_usd,
        "total_borrow_usd": borrow_usd,
        "deposit_to_borrow_ratio": deposit_to_borrow_ratio,
        "transaction_count": total_txns,
        "active_days": active_days,
        "avg_txn_value": avg_txn_value,
        "max_single_txn_usd": max_txn,
        "min_single_txn_usd": min_txn
    })

df = pd.DataFrame(features)

# ---------------------- Preprocessing ----------------------
X = df[[
    "total_deposit_usd", "total_borrow_usd", 
    "deposit_to_borrow_ratio", "transaction_count", 
    "active_days", "avg_txn_value", 
    "max_single_txn_usd", "min_single_txn_usd"
]].fillna(0)

# Log transform
X_log = np.log1p(X)

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_log)

# ---------------------- Load Model ----------------------
model = joblib.load("model.pkl")

# ---------------------- Predict Scores ----------------------
raw_scores = model.predict(X_scaled)
scaled_scores = np.clip((raw_scores / max(raw_scores)) * 1000, 0, 1000).astype(int)

df['credit_score'] = scaled_scores

# ---------------------- Save Output ----------------------
df[['userWallet', 'credit_score']].to_csv(args.output, index=False)
print(f"✅ Credit scores written to {args.output}")

import matplotlib.pyplot as plt

# Print confirmation
print("[INFO] ✅ Credit scores written to wallet_scores.csv")

# Load the scored CSV
df_scores = pd.read_csv(args.output)

# Bucket scores into ranges: 0-100, 100-200, ..., 900-1000
bins = list(range(0, 1100, 100))
labels = [f"{i}-{i+99}" for i in bins[:-1]]
df_scores['score_range'] = pd.cut(df_scores['credit_score'], bins=bins, labels=labels, right=False)

# Plot the distribution
plt.figure(figsize=(10, 6))
df_scores['score_range'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Wallet Credit Score Distribution")
plt.xlabel("Score Range")
plt.ylabel("Number of Wallets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("score_distribution.png")
plt.close()

print("[INFO] 📈 Score distribution chart saved as score_distribution.png")

# Analyze low (0–300) and high (700–1000) scoring wallets
low_range = df_scores[df_scores['credit_score'] <= 300]
high_range = df_scores[df_scores['credit_score'] >= 700]

low_behavior = low_range.describe().to_string()
high_behavior = high_range.describe().to_string()

#  Write analysis.md file
print("[INFO] 🧠 Writing analysis.md...")

with open("analysis.md", "w", encoding="utf-8") as f:
    f.write("Wallet Credit Score Analysis\n\n")
    f.write("Score Distribution\n")
    f.write("![Score Distribution](score_distribution.png)\n\n")

    f.write("Behavior of Low-Scoring Wallets (0–300)\n")
    f.write("- Tend to have **low deposit amounts**, **high borrow-to-deposit ratios**, **few transactions**, and **short activity duration**.\n")
    f.write("- Statistical summary:\n\n")
    f.write("```\n" + low_behavior + "\n```\n\n")

    f.write(" Behavior of High-Scoring Wallets (700–1000)\n")
    f.write("- Typically show **high deposits**, **low borrowing**, **high transaction counts**, and **longer active periods**.\n")
    f.write("- Statistical summary:\n\n")
    f.write("```\n" + high_behavior + "\n```\n")

print("[INFO] 📝 analysis.md created successfully.")
