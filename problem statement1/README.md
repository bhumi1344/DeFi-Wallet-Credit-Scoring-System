# DeFi Wallet Credit Scoring System

## Overview

This project assigns credit scores (ranging from 0 to 1000) to crypto wallets based on their transaction behavior on Aave V2. The purpose is to estimate creditworthiness in a decentralized finance (DeFi) context using explainable, data-driven techniques.

## Project Flow

1. Input: A JSON file containing wallet-level transaction features (deposits, borrows, ratios, active days, etc.)
2. Model: A pre-trained XGBoost regression model that generates raw scores
3. Post-processing: Normalization of raw scores to a 0–1000 range
4. Output: A CSV file with wallet addresses and their corresponding credit scores
5. Analysis: A markdown report summarizing the score distribution and wallet behavior

## Architecture

Input JSON --> Feature Extraction --> Load Trained Model --> Predict Scores -->
Normalize Scores --> Output CSV + Generate Analysis Report

## Features Used

The model uses a subset of the following features:

- total_deposit: Total deposited value
- total_borrow: Total borrowed value
- deposit_to_borrow_ratio: Ratio of deposits to borrows
- tx_count: Number of transactions
- active_days: Number of days wallet was active
- avg_deposit_size: Average value of deposit transactions
- avg_borrow_size: Average value of borrow transactions
- repayment_rate: Fraction of borrowed amount that has been repaid

## Model Training

The model (model.pkl) is an XGBoost regression model trained on historical transaction data of wallets, where credit labels (or score-like values) were derived using behavioral heuristics. The training data is not included in this repository.

## Usage

To generate scores:

```bash
python score_wallets.py --input user-wallet-transactions.json --output wallet_scores.csv
```

This will generate:

- wallet_scores.csv: Contains wallet addresses and their credit scores
- analysis.md: Contains score distribution summary and basic insights

## Score Interpretation

The normalized credit score falls between 0 and 1000 and can be interpreted as:

- 900–1000: Excellent – High deposits, low risk
- 700–899: Good – Stable, consistent usage
- 500–699: Average – Moderate risk profile
- 300–499: Risky – Active borrowing, lower deposit behavior
- 0–299: High Risk – Irregular behavior, high borrowing, or inactivity

## Dependencies

- Python 3.x
- xgboost
- pandas
- scikit-learn
- matplotlib (optional for visualization)

Install with:

```bash
pip install xgboost pandas scikit-learn matplotlib
```

## File Structure

.
├── model.pkl                     # Trained credit scoring model
├── user-wallet-transactions.json  # Input file (wallet features)
├── score_wallets.py             # Main scoring script
├── wallet_scores.csv            # Output scores
├── analysis.md                  # Score distribution analysis
├── README.md                    # Project documentation

## Extensibility

This system can be improved by:

- Adding new behavioral features
- Including time-based trends
- Expanding to support multiple DeFi protocols
- Building a dashboard or API for real-time scoring

## Transparency

- The scoring logic is based on explainable, interpretable features
- Scores are normalized for consistency
- The analysis file gives clear insights into score distribution and logic