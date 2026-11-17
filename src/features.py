"""Feature engineering for Kaggle Playground S6E11 - Predicting Credit Card Fraud."""
import numpy as np
import pandas as pd

TARGET = "is_fraud"
ID = "id"

CAT_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]


ORIGINAL = "orig/credit-card-fraud-detection-2026/credit_card_fraud_2026.csv"


# Numeric columns whose exact values are target-encoded (treated as categories).
TE_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
    "card_age_months",
    "customer_age",
]


def target_encode(X_fit, y_fit, X_apply_list, cols=TE_COLS, smoothing=20, n_splits=5, seed=1):
    """Smoothed target encoding without leakage.

    Rows in X_fit get out-of-fold encodings (inner K-fold), rows in each X_apply
    get encodings learned on all of X_fit. Returns new frames with `<col>_te` columns.
    """
    from sklearn.model_selection import StratifiedKFold

    X_fit = X_fit.copy()
    X_apply_list = [x.copy() for x in X_apply_list]
    prior = y_fit.mean()

    def mapping(keys, target):
        s = pd.DataFrame({"k": keys, "t": target}).groupby("k")["t"].agg(["sum", "count"])
        return (s["sum"] + prior * smoothing) / (s["count"] + smoothing)

    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for c in cols:
        keys = X_fit[c].values
        enc = np.empty(len(X_fit))
        for p, q in inner.split(keys, y_fit):
            enc[q] = pd.Series(keys[q]).map(mapping(keys[p], y_fit[p])).fillna(prior).values
        X_fit[f"{c}_te"] = enc
        full_map = mapping(keys, y_fit)
        for x in X_apply_list:
            x[f"{c}_te"] = x[c].map(full_map).fillna(prior).values
    return X_fit, X_apply_list


def load(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Competition train/test plus the original dataset the synthetic data was generated from."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    orig = pd.read_csv(f"{data_dir}/{ORIGINAL}").rename(columns={"transaction_id": ID})
    orig = orig[train.columns]
    return train, test, orig


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven features on top of the raw columns."""
    df = df.copy()
    for c in BOOL_COLS:
        df[c] = df[c].astype(str).str.lower().eq("true").astype(np.int8)

    # Spending relative to what the customer can afford.
    df["log_amount"] = np.log1p(df["amount_usd"])
    df["amount_to_balance"] = df["amount_usd"] / (df["account_balance_usd"] + 1)
    df["log_balance"] = np.log1p(df["account_balance_usd"])

    # Activity bursts: many transactions in a short window.
    df["txn_rate"] = df["txn_count_last_24h"] / (df["hours_since_last_txn"] + 0.1)
    df["velocity_x_count"] = df["velocity_score"] * df["txn_count_last_24h"]

    # Count of red flags on the transaction.
    df["risk_flag_count"] = df[BOOL_COLS].sum(axis=1) + (df["cvv_retry_count"] > 0) + (df["prior_disputes"] > 0)

    # Time patterns.
    df["is_night"] = df["time_of_day_hour"].isin([0, 1, 2, 3, 4, 5]).astype(np.int8)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["time_of_day_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time_of_day_hour"] / 24)

    # Interaction of weak authentication with a risky merchant.
    df["no_auth"] = (df["auth_method"] == "No Authentication").astype(np.int8)
    df["no_auth_x_merchant_risk"] = df["no_auth"] * df["merchant_risk_score"]
    df["distance_x_foreign"] = df["distance_from_home_km"] * (1 + df["is_foreign_transaction"])

    # Frequency encoding: how common each exact value is across train + test.
    for c in TE_COLS:
        df[f"{c}_cnt"] = df[c].map(df[c].value_counts())

    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df


def build(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame):
    """Engineer features jointly so category levels and frequencies match."""
    n_tr, n_te = len(train), len(test)
    full = pd.concat([train.drop(columns=[TARGET]), test, orig.drop(columns=[TARGET])], ignore_index=True)
    full = add_features(full).drop(columns=[ID])
    X = full.iloc[:n_tr].reset_index(drop=True)
    X_test = full.iloc[n_tr:n_tr + n_te].reset_index(drop=True)
    X_orig = full.iloc[n_tr + n_te:].reset_index(drop=True)
    y = train[TARGET].astype(int).values
    y_orig = orig[TARGET].astype(int).values
    return X, y, X_test, X_orig, y_orig
