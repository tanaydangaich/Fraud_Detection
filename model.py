import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def load_datasets():
    """Load loan applications and transaction logs."""
    loan_df = pd.read_csv("loan_applications.csv")
    txn_df = pd.read_csv("transactions.csv")
    return loan_df, txn_df


def aggregate_transactions(txn_df):
    """Aggregate transaction information per customer."""
    agg = txn_df.groupby("customer_id").agg(
        txn_count=("transaction_id", "count"),
        txn_total_amount=("transaction_amount", "sum"),
        txn_avg_amount=("transaction_amount", "mean"),
        intl_txn_rate=("is_international_transaction", "mean"),
    )
    return agg.reset_index()


def engineer_features(loan_df, txn_df):
    """Create additional features for modeling."""
    # Ratio of requested amount to monthly income
    loan_df["loan_amount_ratio"] = loan_df["loan_amount_requested"] / loan_df["monthly_income"]
    # Flag unemployment
    loan_df["is_unemployed"] = loan_df["employment_status"].eq("Unemployed").astype(int)

    # Merge aggregated transaction statistics
    txn_agg = aggregate_transactions(txn_df)
    loan_df = loan_df.merge(txn_agg, on="customer_id", how="left")

    # Fill NaNs for customers without transactions
    loan_df[["txn_count", "txn_total_amount", "txn_avg_amount", "intl_txn_rate"]] = \
        loan_df[["txn_count", "txn_total_amount", "txn_avg_amount", "intl_txn_rate"]].fillna(0)

    return loan_df


def build_pipeline(numeric_features, categorical_features):
    """Create preprocessing and modeling pipeline."""
    preprocess = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    clf = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    return clf


def train_and_evaluate(df):
    """Split the data, train a logistic regression model, and print metrics."""
    target = "fraud_flag"

    numeric_features = [
        "loan_amount_requested",
        "loan_tenure_months",
        "interest_rate_offered",
        "monthly_income",
        "cibil_score",
        "existing_emis_monthly",
        "debt_to_income_ratio",
        "applicant_age",
        "number_of_dependents",
        "loan_amount_ratio",
        "txn_count",
        "txn_total_amount",
        "txn_avg_amount",
        "intl_txn_rate",
    ]

    categorical_features = [
        "loan_type",
        "purpose_of_loan",
        "employment_status",
        "gender",
        "property_ownership_status",
        "loan_status",
    ]

    X = df[numeric_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    loan_df, txn_df = load_datasets()
    data = engineer_features(loan_df, txn_df)
    train_and_evaluate(data)