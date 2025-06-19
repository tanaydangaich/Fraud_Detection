import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    # Load datasets
    loan_df = pd.read_csv('loan_applications.csv')
    transactions_df = pd.read_csv('transactions.csv')

    # Basic info
    print('Loan dataset shape:', loan_df.shape)
    print('Transaction dataset shape:', transactions_df.shape)

    print('\nLoan columns:')
    print(loan_df.columns.tolist())

    print('\nTransaction columns:')
    print(transactions_df.columns.tolist())

    print('\nMissing values in loan dataset:')
    print(loan_df.isnull().sum())

    print('\nMissing values in transactions dataset:')
    print(transactions_df.isnull().sum())

    print('\nSummary statistics for loan numeric columns:')
    print(loan_df.describe())

    print('\nSummary statistics for transaction numeric columns:')
    print(transactions_df.describe())

    print('\nLoan dataset fraud rate:', loan_df['fraud_flag'].mean())
    print('Transactions dataset fraud rate:', transactions_df['fraud_flag'].mean())

    print('\nCorrelation matrix (loan dataset):')
    print(loan_df.corr(numeric_only=True))

    print('\nCorrelation matrix (transactions dataset):')
    print(transactions_df.corr(numeric_only=True))

    # Create output directory for figures
    os.makedirs('figures', exist_ok=True)

    # Loan amount distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(loan_df['loan_amount_requested'], bins=30, kde=True)
    plt.title('Loan Amount Distribution')
    plt.xlabel('Loan Amount Requested')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('figures/loan_amount_distribution.png')
    plt.close()

    # Fraud rate by loan type
    plt.figure(figsize=(10, 6))
    fraud_by_type = loan_df.groupby('loan_type')['fraud_flag'].mean().sort_values(ascending=False)
    sns.barplot(x=fraud_by_type.index, y=fraud_by_type.values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Fraud Rate by Loan Type')
    plt.xlabel('Loan Type')
    plt.ylabel('Fraud Rate')
    plt.tight_layout()
    plt.savefig('figures/loan_fraud_rate_by_type.png')
    plt.close()

    # Correlation heatmap for loan dataset
    plt.figure(figsize=(12, 10))
    loan_corr = loan_df.select_dtypes(include='number').corr()
    sns.heatmap(loan_corr, cmap='coolwarm', annot=False)
    plt.title('Loan Numeric Feature Correlation')
    plt.tight_layout()
    plt.savefig('figures/loan_correlation_heatmap.png')
    plt.close()

    # Transaction amount distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(transactions_df['transaction_amount'], bins=30, kde=True)
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('figures/transaction_amount_distribution.png')
    plt.close()

    # Fraud rate by transaction type
    plt.figure(figsize=(10, 6))
    tx_fraud_by_type = transactions_df.groupby('transaction_type')['fraud_flag'].mean().sort_values(ascending=False)
    sns.barplot(x=tx_fraud_by_type.index, y=tx_fraud_by_type.values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Fraud Rate by Transaction Type')
    plt.xlabel('Transaction Type')
    plt.ylabel('Fraud Rate')
    plt.tight_layout()
    plt.savefig('figures/transaction_fraud_rate_by_type.png')
    plt.close()

    # Correlation heatmap for transactions dataset
    plt.figure(figsize=(12, 10))
    tx_corr = transactions_df.select_dtypes(include='number').corr()
    sns.heatmap(tx_corr, cmap='coolwarm', annot=False)
    plt.title('Transaction Numeric Feature Correlation')
    plt.tight_layout()
    plt.savefig('figures/transaction_correlation_heatmap.png')
    plt.close()


if __name__ == '__main__':
    main()