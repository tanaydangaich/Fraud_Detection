# Fraud Detection Example

This repository contains a simple demonstration of exploratory data analysis (EDA) and machine learning on a pair of financial datasets.

## Dataset Origin

The CSV files `loan_applications.csv` and `transactions.csv` are included in this repository for demonstration purposes. They are synthetic datasets describing loan applications and transaction logs for the same set of customers.

## Setup

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

## Running the EDA Script

`eda.py` loads the two CSV files and prints dataset statistics to the console. It also creates several plots that are saved under the `figures/` directory.

```bash
python eda.py
```

After running, expect console output showing shapes, column names and summary statistics, and six PNG files inside `figures/` with distribution and correlation plots.

## Running the Model Script

`model.py` engineers features from both datasets and trains a logistic regression model to predict the `fraud_flag` field. The script prints a scikit‑learn classification report in the terminal.

```bash
python model.py
```

The output will be precision/recall/f1 metrics for the fraud detection model.
