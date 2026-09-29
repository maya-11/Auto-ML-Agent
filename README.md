# AutoML Model Benchmarking App

Upload a CSV, pick the column you want to predict, and the app preprocesses the data, trains several classification models, and ranks them on a leaderboard. The best model can be downloaded as a ready-to-use scikit-learn pipeline.

<!-- Add screenshots: upload + data summary, leaderboard chart, confusion matrix -->

## Tech stack

- **Python**, **Streamlit**
- **scikit-learn** pipelines and models
- **pandas**, **NumPy**, **Plotly**, **joblib**

## Features

- **Data summary:** shape, column types and missing values for the uploaded dataset
- **Automatic preprocessing** with a `ColumnTransformer`: median imputation and scaling for numeric columns, imputation and one-hot encoding for categorical columns
- **Model comparison:** Logistic Regression, Random Forest and Gradient Boosting trained on the same train/test split
- **Leaderboard:** accuracy, precision, recall and F1 per model, with interactive Plotly charts
- **Confusion matrix** and plain-language insights for the best model
- **Export:** download the best full pipeline (preprocessing + model) as a `.pkl`

## Running locally

```bash
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

A sample dataset is included (`professional_dataset.csv`), generated with `generate_dataset.py`.

## Next steps

- Hyperparameter search (e.g. Optuna) and cross-validation
- More model families (kNN, MLP) and regression support
- Serve the exported pipeline behind a FastAPI endpoint
