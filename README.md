# AutoML Pipeline System

Config-driven end-to-end AutoML pipeline system for training, evaluating, and managing machine learning workflows.

---

## Tech Stack

- Python
- pandas
- scikit-learn
- PyYAML

Future integrations:
- XGBoost
- LightGBM
- MLflow
- Optuna
- FastAPI
- SHAP

---

## Project Structure

```text
automl_pipeline/
├── src/
├── configs/
├── data/
├── models/
├── mlruns/
├── notebooks/
├── tests/
├── main.py
├── requirements.txt
└── README.md

```

## Setup

Create conda environment:

```bash
conda create -n automl-pipeline python=3.11 -y
conda activate automl-pipeline
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
python main.py
```

---

## Features

- YAML config-driven workflow
- Dataset loading abstraction
- Feature and target separation
- Reusable train/test splitting
- Reproducible ML workflow