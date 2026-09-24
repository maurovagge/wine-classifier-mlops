# Wine Classifier MLOps

## Overview

Wine Classifier MLOps is a Python machine-learning project that trains and serves a multiclass wine classifier. It uses the built-in scikit-learn Wine dataset, prepares the data with pandas, selects a Random Forest model through cross-validated hyperparameter search, evaluates the resulting estimator, and persists it with `joblib`.

The project also includes a Streamlit web application for interactive inference. Users can provide wine chemical measurements through sliders and receive the predicted wine class from the trained model.

## Technology Stack

- **Python** — application and pipeline implementation.
- **pandas** — tabular data representation, feature/target separation, CSV persistence, and inference input construction.
- **scikit-learn** — Wine dataset, train/test splitting, Random Forest classification, grid search, cross-validation, and evaluation metrics.
- **PyYAML** — loading runtime and pipeline configuration from `config.yaml`.
- **joblib** — serialization and loading of the trained estimator.
- **Streamlit** — interactive browser-based inference interface.

## Project Structure

```text
.
├── app.py                      # Streamlit inference application
├── main.py                     # Training and evaluation pipeline entry point
├── config.yaml                 # Paths and pipeline parameters
├── src/
│   ├── config_loader.py        # YAML configuration loading
│   ├── data_preprocessing.py   # Dataset creation and train/test splitting
│   ├── model_training.py       # Random Forest and GridSearchCV training
│   ├── evaluation.py           # Accuracy and classification report
│   └── model_io.py             # Model serialization and loading
├── data/
│   ├── raw/                    # Generated raw dataset CSV
│   └── processed/              # Generated processed training data
└── model/                      # Generated serialized model
```

The `data/` and `model/` outputs are generated at runtime and are not intended to be committed as source files.

## Configuration

The default configuration is stored in `config.yaml`:

| Setting | Purpose | Default |
|---|---|---|
| `raw_data_path` | Location of the generated raw dataset | `data/raw/raw_data.csv` |
| `processed_data_path` | Location of the generated training data | `data/processed/processed_data.csv` |
| `saved_model_path` | Location of the serialized estimator | `model/random_forest.joblib` |
| `max_depth` | Reserved model configuration value | `5` |
| `n_estimators` | Reserved model configuration value | `100` |
| `test_size` | Fraction reserved for the test split | `0.2` |
| `random_state` | Reproducibility seed | `42` |

The training implementation currently performs its own grid search over `n_estimators` values of `50`, `100`, and `200`, and `max_depth` values of `3`, `5`, `10`, and `None`. The corresponding configuration fields are loaded but are not currently used to build that grid.

## Training and Evaluation Flow

Run `main.py` to execute the complete pipeline:

1. `config_loader.py` reads `config.yaml`.
2. `data_preprocessing.py` loads `sklearn.datasets.load_wine`.
3. The dataset is converted into a pandas `DataFrame` with chemical feature columns and a `target` column.
4. The raw data is written to `data/raw/raw_data.csv`.
5. Features and targets are separated, then split into training and test sets using `train_test_split`.
6. The training features are written to `data/processed/processed_data.csv`.
7. `model_training.py` creates a `RandomForestClassifier` and evaluates 12 hyperparameter combinations with five-fold `GridSearchCV`.
8. The best estimator is evaluated against the held-out test set.
9. `evaluation.py` prints accuracy and a detailed classification report.
10. `model_io.py` serializes the selected estimator to `model/random_forest.joblib`.

The pipeline uses all available CPU cores for the grid search through `n_jobs=-1`.

## Streamlit Inference Application

The web application is implemented in `app.py` and uses the same configuration and model persistence layer as the training pipeline.

At startup, the application:

1. Loads `config.yaml`.
2. Loads the serialized estimator from `model/random_forest.joblib`.
3. Loads the raw CSV dataset to derive feature names and slider ranges.
4. Creates one slider per chemical feature, using the observed feature minimum, maximum, and mean as the default value.
5. Builds a one-row pandas `DataFrame` from the selected values.
6. Calls `model.predict()` and maps the predicted class index to `Class_0`, `Class_1`, or `Class_2`.

The model and dataset are cached with Streamlit's `st.cache_resource`. The training pipeline must be executed successfully before starting the application so that both the raw dataset and serialized model exist.

## Setup

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the runtime dependencies:

```bash
python -m pip install pandas scikit-learn PyYAML joblib streamlit
```

## Running the Project

From the repository root, run the training and evaluation pipeline:

```bash
python main.py
```

After the model and data artifacts have been generated, start the Streamlit application:

```bash
streamlit run app.py
```

The command prints a local URL that can be opened in a browser. To retrain the model, run `python main.py` again before restarting or refreshing the application.

## Generated Artifacts

The pipeline creates the following runtime files:

```text
data/raw/raw_data.csv
data/processed/processed_data.csv
model/random_forest.joblib
```

These files are reproducible outputs derived from the configured dataset source and training parameters. They are excluded from version control by the repository's `.gitignore`.
