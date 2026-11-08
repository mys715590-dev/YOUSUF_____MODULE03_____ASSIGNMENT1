from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

# ---------------- CONFIGURATION ----------------

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Churn_Modelling.csv"
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

TARGET = "EstimatedSalary"

NUMERIC = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember"
]

CATEGORICAL = ["Geography", "Gender"]
FEATURES = NUMERIC + CATEGORICAL

ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]


# ---------------- PREPROCESSING ----------------

def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])

    return preprocessor


# ---------------- ANN ARCHITECTURES ----------------

def build_model(input_dim, architecture):

    configurations = {
        "Balanced ANN": [128, 64, 32],
        "Deep ANN": [256, 128, 64, 32, 16],
        "Wide ANN": [256, 256, 128]
    }

    dropout_rates = {
        "Balanced ANN": 0.20,
        "Deep ANN": 0.25,
        "Wide ANN": 0.15
    }

    layers = [
        tf.keras.layers.Input(shape=(input_dim,))
    ]

    for units in configurations[architecture]:

        layers.append(
            tf.keras.layers.Dense(
                units,
                activation="relu"
            )
        )

        layers.append(
            tf.keras.layers.Dropout(
                dropout_rates[architecture]
            )
        )

    # Regression output: one neuron with linear activation
    layers.append(
        tf.keras.layers.Dense(
            1,
            activation="linear"
        )
    )

    model = tf.keras.Sequential(
        layers,
        name=architecture.lower().replace(" ", "_")
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


# ---------------- EVALUATION ----------------

def calculate_metrics(actual, predicted):

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),

        "R2": float(
            r2_score(actual, predicted)
        )
    }


# ---------------- TRAINING ----------------

def train():

    print("\nLoading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place Churn_Modelling.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    print("\nDataset shape:", df.shape)
    print("\nDataset preview:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    missing_columns = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # Remove identifiers and keep only model features
    X = df[FEATURES].copy()

    # Target is continuous EstimatedSalary
    y = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    print("\nInput features:", FEATURES)
    print("Target:", TARGET)

    # Split before fitting preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # Fit preprocessing only on training data
    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    # Save preprocessing artifacts
    joblib.dump(
        preprocessor,
        ART / "preprocessor.joblib"
    )

    joblib.dump(
        FEATURES,
        ART / "feature_order.joblib"
    )

    # ANN architectures
    architectures = {
        "Balanced ANN": "salary_ann_balanced.keras",
        "Deep ANN": "salary_ann_deep.keras",
        "Wide ANN": "salary_ann_wide.keras"
    }

    results = {}

    for architecture, filename in architectures.items():

        print("\n" + "=" * 55)
        print("Training:", architecture)
        print("=" * 55)

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        model = build_model(
            X_train_processed.shape[1],
            architecture
        )

        callbacks = [

            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=6,
                min_lr=0.000001
            )
        ]

        history = model.fit(
            X_train_processed,
            y_train.to_numpy(),
            validation_split=0.20,
            epochs=180,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        predictions = model.predict(
            X_test_processed,
            verbose=0
        ).reshape(-1)

        model_metrics = calculate_metrics(
            y_test,
            predictions
        )

        model_metrics["epochs_trained"] = len(
            history.history["loss"]
        )

        results[architecture] = model_metrics

        # Save trained model
        model.save(ART / filename)

        # Save training history
        history_name = architecture.split()[0].lower()

        pd.DataFrame(
            history.history
        ).to_csv(
            ART / f"history_{history_name}.csv",
            index=False
        )

        print("\nResults:", architecture)
        print(model_metrics)

    # Median baseline for honest comparison
    baseline = DummyRegressor(
        strategy="median"
    )

    baseline.fit(
        X_train_processed,
        y_train
    )

    baseline_predictions = baseline.predict(
        X_test_processed
    )

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions
    )

    # Best architecture according to test MAE
    best_architecture = min(
        results,
        key=lambda name: results[name]["MAE"]
    )

    report = {
        "target": TARGET,
        "features": FEATURES,
        "excluded_columns": ID_COLUMNS + ["Exited"],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "models": results,
        "median_baseline": baseline_metrics,
        "best_architecture_by_MAE": best_architecture,
        "note": (
            "Educational exercise only. "
            "Not a production salary-estimation claim."
        )
    }

    # Save metrics
    with open(
        ART / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # Save comparison table
    comparison = pd.DataFrame([
        {
            "Architecture": name,
            **values
        }
        for name, values in results.items()
    ])

    comparison.to_csv(
        ART / "architecture_comparison.csv",
        index=False
    )

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)

    print("\nArchitecture comparison:")
    print(comparison.to_string(index=False))

    print("\nMedian baseline:")
    print(baseline_metrics)

    print("\nBest architecture by test MAE:")
    print(best_architecture)

    print("\nAll models and preprocessing files saved in artifacts/.")


if __name__ == "__main__":
    train()