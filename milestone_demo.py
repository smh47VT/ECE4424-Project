"""
Stress Level Predictor from Sleep Data
========================================
A linear regression model
Uses gradient descent to learn weights that predict stress level from:
  - Sleep Duration (hours)
  - Quality of Sleep  (1-10)

Data: Sleep_health_and_lifestyle_dataset.csv
"""

import numpy as np
import pandas as pd
import sys
import os

# ----------------------------------------------
# 1.  DATA LOADING & PREPROCESSING
# ----------------------------------------------

def load_data(filepath: str):
    """Load CSV and extract the two features + target."""
    df = pd.read_csv(filepath)

    # Select columns by name for robustness
    sleep_duration = df["Sleep Duration"].values.astype(float)    # feature 1
    sleep_quality  = df["Quality of Sleep"].values.astype(float)  # feature 2
    stress_level   = df["Stress Level"].values.astype(float)      # target

    X = np.column_stack([sleep_duration, sleep_quality])   # (n, 2)
    y = stress_level                                        # (n,)
    return X, y, df


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Randomly split data into train / test sets."""
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """Z-score normalization (mean=0, std=1) -- built from scratch."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        # guard against zero-std columns
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----------------------------------------------
# 2.  MODEL -- Linear Regression via Gradient Descent
# ----------------------------------------------

class LinearRegressionGD:
    """
    Multivariate linear regression trained with mini-batch
    gradient descent.  Everything from scratch -- no sklearn.

    Model:  y_hat = X @ w + b
    Loss :  MSE   = (1/n) * sum( (y_hat - y)^2 )
    """

    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32, seed=42):
        self.lr         = learning_rate
        self.epochs     = epochs
        self.batch_size = batch_size
        self.seed       = seed
        self.weights    = None
        self.bias       = None
        self.loss_history = []

    # ---- core math ------------------------------------------------

    @staticmethod
    def _mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def _predict(self, X):
        return X @ self.weights + self.bias

    # ---- training -------------------------------------------------

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape

        # Xavier-like initialization
        self.weights = rng.normal(0, np.sqrt(2 / n_features), size=n_features)
        self.bias    = 0.0
        self.loss_history = []

        for epoch in range(1, self.epochs + 1):
            # shuffle every epoch
            idx = rng.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]

            for start in range(0, n_samples, self.batch_size):
                end   = min(start + self.batch_size, n_samples)
                X_b   = X_shuf[start:end]
                y_b   = y_shuf[start:end]
                m     = len(y_b)

                # forward
                y_hat = self._predict(X_b)

                # gradients
                error = y_hat - y_b                        # (m,)
                dw    = (2 / m) * (X_b.T @ error)          # (n_features,)
                db    = (2 / m) * np.sum(error)             # scalar

                # parameter update
                self.weights -= self.lr * dw
                self.bias    -= self.lr * db

            # epoch loss (full training set)
            train_pred = self._predict(X)
            loss = self._mse(y, train_pred)
            self.loss_history.append(loss)

            if epoch % 200 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>5d}/{self.epochs}  --  MSE: {loss:.4f}")

        return self

    # ---- inference ------------------------------------------------

    def predict(self, X):
        return self._predict(X)


# ----------------------------------------------
# 3.  EVALUATION METRICS  (from scratch)
# ----------------------------------------------

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ----------------------------------------------
# 4.  MAIN PIPELINE
# ----------------------------------------------

def main():
    # --- locate file ---
    filepath = "Sleep_health_and_lifestyle_dataset.csv"
    if not os.path.exists(filepath):
        alt = os.path.join("/mnt/user-data/uploads", filepath)
        if os.path.exists(alt):
            filepath = alt
        else:
            print(f"ERROR: Cannot find '{filepath}'. Place it in the working directory.")
            sys.exit(1)

    print("=" * 60)
    print("  STRESS LEVEL PREDICTOR -- Linear Regression")
    print("=" * 60)

    # --- load ---
    X, y, df = load_data(filepath)
    print(f"\nDataset loaded: {len(y)} samples")
    print(f"  Features : Sleep Duration (hrs), Quality of Sleep (1-10)")
    print(f"  Target   : Stress Level")
    print(f"  Stress range: {y.min():.0f} - {y.max():.0f}")

    # --- split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
    print(f"\nTrain / Test split: {len(y_train)} / {len(y_test)}")

    # --- scale ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # --- train ---
    print("\n-- Training --")
    model = LinearRegressionGD(
        learning_rate = 0.01,
        epochs        = 2000,
        batch_size    = 32,
        seed          = 42,
    )
    model.fit(X_train_s, y_train)

    # --- evaluate ---
    y_pred_train = model.predict(X_train_s)
    y_pred_test  = model.predict(X_test_s)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<22} {'Train':>10} {'Test':>10}")
    print("-" * 44)
    print(f"{'MAE':<22} {mean_absolute_error(y_train, y_pred_train):>10.4f} {mean_absolute_error(y_test, y_pred_test):>10.4f}")
    print(f"{'RMSE':<22} {root_mean_squared_error(y_train, y_pred_train):>10.4f} {root_mean_squared_error(y_test, y_pred_test):>10.4f}")
    print(f"{'R^2 score':<22} {r_squared(y_train, y_pred_train):>10.4f} {r_squared(y_test, y_pred_test):>10.4f}")

    # --- learned parameters (unscaled for interpretability) ---
    w_orig = model.weights / scaler.std_
    b_orig = model.bias - np.sum(model.weights * scaler.mean_ / scaler.std_)
    print(f"\nLearned equation (original scale):")
    print(f"  Stress ~ {w_orig[0]:+.4f} * SleepDuration "
          f"{w_orig[1]:+.4f} * SleepQuality "
          f"{b_orig:+.4f}")

    # --- sample predictions ---
    print(f"\n{'Sleep Dur':>10} {'Quality':>10} {'Predicted':>12} {'Actual':>10}")
    print("-" * 45)
    rng = np.random.default_rng(99)
    sample_idx = rng.choice(len(y_test), size=min(10, len(y_test)), replace=False)
    for i in sample_idx:
        print(f"{X_test[i, 0]:>10.1f} {X_test[i, 1]:>10.0f}"
              f" {y_pred_test[i]:>12.2f} {y_test[i]:>10.0f}")

    # --- interactive prediction ---
    print("\n" + "=" * 60)
    print("  INTERACTIVE PREDICTOR")
    print("=" * 60)
    print("Enter sleep data to predict stress level (or 'q' to quit):\n")
    while True:
        try:
            dur_input = input("  Sleep Duration (hours) : ").strip()
            if dur_input.lower() == 'q':
                break
            qual_input = input("  Sleep Quality  (1-10)  : ").strip()
            if qual_input.lower() == 'q':
                break

            dur  = float(dur_input)
            qual = float(qual_input)
            x_new   = np.array([[dur, qual]])
            x_new_s = scaler.transform(x_new)
            pred    = model.predict(x_new_s)[0]
            print(f"  -> Predicted Stress Level: {pred:.2f}\n")
        except (ValueError, EOFError):
            break

    print("\nDone.")

# ----------------------------------------------
# 5.  HEALTH DATA LOADING
# ----------------------------------------------

def load_health_data(filepath: str):
    """
    Loads the same CSV file but uses more inputs and outputs.
    """

    df = pd.read_csv(filepath)

    # -------------------------------
    # INPUT 1: Sleep Duration
    # -------------------------------
    sleep_duration = df["Sleep Duration"].values.astype(float)

    # -------------------------------
    # INPUT 2: Quality of Sleep
    # -------------------------------
    sleep_quality = df["Quality of Sleep"].values.astype(float)

    # -------------------------------
    # INPUT 3: Age
    # -------------------------------
    age = df["Age"].values.astype(float)

    # -------------------------------
    # INPUT 4: Gender
    # -------------------------------
    gender = df["Gender"].map(
        {
            "Male": 0,
            "Female": 1
        }
    ).values.astype(float)

    X = np.column_stack(
        [
            sleep_duration,
            sleep_quality,
            age,
            gender
        ]
    )

    return X, df

def prepare_health_outputs(df):
    """
    Prepares the health outputs that the program will predict.
    """

    # -------------------------------
    # OUTPUT 1: BMI Category
    # -------------------------------
    bmi_categories = sorted(df["BMI Category"].unique())

    bmi_to_number = {}

    for i in range(len(bmi_categories)):
        bmi_to_number[bmi_categories[i]] = i

    bmi_category = df["BMI Category"].map(bmi_to_number).values.astype(float)

    # -------------------------------
    # OUTPUT 2: Blood Pressure
    # -------------------------------
    bp_parts = df["Blood Pressure"].str.split("/", expand=True)

    systolic_bp = bp_parts[0].values.astype(float)
    diastolic_bp = bp_parts[1].values.astype(float)

    # -------------------------------
    # OUTPUT 3: Resting Heart Rate
    # -------------------------------
    heart_rate = df["Heart Rate"].values.astype(float)

    # -------------------------------
    # OUTPUT 4: Daily Steps
    # -------------------------------
    daily_steps = df["Daily Steps"].values.astype(float)

    # -------------------------------
    # OUTPUT 5: Stress Level
    # -------------------------------
    stress_level = df["Stress Level"].values.astype(float)

    # -------------------------------
    # OUTPUT 6: Physical Activity Level
    # -------------------------------
    activity_level = df["Physical Activity Level"].values.astype(float)

    y = np.column_stack(
        [
            bmi_category,
            systolic_bp,
            diastolic_bp,
            heart_rate,
            daily_steps,
            stress_level,
            activity_level
        ]
    )

    return y, bmi_categories

def run_health_prediction(filepath: str):
    """
    Runs the health prediction part of the project.
    """

    print("\n" + "=" * 60)
    print("  HEALTH PREDICTION FROM SLEEP DATA")
    print("=" * 60)

    X, df = load_health_data(filepath)
    y, bmi_categories = prepare_health_outputs(df)

    print(f"\nDataset loaded: {len(y)} samples")

    print("\nInputs:")
    print("  1. Sleep Duration")
    print("  2. Quality of Sleep")
    print("  3. Age")
    print("  4. Gender")

    print("\nPredictions:")
    print("  1. BMI Category")
    print("  2. Blood Pressure")
    print("  3. Resting Heart Rate")
    print("  4. Daily Steps")
    print("  5. Stress Level")
    print("  6. Physical Activity Level")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_ratio=0.2,
        seed=42
    )

    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\n-- Training --")

    model = LinearRegressionGD(
        learning_rate=0.01,
        epochs=2000,
        batch_size=32,
        seed=42
    )

    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    output_names = [
        "BMI Category",
        "Systolic Blood Pressure",
        "Diastolic Blood Pressure",
        "Resting Heart Rate",
        "Daily Steps",
        "Stress Level",
        "Physical Activity Level"
    ]

    for i in range(len(output_names)):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = root_mean_squared_error(y_test[:, i], y_pred[:, i])

        print(f"\n{output_names[i]}")
        print(f"  MAE : {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")

    print("\n" + "=" * 60)
    print("  SAMPLE PREDICTIONS")
    print("=" * 60)

    for i in range(min(10, len(y_test))):
        predicted_bmi_number = int(round(y_pred[i, 0]))

        if predicted_bmi_number < 0:
            predicted_bmi_number = 0

        if predicted_bmi_number >= len(bmi_categories):
            predicted_bmi_number = len(bmi_categories) - 1

        actual_bmi_number = int(y_test[i, 0])

        print(f"\nPerson {i + 1}")
        print(f"  Actual BMI Category    : {bmi_categories[actual_bmi_number]}")
        print(f"  Predicted BMI Category : {bmi_categories[predicted_bmi_number]}")
        print(f"  Predicted BP           : {y_pred[i, 1]:.0f}/{y_pred[i, 2]:.0f}")
        print(f"  Predicted Heart Rate   : {y_pred[i, 3]:.0f}")
        print(f"  Predicted Daily Steps  : {y_pred[i, 4]:.0f}")
        print(f"  Predicted Stress Level : {y_pred[i, 5]:.1f}")
        print(f"  Predicted Activity Lvl : {y_pred[i, 6]:.1f}")

if __name__ == "__main__":
    main()