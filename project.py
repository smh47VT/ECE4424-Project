"""
Stress Level Predictor from Sleep Data
========================================
A linear regression model
Uses gradient descent to learn weights that predict
lifestyle indicators

Data: Sleep_health_and_lifestyle_dataset.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

# ----------------------------------------------
#  OUTPUT DIRECTORY
# ----------------------------------------------

OUTPUT_DIR = "output"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------
#  DATA LOADING & PREPROCESSING
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
    """Z-score normalization (mean=0, std=1)."""

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
#  MODEL -- Linear Regression via Gradient Descent
# ----------------------------------------------

class LinearRegressionGD:
    """
    Multivariate linear regression trained with mini-batch
    gradient descent.

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
#  EVALUATION METRICS
# ----------------------------------------------

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot

def mean_absolute_pct_error(y_true, y_pred):
    """MAPE -- skips zeros in y_true to avoid division by zero."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


# ----------------------------------------------
#  MAIN PIPELINE
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

    print("\nDone.")

# ----------------------------------------------
#  HEALTH DATA LOADING
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
    gender_raw = df["Gender"].values
    gender = np.zeros(len(gender_raw))

    for i in range(len(gender_raw)):
        if gender_raw[i].lower() == "female":
            gender[i] = 1.0
        else:
            gender[i] = 0.0

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

# ----------------------------------------------
#  LINEAR REGRESSION FOR MULTIPLE OUTPUTS
# ----------------------------------------------

class MultiOutputLinearRegressionGD:
    """
    This is almost the same idea as the first linear regression class,
    but this one can predict more than one output at the same time.
    """

    def __init__(self, learning_rate=0.01, epochs=2000, seed=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights = None
        self.bias = None
        self.loss_history = []

    def predict(self, X):
        return X @ self.weights + self.bias

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_outputs = y.shape[1]

        self.weights = rng.normal(0, 0.1, size=(n_features, n_outputs))
        self.bias = np.zeros(n_outputs)

        for epoch in range(1, self.epochs + 1):
            y_hat = self.predict(X)

            error = y_hat - y

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error, axis=0)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            if epoch == 1 or epoch % 500 == 0:
                print(f"  Epoch {epoch:>5d}/{self.epochs}  --  MSE: {loss:.4f}")

        return self


# ----------------------------------------------
#  VISUALIZATION FUNCTIONS
# ----------------------------------------------

def plot_r2_comparison(output_names, train_r2, test_r2):
    """Bar chart comparing R^2 for each target."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(output_names))
    w = 0.35

    bars_tr = ax.bar(x - w/2, train_r2, w, label="Train", color="#4C72B0")
    bars_te = ax.bar(x + w/2, test_r2,  w, label="Test",  color="#DD8452")

    ax.set_ylabel("R-squared")
    ax.set_title("R-squared by Target (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(output_names, rotation=30, ha="right")
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(min(min(train_r2), min(test_r2), -0.1) - 0.05, 1.05)

    for bar in bars_tr:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars_te:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_r2_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mae_comparison(output_names, train_mae, test_mae):
    """Bar chart comparing MAE for each target."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(output_names))
    w = 0.35

    ax.bar(x - w/2, train_mae, w, label="Train", color="#4C72B0")
    ax.bar(x + w/2, test_mae,  w, label="Test",  color="#DD8452")

    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("MAE by Target (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(output_names, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_mae_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_predicted_vs_actual(output_names, y_test, y_pred):
    """Scatter plot of predicted vs actual for each target."""
    n_targets = len(output_names)
    cols = 4
    rows = (n_targets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = axes.flatten()

    for j, name in enumerate(output_names):
        ax = axes[j]
        actual = y_test[:, j]
        predicted = y_pred[:, j]

        ax.scatter(actual, predicted, alpha=0.6, s=30, color="#4C72B0",
                   edgecolors="white", linewidth=0.5, zorder=3)

        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        margin = (hi - lo) * 0.08
        axis_lo = lo - margin
        axis_hi = hi + margin
        ax.set_xlim(axis_lo, axis_hi)
        ax.set_ylim(axis_lo, axis_hi)
        ax.set_aspect("equal", adjustable="box")

        # y = x perfect prediction line
        ax.plot([axis_lo, axis_hi], [axis_lo, axis_hi],
                "r--", linewidth=1, label="Perfect (y=x)", zorder=2)

        # best-fit line
        if len(actual) > 1 and np.std(actual) > 0:
            slope = (np.sum((actual - actual.mean()) * (predicted - predicted.mean()))
                     / np.sum((actual - actual.mean()) ** 2))
            intercept = predicted.mean() - slope * actual.mean()
            fit_x = np.array([axis_lo, axis_hi])
            fit_y = slope * fit_x + intercept
            ax.plot(fit_x, fit_y, "-", color="#2ca02c", linewidth=1.5,
                    label=f"Best fit (slope={slope:.2f})", zorder=2)

        r2 = r_squared(actual, predicted)
        ax.text(0.05, 0.95, f"R^2 = {r2:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(name)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.2)

    for k in range(n_targets, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle("Predicted vs Actual (Test Set)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_predicted_vs_actual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_residual_distributions(output_names, y_test, y_pred):
    """Histogram of residuals for each target."""
    n_targets = len(output_names)
    cols = 4
    rows = (n_targets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for j, name in enumerate(output_names):
        ax = axes[j]
        residuals = y_pred[:, j] - y_test[:, j]
        ax.hist(residuals, bins=20, color="#4C72B0", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (Predicted - Actual)")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.text(0.95, 0.95, f"mean={residuals.mean():.2f}\nstd={residuals.std():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    for k in range(n_targets, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle("Residual Distributions (Test Set)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_residual_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_loss(loss_history):
    """Training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, color="#4C72B0", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Multi-Output Regression Training Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_training_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(weights, feat_names, output_names):
    """Heatmap of learned weight magnitudes (standardized coefficients)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    W = weights.copy()

    im = ax.imshow(np.abs(W).T, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Absolute Weight")

    ax.set_xticks(range(len(feat_names)))
    ax.set_yticks(range(len(output_names)))
    ax.set_xticklabels(feat_names)
    ax.set_yticklabels(output_names)
    ax.set_title("Feature Influence on Each Target (Standardized Weight Magnitude)")

    for i in range(len(feat_names)):
        for j in range(len(output_names)):
            val = W[i, j]
            color = "white" if np.abs(val) > np.abs(W).max() * 0.6 else "black"
            ax.text(i, j, f"{val:+.2f}", ha="center", va="center",
                    color=color, fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ----------------------------------------------
#  CSV STATISTICS EXPORT
# ----------------------------------------------

def export_statistics_csv(output_names, train_stats, test_stats):
    """Write a CSV with per-target train and test metrics."""
    rows = []

    for j, name in enumerate(output_names):
        rows.append({
            "Target":         name,
            "Train MAE":      round(train_stats["mae"][j], 4),
            "Train RMSE":     round(train_stats["rmse"][j], 4),
            "Train R^2":      round(train_stats["r2"][j], 4),
            "Train MAPE (%)": round(train_stats["mape"][j], 2),
            "Test MAE":       round(test_stats["mae"][j], 4),
            "Test RMSE":      round(test_stats["rmse"][j], 4),
            "Test R^2":       round(test_stats["r2"][j], 4),
            "Test MAPE (%)":  round(test_stats["mape"][j], 2),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "model_statistics.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ----------------------------------------------
#  HEALTH PREDICTION PIPELINE (with stats + plots)
# ----------------------------------------------

def run_health_prediction(filepath: str):
    """
    Runs the health prediction part of the project.
    Now includes per-target statistics, CSV export, and visualizations.
    """

    ensure_output_dir()

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

    output_names = [
        "BMI Category",
        "Systolic BP",
        "Diastolic BP",
        "Heart Rate",
        "Daily Steps",
        "Stress Level",
        "Physical Activity"
    ]

    print("\nPredictions:")
    for i, name in enumerate(output_names):
        print(f"  {i + 1}. {name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_ratio=0.2, seed=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\n-- Training --")

    model = MultiOutputLinearRegressionGD(
        learning_rate=0.01,
        epochs=2000,
        seed=42
    )

    model.fit(X_train_s, y_train)

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    # ------------------------------------------
    #  Collect per-target statistics
    # ------------------------------------------

    train_stats = {"mae": [], "rmse": [], "r2": [], "mape": []}
    test_stats  = {"mae": [], "rmse": [], "r2": [], "mape": []}

    for i in range(len(output_names)):
        train_stats["mae"].append(mean_absolute_error(y_train[:, i], y_pred_train[:, i]))
        train_stats["rmse"].append(root_mean_squared_error(y_train[:, i], y_pred_train[:, i]))
        train_stats["r2"].append(r_squared(y_train[:, i], y_pred_train[:, i]))
        train_stats["mape"].append(mean_absolute_pct_error(y_train[:, i], y_pred_train[:, i]))

        test_stats["mae"].append(mean_absolute_error(y_test[:, i], y_pred_test[:, i]))
        test_stats["rmse"].append(root_mean_squared_error(y_test[:, i], y_pred_test[:, i]))
        test_stats["r2"].append(r_squared(y_test[:, i], y_pred_test[:, i]))
        test_stats["mape"].append(mean_absolute_pct_error(y_test[:, i], y_pred_test[:, i]))

    # ------------------------------------------
    #  Print per-target results table
    # ------------------------------------------

    print("\n" + "=" * 60)
    print("  PER-TARGET RESULTS")
    print("=" * 60)
    print(f"\n  {'':22} {'--- Train ---':^26s}  |  {'--- Test ---':^26s}")
    print(f"  {'Target':<22} {'MAE':>8} {'RMSE':>8} {'R^2':>8}  |  {'MAE':>8} {'RMSE':>8} {'R^2':>8}")
    print("  " + "-" * 76)

    for i, name in enumerate(output_names):
        print(f"  {name:<22} "
              f"{train_stats['mae'][i]:>8.2f} {train_stats['rmse'][i]:>8.2f} "
              f"{train_stats['r2'][i]:>8.4f}  |  "
              f"{test_stats['mae'][i]:>8.2f} {test_stats['rmse'][i]:>8.2f} "
              f"{test_stats['r2'][i]:>8.4f}")

    # ------------------------------------------
    #  Generate visualizations
    # ------------------------------------------

    print("=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_r2_comparison(output_names, train_stats["r2"], test_stats["r2"])
    plot_mae_comparison(output_names, train_stats["mae"], test_stats["mae"])
    plot_predicted_vs_actual(output_names, y_test, y_pred_test)
    plot_residual_distributions(output_names, y_test, y_pred_test)
    plot_training_loss(model.loss_history)

    feat_names = ["Sleep Duration", "Sleep Quality", "Age", "Gender"]
    plot_feature_importance(model.weights, feat_names, output_names)

    # ------------------------------------------
    #  Export CSV
    # ------------------------------------------

    print("\n" + "=" * 60)
    print("  EXPORTING STATISTICS")
    print("=" * 60)

    export_statistics_csv(output_names, train_stats, test_stats)

    # ------------------------------------------
    #  Sample predictions
    # ------------------------------------------

    print("\n" + "=" * 60)
    print("  SAMPLE PREDICTIONS")
    print("=" * 60)

    for i in range(min(10, len(y_test))):
        predicted_bmi_number = int(round(y_pred_test[i, 0]))

        if predicted_bmi_number < 0:
            predicted_bmi_number = 0

        if predicted_bmi_number >= len(bmi_categories):
            predicted_bmi_number = len(bmi_categories) - 1

        actual_bmi_number = int(y_test[i, 0])

        print(f"\nPerson {i + 1}")
        print(f"  Actual BMI Category    : {bmi_categories[actual_bmi_number]}")
        print(f"  Predicted BMI Category : {bmi_categories[predicted_bmi_number]}")
        print(f"  Predicted BP           : {y_pred_test[i, 1]:.0f}/{y_pred_test[i, 2]:.0f}")
        print(f"  Predicted Heart Rate   : {y_pred_test[i, 3]:.0f}")
        print(f"  Predicted Daily Steps  : {y_pred_test[i, 4]:.0f}")
        print(f"  Predicted Stress Level : {y_pred_test[i, 5]:.1f}")
        print(f"  Predicted Activity Lvl : {y_pred_test[i, 6]:.1f}")

    print("\nDone.")
    print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()

    filepath = "Sleep_health_and_lifestyle_dataset.csv"

    if not os.path.exists(filepath):
        alt = os.path.join("/mnt/user-data/uploads", filepath)
        if os.path.exists(alt):
            filepath = alt

    if os.path.exists(filepath):
        run_health_prediction(filepath)