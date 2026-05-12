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

# this part is just getting the data ready before the model can use it
def load_data(filepath: str):
    """Load CSV and extract the two features + target."""

    # reads the csv file so we can use the columns from it
    df = pd.read_csv(filepath)

    # Select columns by name for robustness
    # these are the main inputs the model looks at
    sleep_duration = df["Sleep Duration"].values.astype(float)    # feature 1
    sleep_quality  = df["Quality of Sleep"].values.astype(float)  # feature 2

    # this is what the model is trying to predict
    stress_level   = df["Stress Level"].values.astype(float)      # target

    # puts the two input columns together
    X = np.column_stack([sleep_duration, sleep_quality])   # (n, 2)

    # stores the answer column
    y = stress_level                                        # (n,)

    # sends back the inputs outputs and full dataset
    return X, y, df


# splits the data so we can train on one part and test on another part
def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Randomly split data into train / test sets."""

    # makes the random split stay the same every time we run it
    rng = np.random.default_rng(seed)

    # finds piut how many rows are in the data
    n = len(y)

    indices = rng.permutation(n)

    # finds where to split data
    split = int(n * (1 - test_ratio))

    # shuffled rows into train and test groups
    train_idx, test_idx = indices[:split], indices[split:]

    # return the training and testing data
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """Z-score normalization (mean=0, std=1) -- built from scratch."""

    def fit(self, X):
        # average for each feature
        self.mean_ = X.mean(axis=0)

        # standard dev
        self.std_  = X.std(axis=0)

        self.std_[self.std_ == 0] = 1.0

        # return scale
        return self

    def transform(self, X):
        # makes the data easier for gd 
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        # learns scale + applies 
        return self.fit(X).transform(X)


# ----------------------------------------------
# 2.  MODEL -- Linear Regression via Gradient Descent
# ----------------------------------------------

# this is the actual model that learns from the data
class LinearRegressionGD:
    """
    Multivariate linear regression trained with mini-batch
    gradient descent.  Everything from scratch -- no sklearn.

    Model:  y_hat = X @ w + b
    Loss :  MSE   = (1/n) * sum( (y_hat - y)^2 )
    """

    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32, seed=42):
        # controls how big each update step is
        self.lr         = learning_rate

        # how many times the model trains
        self.epochs     = epochs

        # how much data the model looks at at once
        self.batch_size = batch_size

        # keeps the randomness consistent
        self.seed       = seed

        # empty bc model has not learned yet
        self.weights    = None
        self.bias       = None

        # keeps track of the error after each it
        self.loss_history = []

    # ---- core math ------------------------------------------------

    @staticmethod
    def _mse(y_true, y_pred):
        # measures how many errors on avg
        return np.mean((y_true - y_pred) ** 2)

    def _predict(self, X):
        # basic prediction formula for linear regression
        return X @ self.weights + self.bias

    # ---- training -------------------------------------------------

    def fit(self, X, y):
        # starts the random number generator
        rng = np.random.default_rng(self.seed)

        # gets the number of rows and columns
        n_samples, n_features = X.shape

        # weights with random values instead of zeros
        self.weights = rng.normal(0, np.sqrt(2 / n_features), size=n_features)

        self.bias    = 0.0

        self.loss_history = []

        # training process
        for epoch in range(1, self.epochs + 1):
            # shuffle every epoch
            # mixes up the data so the model does not rely on order
            idx = rng.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]

            # this goes through the data in small chunks
            for start in range(0, n_samples, self.batch_size):
                end   = min(start + self.batch_size, n_samples)

                # one batch of inputs
                X_b   = X_shuf[start:end]

                # gets the matching real answers
                y_b   = y_shuf[start:end]

                # finds the size of the batch
                m     = len(y_b)

                # makes predictions using the curr weights
                y_hat = self._predict(X_b)

                # gradients
              
                # checks how far off the predictions are
                error = y_hat - y_b                        

                # tells the model how to change the weights
                dw    = (2 / m) * (X_b.T @ error)          

                # tells the model how to change the bias
                db    = (2 / m) * np.sum(error)             

                # parameter update
              
                # updates the weights to lower the error
                self.weights -= self.lr * dw

                # updates the bias to lower error
                self.bias    -= self.lr * db

            # epoch loss 
          
            # checks model error after one full training round
            train_pred = self._predict(X)

            # calculates the training loss
            loss = self._mse(y, train_pred)

            # saves the loss
            self.loss_history.append(loss)

            # prints progress while model is training
            if epoch % 200 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>5d}/{self.epochs}  --  MSE: {loss:.4f}")

        return self

    # ---- inference ------------------------------------------------

    def predict(self, X):
        # this uses the learned weights to predict new values
        return self._predict(X)


# ----------------------------------------------
# 3.  EVALUATION METRICS  (from scratch)
# ----------------------------------------------

# checks average prediction error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# checks error but makes bigger mistakes count more
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# checks how well the model explains the data
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ----------------------------------------------
# 4.  MAIN PIPELINE
# ----------------------------------------------

# runs the whole first model 
def main():
    # --- locate the actual data file ---
    filepath = "Sleep_health_and_lifestyle_dataset.csv"

    # checks if the file exists before trying to use it
    if not os.path.exists(filepath):
        alt = os.path.join("/mnt/user-data/uploads", filepath)

        # checks another place where the file might be 
        if os.path.exists(alt):
            filepath = alt

        # file is missing then the program stops
        else:
            print(f"ERROR: Cannot find '{filepath}'. Place it in the working directory.")
            sys.exit(1)

    # prints the title section
    print("=" * 60)
    print("  STRESS LEVEL PREDICTOR -- Linear Regression")
    print("=" * 60)

    # --- load ---
    # loads the input and output data
    X, y, df = load_data(filepath)

    # prints basic info about the dataset
    print(f"\nDataset loaded: {len(y)} samples")
    print(f"  Features : Sleep Duration (hrs), Quality of Sleep (1-10)")
    print(f"  Target   : Stress Level")
    print(f"  Stress range: {y.min():.0f} - {y.max():.0f}")

    # --- split ---
    # separates the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

    # shows how many rows are in each split
    print(f"\nTrain / Test split: {len(y_train)} / {len(y_test)}")

    # --- scale ---
    # input values easier for the model to learn from
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # --- train ---
    print("\n-- Training --")

    # creates the model with the settings 
    model = LinearRegressionGD(
        learning_rate = 0.01,
        epochs        = 2000,
        batch_size    = 32,
        seed          = 42,
    )

    # trains the model
    model.fit(X_train_s, y_train)

    # --- evaluate ---
    # makes predictions for both training and testing data
    y_pred_train = model.predict(X_train_s)
    y_pred_test  = model.predict(X_test_s)

    # results table
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<22} {'Train':>10} {'Test':>10}")
    print("-" * 44)

    # model errors
    print(f"{'MAE':<22} {mean_absolute_error(y_train, y_pred_train):>10.4f} {mean_absolute_error(y_test, y_pred_test):>10.4f}")
    print(f"{'RMSE':<22} {root_mean_squared_error(y_train, y_pred_train):>10.4f} {root_mean_squared_error(y_test, y_pred_test):>10.4f}")
    print(f"{'R^2 score':<22} {r_squared(y_train, y_pred_train):>10.4f} {r_squared(y_test, y_pred_test):>10.4f}")

    # --- learned parameters (unscaled for interpretability) ---
    # changes the weights for equation
    w_orig = model.weights / scaler.std_
    b_orig = model.bias - np.sum(model.weights * scaler.mean_ / scaler.std_)

    # prints the equation the model learned
    print(f"\nLearned equation (original scale):")
    print(f"  Stress ~ {w_orig[0]:+.4f} * SleepDuration "
          f"{w_orig[1]:+.4f} * SleepQuality "
          f"{b_orig:+.4f}")

    # --- sample predictions ---
    # shows some predicted values next to the real values
    print(f"\n{'Sleep Dur':>10} {'Quality':>10} {'Predicted':>12} {'Actual':>10}")
    print("-" * 45)

    # chooses random test examples
    rng = np.random.default_rng(99)
    sample_idx = rng.choice(len(y_test), size=min(10, len(y_test)), replace=False)

    # prints sample predictions
    for i in sample_idx:
        print(f"{X_test[i, 0]:>10.1f} {X_test[i, 1]:>10.0f}"
              f" {y_pred_test[i]:>12.2f} {y_test[i]:>10.0f}")

    # --- interactive prediction ---
    # lets someone type in their own sleep data
    print("\n" + "=" * 60)
    print("  INTERACTIVE PREDICTOR")
    print("=" * 60)
    print("Enter sleep data to predict stress level (or 'q' to quit):\n")

    # keeps asking until q
    while True:
        try:
            dur_input = input("  Sleep Duration (hours) : ").strip()

            # stops if the user types q
            if dur_input.lower() == 'q':
                break

            qual_input = input("  Sleep Quality  (1-10)  : ").strip()

            # q --> quit
            if qual_input.lower() == 'q':
                break

            # turns the typed values into numbers
            dur  = float(dur_input)
            qual = float(qual_input)

            # formats the new input like the training data
            x_new   = np.array([[dur, qual]])

            # scale new input
            x_new_s = scaler.transform(x_new)

            # makes the prediction
            pred    = model.predict(x_new_s)[0]

            # prints the final predicted stress level
            print(f"  -> Predicted Stress Level: {pred:.2f}\n")

        # input is not valid end loop
        except (ValueError, EOFError):
            break

    print("\nDone.")

# ----------------------------------------------
# 5.  HEALTH DATA LOADING
# ----------------------------------------------

# this next part of the code uses more inputs and predicts more health related things
def load_health_data(filepath: str):
    """
    Loads the same CSV file but uses more inputs and outputs.
    """

    # reads the dataset again
    df = pd.read_csv(filepath)

    # -------------------------------
    # INPUT 1: Sleep Duration
    # -------------------------------
    # first input
    sleep_duration = df["Sleep Duration"].values.astype(float)

    # -------------------------------
    # INPUT 2: Quality of Sleep
    # -------------------------------
    # second input
    sleep_quality = df["Quality of Sleep"].values.astype(float)

    # -------------------------------
    # INPUT 3: Age
    # -------------------------------
    # third input
    age = df["Age"].values.astype(float)

    # -------------------------------
    # INPUT 4: Gender
    # -------------------------------
    # gender as text first
    gender_raw = df["Gender"].values

    # number column for gender
    gender = np.zeros(len(gender_raw))

    # changes gender into numbers on normalized scale
    for i in range(len(gender_raw)):
        if gender_raw[i].lower() == "female":
            gender[i] = 1.0
        else:
            gender[i] = 0.0

    # puts all the inputs into one matrix
    X = np.column_stack(
        [
            sleep_duration,
            sleep_quality,
            age,
            gender
        ]
    )

    # returns the input matrix + full dataframe
    return X, df

def prepare_health_outputs(df):
    """
    Prepares the health outputs that the program will predict.
    """

    # -------------------------------
    # OUTPUT 1: BMI Category
    # -------------------------------
    # all BMI category names
    bmi_categories = sorted(df["BMI Category"].unique())

    # match each BMI category to a number
    bmi_to_number = {}

    # each BMI category has a number
    for i in range(len(bmi_categories)):
        bmi_to_number[bmi_categories[i]] = i

    # BMI category text into numbers
    bmi_category = df["BMI Category"].map(bmi_to_number).values.astype(float)

    # -------------------------------
    # OUTPUT 2: Blood Pressure
    # -------------------------------
    # splits blood pressure into top and bottom numbers
    bp_parts = df["Blood Pressure"].str.split("/", expand=True)

    # top blood pressure number
    systolic_bp = bp_parts[0].values.astype(float)

    # bottom blood pressure number
    diastolic_bp = bp_parts[1].values.astype(float)

    # -------------------------------
    # OUTPUT 3: Resting Heart Rate
    # -------------------------------
    # heart rate as an output
    heart_rate = df["Heart Rate"].values.astype(float)

    # -------------------------------
    # OUTPUT 4: Daily Steps
    # -------------------------------
    # daily steps as an output
    daily_steps = df["Daily Steps"].values.astype(float)

    # -------------------------------
    # OUTPUT 5: Stress Level
    # -------------------------------
    # stress level as an output
    stress_level = df["Stress Level"].values.astype(float)

    # -------------------------------
    # OUTPUT 6: Physical Activity Level
    # -------------------------------
    # physical activity level as an output
    activity_level = df["Physical Activity Level"].values.astype(float)

    # all the outputs into one matrix
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

    # returns the outputs and the BMI category names
    return y, bmi_categories
    
# ----------------------------------------------
# 7.  LINEAR REGRESSION FOR MULTIPLE OUTPUTS
# ----------------------------------------------

# this model is for predicting multiple outputs at once
class MultiOutputLinearRegressionGD:
    """
    This is almost the same idea as the first linear regression class,
    but this one can predict more than one output at the same time.
    """

    def __init__(self, learning_rate=0.01, epochs=2000, seed=42):
        # how fast the model learns
        self.learning_rate = learning_rate

        # this controls how long the model trains
        self.epochs = epochs

        # random values stay consistent
        self.seed = seed

        # start empty before training
        self.weights = None
        self.bias = None

        # loss values
        self.loss_history = []

    def predict(self, X):
        # predicts all the outputs
        return X @ self.weights + self.bias

    def fit(self, X, y):
        # random number generator
        rng = np.random.default_rng(self.seed)

        # number of samples
        n_samples = X.shape[0]

        # umber of input columns
        n_features = X.shape[1]

        # number of output columns
        n_outputs = y.shape[1]

        self.weights = rng.normal(0, 0.1, size=(n_features, n_outputs))

        self.bias = np.zeros(n_outputs)

        # this trains the model
        for epoch in range(1, self.epochs + 1):
            # predictions with the current weights
            y_hat = self.predict(X)

            # checks how wrong the predictions are
            error = y_hat - y

            # alculates how the weights should change
            dw = (2 / n_samples) * (X.T @ error)

            db = (2 / n_samples) * np.sum(error, axis=0)

            self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db

            # calculates the error epoch
            loss = np.mean(error ** 2)

            self.loss_history.append(loss)

            if epoch == 1 or epoch % 500 == 0:
                print(f"  Epoch {epoch:>5d}/{self.epochs}  --  MSE: {loss:.4f}")

        # returns the trained model
        return self
    
def run_health_prediction(filepath: str):
    """
    Runs the health prediction part of the project.
    """

    print("\n" + "=" * 60)
    print("  HEALTH PREDICTION FROM SLEEP DATA")
    print("=" * 60)

    # loads health input data
    X, df = load_health_data(filepath)

    y, bmi_categories = prepare_health_outputs(df)

    print(f"\nDataset loaded: {len(y)} samples")

    # model inputs
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

    # splits data into train and test groups
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_ratio=0.2,
        seed=42
    )

    # creates scaler
    scaler = StandardScaler()

    # scales training inputs
    X_train_s = scaler.fit_transform(X_train)

    # scales testing inputs
    X_test_s = scaler.transform(X_test)

    print("\n-- Training --")

    # multi output model
    model = MultiOutputLinearRegressionGD(
    learning_rate=0.01,
    epochs=2000,
    seed=42
)

    # trains the model
    model.fit(X_train_s, y_train)

    # predicts the test values
    y_pred = model.predict(X_test_s)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    # output names for the results
    output_names = [
        "BMI Category",
        "Systolic Blood Pressure",
        "Diastolic Blood Pressure",
        "Resting Heart Rate",
        "Daily Steps",
        "Stress Level",
        "Physical Activity Level"
    ]

    # prints the errors for each output
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

    # user try their own values
    print("\n" + "=" * 60)
    print("  TRY YOUR OWN INPUT")
    print("=" * 60)
    print("Type q to quit.\n")

    # keeps asking for input until the user quits
    while True:
        try:
            dur_input = input("  Sleep Duration (hours) : ").strip()

            # quits if the user types q
            if dur_input.lower() == "q":
                break

            qual_input = input("  Quality of Sleep (1-10): ").strip()

            # quits if the user types q
            if qual_input.lower() == "q":
                break

            age_input = input("  Age                    : ").strip()

            # quits if the user types q
            if age_input.lower() == "q":
                break

            gender_input = input("  Gender (Male/Female)   : ").strip()

            # quits if the user types q
            if gender_input.lower() == "q":
                break

            # typed numbers into floats
            sleep_duration = float(dur_input)
            sleep_quality = float(qual_input)
            age = float(age_input)

            if gender_input.lower().startswith("m"):
                gender = 0
            else:
                gender = 1

            # puts the user input into a readable format
            user_data = np.array(
                [
                    [
                        sleep_duration,
                        sleep_quality,
                        age,
                        gender
                    ]
                ]
            )

            # scales the user input
            user_data_s = scaler.transform(user_data)

            # gets the prediction
            prediction = model.predict(user_data_s)[0]

            # this turns the BMI prediction into a category number
            predicted_bmi_number = int(round(prediction[0]))

            if predicted_bmi_number < 0:
                predicted_bmi_number = 0

            if predicted_bmi_number >= len(bmi_categories):
                predicted_bmi_number = len(bmi_categories) - 1

            # prints the predicted health results
            print("\n  Predicted Results")
            print("  " + "-" * 35)
            print(f"  BMI Category            : {bmi_categories[predicted_bmi_number]}")
            print(f"  Blood Pressure          : {prediction[1]:.0f}/{prediction[2]:.0f}")
            print(f"  Resting Heart Rate      : {prediction[3]:.0f}")
            print(f"  Daily Steps             : {prediction[4]:.0f}")
            print(f"  Stress Level            : {prediction[5]:.1f}")
            print(f"  Physical Activity Level : {prediction[6]:.1f}")
            print()

        # handles invalid input
        except (ValueError, EOFError):
            print("\nInvalid input. Please try again.\n")
            break

if __name__ == "__main__":
    main()

    filepath = "Sleep_health_and_lifestyle_dataset.csv"

    # runs the health prediction part if the file exists
    if os.path.exists(filepath):
        run_health_prediction(filepath)
