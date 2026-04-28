# ECE4424 Project

## Sleep Health and Lifestyle Prediction Model

This project uses the Sleep Health and Lifestyle Dataset to explore how sleep and personal health factors can be used to predict lifestyle and wellness outcomes. The dataset contains information such as age, gender, sleep duration, sleep quality, physical activity level, stress level, BMI category, blood pressure, heart rate, daily steps, and sleep disorder information.

The goal of this project is to build a machine learning model that can take user inputs related to sleep and health and predict important lifestyle metrics.

## Project Goal

The main goal of this project is to predict health-related outcomes using sleep and lifestyle data.

For our model, the user inputs include:

- Sleep quality
- Sleep duration
- Age
- Gender

The model is trained to predict:

- BMI category
- Blood pressure
- Resting heart rate
- Daily steps
- Stress level
- Activity level

This allows the project to show how sleep habits and basic personal information may be connected to overall health and lifestyle patterns.

## Dataset

The dataset used in this project is the Sleep Health and Lifestyle Dataset from Kaggle. Each row represents one person, and each column represents a different health or lifestyle feature.

Some important columns include:

- Age
- Gender
- Sleep Duration
- Quality of Sleep
- Physical Activity Level
- Stress Level
- BMI Category
- Blood Pressure
- Heart Rate
- Daily Steps
- Sleep Disorder

Since the dataset is small, the model may not generalize perfectly to all real-world cases. However, it is useful for learning how machine learning can be applied to health and lifestyle prediction.

## Preprocessing

Before training the model, the data is cleaned and prepared.

The preprocessing steps include:

1. Loading the CSV file using pandas
2. Selecting the input columns and output columns
3. Handling categorical values such as gender, BMI category, and blood pressure
4. Splitting the data into training and testing sets
5. Standardizing numerical input values so the model trains more smoothly

## Model

## Models Used

This project uses supervised learning models for both regression and classification tasks.

For numerical outputs, such as resting heart rate, daily steps, stress level, and activity level, regression models are used. These models predict a number based on the input features.

For categorical outputs, such as BMI category and blood pressure category, classification models are both being used to predict labels.

The model is trained using the dataset and then tested on separate data to check how well it performs. The goal is not only to make predictions, but also to understand how sleep and lifestyle factors may be related to health outcomes.

## How to Run the Project

### 1. Clone the repository

```bash
git clone <https://github.com/smh47VT/ECE4424-Project.git>