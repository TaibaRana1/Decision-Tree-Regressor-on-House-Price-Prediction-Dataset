# Decision-Tree-Regressor-on-House-Price-Prediction-Dataset

# Overview
This project involves training and evaluating a Decision Tree Regressor model on a comprehensive house price prediction dataset. The goal is to predict housing prices based on various features using decision tree algorithms. This project builds upon prior work where the same dataset was used for linear regression.

# Dataset
The dataset used in this project is the House Price Prediction dataset, which contains 81 columns representing various features of houses (e.g., overall quality, area, year built, number of rooms, etc.).

# Objective
The primary objective is to:
* Train a DecisionTreeRegressor model using the scikit-learn library
* Explore and fine-tune key hyperparameters
* Evaluate model performance using relevant regression metrics
* Visualize the trained decision tree if feasible

# Tools and Technologies
Jupyter Notebook / Google Colab
Pandas for data manipulation
NumPy for numerical computations
Scikit-learn (sklearn) for machine learning modeling
Matplotlib / Seaborn for plotting and visualization
Graphviz / sklearn.tree.plot_tree for decision tree visualization (optional)

## Model Implementation
The core model used is DecisionTreeRegressor from the sklearn.tree module. 
# The following hyperparameters are explored:
* max_depth: Maximum depth of the tree
* min_samples_split: Minimum number of samples required to split an internal node
* min_samples_leaf: Minimum number of samples required to be at a leaf node
* criterion: Function to measure the quality of a split (mse, friedman_mse, mae, etc.)
* random_state: Ensures reproducibility

# Steps Followed
* Load the Dataset
Load the preprocessed dataset using Pandas
* Train-Test Split
Split the dataset into training and testing sets (e.g., 80-20 ratio)

* Model Training
Train DecisionTreeRegressor with default parameters
Tune hyperparameters using manual experimentation or GridSearchCV

Model Evaluation

Evaluate the model using the following metrics:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared (R² Score)


# Evaluation Metrics
Metric	Description
* MAE	Measures the average magnitude of errors in predictions, without considering their direction
* MSE	Measures the average of the squared differences between actual and predicted values
* RMSE	Square root of MSE, interpretable in the same units as the target variable
* R² Score	Represents the proportion of variance in the dependent variable explained by the model

# Key Learnings
Understanding the basics of decision tree regression
Importance of controlling tree depth to avoid overfitting
How decision trees capture non-linear patterns
Performance evaluation of regression models using various metrics
Practical use of scikit-learn's decision tree visualization tools



