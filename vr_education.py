# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time  # Import time module

# Function to find the features with the highest correlation to the label
def select_features_based_on_correlation(data, label, top_n=5):
    # Drop non-numeric and irrelevant columns
    data = data.drop(columns=['Student_ID'], errors='ignore')
    data = data.select_dtypes(include=[np.number])

    # Calculate the correlation matrix
    corr_matrix = data.corr()
    
    # Get the correlations of the label with all other features
    correlations = corr_matrix[label].drop(label)  # Drop the label itself

    # Sort by absolute correlation values
    sorted_correlations = correlations.abs().sort_values(ascending=False)

    # Select top N features with the highest correlation
    top_features = sorted_correlations.head(top_n).index.tolist()

    return top_features

# Function to find the two variables with the strongest correlation
def find_strongest_correlation(data):
    # Drop the 'Student_ID' column (or any other non-useful columns)
    data = data.drop(columns=['Student_ID'])

    # Drop rows with missing values
    data = data.dropna()

    # Convert categorical columns to numeric using one-hot encoding
    data = pd.get_dummies(data, drop_first=True)

    # Select only numeric columns for correlation calculation
    numeric_data = data.select_dtypes(include=[np.number])

    # Ensure all columns are numeric
    if numeric_data.empty:
        print("No numeric columns available for correlation.", end="\n\n")
        return None, None

    # Calculate the correlation matrix
    corr_matrix = numeric_data.corr()

    # Mask the diagonal (self-correlation) by setting them to NaN
    np.fill_diagonal(corr_matrix.values, np.nan)

    # Find the maximum correlation value
    max_corr = corr_matrix.unstack().idxmax()
    max_corr_value = corr_matrix.unstack().max()

    print(f"The two variables with the strongest correlation are: {max_corr[0]} and {max_corr[1]}")
    print(f"The correlation value is: {max_corr_value}", end="\n\n")

    return corr_matrix, (max_corr[0], max_corr[1])

# Function to run regression analysis
def run_regression_analysis(data):
    # Define the target and features for regression analysis
    regression_target = 'Perceived_Effectiveness_of_VR'
    # Assuming select_features_based_on_correlation is defined elsewhere and returns the relevant features
    regression_features = select_features_based_on_correlation(data, regression_target)
    
    # Split the data into features and label
    X = data[regression_features]
    y = data[regression_target]

    # Check for NaN values in the features and label
    if X.isnull().values.any() or y.isnull().values.any():
        print("Dropping rows with NaN values in features or label...")
        # Drop rows with NaN values in either features or label
        data = data.dropna(subset=regression_features + [regression_target])
        X = data[regression_features]
        y = data[regression_target]

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if the training data is empty
    if X_train.empty or y_train.empty:
        print("Training data is empty. Cannot proceed with regression analysis.")
        return None  # Exit the function if there's no data

    # Create a Linear Regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Generate predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate MSE and RMSE manually
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # Print MSE and RMSE
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    return None  # No plotting done inside this function

# Function to run logistic regression classification
def run_logistic_regression(data):
    # Define features and target variable
    features = ['Hours_of_VR_Usage_Per_Week', 'Engagement_Level']
    target = 'Improvement_in_Learning_Outcomes'

    # Map the target variable to binary values (assuming 'Yes' and 'No' are the possible values)
    data[target] = data[target].map({'Yes': 1, 'No': 0})

    # Drop rows with NaN values in the target variable
    data = data.dropna(subset=[target])

    # Split the dataset into features and label
    X = data[features]
    y = data[target]

    # Check if the features and target are empty
    if X.empty or y.empty:
        print("Features or target variable is empty. Cannot proceed with train_test_split.")
        return None, None  # Exit the function if there's no data
    
    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Generate predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print accuracy and classification report
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    return conf_matrix

# Main code block
if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    # Define the path to the dataset
    file_path = "sat5165/dataset.csv"

    # Load the dataset into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Find and print the two variables with the strongest correlation
    corr_matrix, strongest_corr_vars = find_strongest_correlation(data)

    # Run regression analysis using the selected features
    run_regression_analysis(data)

    # Run logistic regression classification and get confusion matrix
    conf_matrix = run_logistic_regression(data)

    # Calculate and print the total execution time
    execution_time = time.time() - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")

    # Plot after execution time is printed
    if corr_matrix is not None:
        # Plot the heatmap of the correlation matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title("Correlation Matrix Heatmap")
        plt.show()

    if conf_matrix is not None:
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Improvement', 'Improvement'], 
                    yticklabels=['No Improvement', 'Improvement'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
