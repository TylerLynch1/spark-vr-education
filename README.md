# README: Analyzing VR's Impact on Education

## Project Overview
This project analyzes the impact of Virtual Reality (VR) on education using PySpark for data preprocessing, correlation analysis, regression analysis, and classification. The goal is to identify meaningful insights into the effectiveness of VR in education.

## Features and Capabilities
1. Correlation Analysis
- Identifies features with the strongest correlation to the target variable.
- Finds the pair of variables with the strongest correlation.
2. Regression Analysis
- Uses linear regression topredict the perceived effectiveness of VR based on highly correlated features.
- Evaluates the regression model using RootMeanSquaredError (RMSE).
3. Logistic Regression Classification
- Classifies improvements in learning outcomes based on selected features.
- Evaluates the classification model's accuracy using a Multiclass Classification Evaluator.

## Dependencies
- Python3.x
- PySpark
- Java (requiredforSpark)

## File Structure
### Script:
- vr_education.py: Main script containing all analysis functions.
### Dataset:
- The dataset should be placed at /opt/spark/dataset.csv.

## Important Note
The dataset path /opt/spark/dataset.csv assumes the use of a Linux-based system. Ensure that your environment supports this directory structure or adjust accordingly.

## Installation and Setup
### Install PySpark:
```pip install pyspark```

Place the dataset in the /opt/spark directory with the file name dataset.csv.
Ensure Java and Spark are properly installed and configured.

## How to Run
Open a terminal in the project directory.
Execute the script using Python:
```python vr_education.py```

## Script Breakdown
### 1. Correlation Analysis
Function: find_strongest_correlation(data).
Identifies the two variables with the strongest correlation in the dataset.
Function: select_features_based_on_correlation(data, label, top_n=5)
Selects the top N features most correlated to the target variable.

### 2. Regression Analysis
Function: run_regression_analysis(data).
Predicts the Perceived_Effectiveness_of_VR using linear regression.
Evaluates model performance with RMSE.

### 3. Logistic Regression Classification
Function: run_logistic_regression(data).
Classifies Improvement_in_Learning_Outcomes using logistic regression.
Evaluates accuracy of the model.

## Inputs and Outputs
### Inputs:
Dataset File: dataset.csv should include columns such as:
Perceived_Effectiveness_of_VR (target for regression)
Improvement_in_Learning_Outcomes (target for classification)
Other numerical and categorical features.

### Outputs:
Printed Results:
Strongest correlation pair and value.
RMSE for regression analysis.
Classification accuracy.
Execution Time:
Total script runtime.

## Key Functions
### find_strongest_correlation(data)
Identifies the pair of variables with the highest correlation.

### run_regression_analysis(data)
Selects features based on correlation to Perceived_Effectiveness_of_VR.
Fits and evaluates a linear regression model.

### run_logistic_regression(data)
Performs logistic regression to predict Improvement_in_Learning_Outcomes.
Outputs model accuracy.

## Performance Metrics
Regression: Root Mean Squared Error (RMSE).
Classification: Model Accuracy.

## Execution Time
The script outputs the total runtime for analysis.

## License
This project is provided under the MIT License.

## Acknowledgments
PySpark Documentation: https://spark.apache.org/docs/latest/api/python/.

Dataset: Ensure the dataset used conforms to privacy and sharing policies.
