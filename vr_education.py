from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
import time

# Start Spark session
spark = SparkSession.builder.appName("vr-education").getOrCreate()

# Load the dataset
file_path = "/opt/spark/dataset.csv"
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Select features based on correlation to target label
def select_features_based_on_correlation(data, label, top_n=5):
    # Remove non-numeric and irrelevant columns
    numeric_data = data.select([col for col, dtype in data.dtypes if dtype in ('int', 'double')])
    
    # Calculate correlation with the label for each feature
    correlations = [(col, numeric_data.stat.corr(col, label)) for col in numeric_data.columns if col != label]
    
    # Sort by absolute correlation values and select top N features
    sorted_correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)
    top_features = [item[0] for item in sorted_correlations[:top_n]]
    
    return top_features

# Find the two variables with the strongest correlation
def find_strongest_correlation(data):
    # Filter only numeric columns
    numeric_data = data.select([col for col, dtype in data.dtypes if dtype in ('int', 'double')])
    
    # Generate pairwise correlations and find the highest correlation pair
    max_corr_value, max_corr_cols = 0, None
    columns = numeric_data.columns
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            corr_value = numeric_data.stat.corr(columns[i], columns[j])
            if abs(corr_value) > abs(max_corr_value):
                max_corr_value, max_corr_cols = corr_value, (columns[i], columns[j])
    
    print(f"Strongest correlation is between {max_corr_cols[0]} and {max_corr_cols[1]} with a value of {max_corr_value}")

# Regression analysis
def run_regression_analysis(data):
    regression_target = 'Perceived_Effectiveness_of_VR'
    regression_features = select_features_based_on_correlation(data, regression_target)
    
    # Assemble features into a single vector column
    assembler = VectorAssembler(inputCols=regression_features, outputCol="features")
    data = assembler.transform(data)
    
    # Split dataset into training and testing sets
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    
    # Create and train a Linear Regression model
    lr = LinearRegression(featuresCol="features", labelCol=regression_target)
    lr_model = lr.fit(train)
    
    # Predict on the test set
    predictions = lr_model.transform(test)
    
    # Evaluate model performance
    evaluator = RegressionEvaluator(labelCol=regression_target, predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    
    print(f"Root Mean Squared Error (RMSE): {rmse}")

# Logistic regression classification
def run_logistic_regression(data):
    # Define features and target variable
    features = ['Hours_of_VR_Usage_Per_Week', 'Engagement_Level']
    target = 'Improvement_in_Learning_Outcomes'
    
    # Index the target variable
    indexer = StringIndexer(inputCol=target, outputCol="label")
    data = indexer.fit(data).transform(data)
    
    # Assemble features
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    data = assembler.transform(data)
    
    # Split dataset into training and testing sets
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    
    # Create and train a Logistic Regression model
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    lr_model = lr.fit(train)
    
    # Predict on the test set
    predictions = lr_model.transform(test)
    
    # Evaluate accuracy and confusion matrix
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    
    print(f"Accuracy: {accuracy}")

# Main block
if __name__ == "__main__":
    start_time = time.time()
    
    # Run correlation analysis
    find_strongest_correlation(data)
    
    # Run regression and classification
    run_regression_analysis(data)
    run_logistic_regression(data)
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

# Stop Spark session
spark.stop()
