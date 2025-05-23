import dagshub
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting the script...")

dagshub.init(repo_owner='piyushugale', repo_name='6-mlops-experiments-with-mlflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/piyushugale/6-mlops-experiments-with-mlflow.mlflow")
mlflow.autolog()

print("Autolog enabled...")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 20
n_estimators = 500

# Mention your experiment below
mlflow.set_experiment('mlops-experiment-autolog-01')

with mlflow.start_run():
    print("Training started...")
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Training completed...")

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts using mlflow
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'piyush', "Project": "Wine Classification"})

    print(accuracy)