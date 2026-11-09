import mlflow, mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("iris-demo")

X, y = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="IrisClassifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    print(f"Accuracy: {acc:.4f}")