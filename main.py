import yaml
import joblib
import mlflow
import mlflow.sklearn

from src.data_loader import (
    load_dataset, 
    get_features_and_target,
    split_dataset
)
from src.preprocessor import make_preprocessor
from src.trainer import train_model
from src.evaluator import evaluate

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "automl-regression"
)

def main():
    """
    Main workflow entrypoint.
    """

    with open("configs/regression.yaml", "r") as file:
        config = yaml.safe_load(file)

    model_name = config["model"]["name"]

    task_type = config["task"]["type"]

    with mlflow.start_run():

        df = load_dataset(config)

        # print("\nDataset Loaded successfully.")
        # print(f"\nDataset Shape: {df.shape}")

        target_col = config["dataset"]["target"]

        X, y = get_features_and_target(
            df,
            target_col
        )
        
        # print(f"\nFeature Matrix shape: {X.shape}")
        # print(f"\nTarget vector shape: {y.shape}")
        
        X_train, X_test, y_train, y_test = split_dataset(
            X,
            y,
            test_size=config["split"]["test_size"],
            random_state=config["split"]["random_state"]
        )

        # print(f"\nTrain Feature shape: {X_train.shape}")

        # print(f"\nTest Feature shape: {X_test.shape}")

        # print(f"\ny_train shape: {y_train.shape}")

        # print(f"\ny_test shape: {y_test.shape}")

        preprocessor = make_preprocessor(
            X_train,
            config,
        )

        X_train_processed = preprocessor.fit_transform(
            X_train
        )

        X_test_processed = preprocessor.transform(
            X_test
        )

        print(
            f"\nProcessed Train Shape: "
            f"{X_train_processed.shape}"
        )

        print(
            f"\nProcessed Test Shape: "
            f"{X_test_processed.shape}"
        )

        joblib.dump(
            preprocessor,
            "models/preprocessor.joblib"
        )

        print(
            "\nPreprocessor saved successfully."
        )

        model = train_model(
            model_name=model_name,
            X_train=X_train_processed,
            y_train=y_train,
            config=config,
        )

        mlflow.log_params(config["model"])

        print(
            f"\nModel trained successfully: "
            f"{model_name}"
        )

        metrics = evaluate(
        model=model,
        X_test=X_test_processed,
        y_test=y_test,
        task_type=task_type,
        )

        mlflow.log_metrics(metrics)
        
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        print("\nEvaluation Metrics")
        print("-" * 40)

        for metric_name, metric_value in metrics.items():
            print(
                f"{metric_name}: "
                f"{metric_value:.4f}"
            )



if __name__ == "__main__":
    main()