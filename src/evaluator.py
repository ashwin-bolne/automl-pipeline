from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def evaluate(
        model,
        X_test,
        y_test,
        task_type: str,
) -> dict:
    """
    Evaluate trained model performance.

    Args:
        model: Trained sklearn model.
        X_test: Processed test features.
        y_test: Ground truth target values.
        task_type (str): Either 'regression' or 'classification'.

    Returns;
        dict: Evaluate metrics.
    """

    predictions = model.predict(X_test)

    if task_type == "regression":

        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        r2 = r2_score(
            y_test,
            predictions
        )

        return {
            "rmse": rmse,
            "r2": r2,
        }
    
    elif task_type == "classification":

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        f1 = f1_score(
            y_test,
            predictions, 
        )

        probabilties = model.predict_proba(
            X_test
        )[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            probabilties,
        )

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }
    
    else:
        raise ValueError(
            f"Unsupported task_type "
            f"{task_type}"
        )