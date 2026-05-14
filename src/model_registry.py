from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)

def get_model(
        model_name: str,
        config: dict,
):
    """
    Return initialized model object base on model name.

    Args:
        model_name (str): Model identifier.
        config (dict): Project configuration dictionary.
    
    Returns:
        sklearn estimator object.
    """

    models = {
        "linear_regression": LinearRegression(),\
        "ridge": Ridge(
            alpha=config["model"].get("alpha", 1.0)
        ),
        "lasso": Lasso(
            alpha=config["model"].get("alpha", 1.0)
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000
        )
    }

    if model_name not in models:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )
    
    return models[model_name]