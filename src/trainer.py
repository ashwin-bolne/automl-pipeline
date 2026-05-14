from src.model_registry import get_model

def train_model(
    model_name: str,
    X_train,
    y_train,
    config: dict,
):
    """
    Train machine learning model.

    Args:
        model_name (str): Model identifier.
        X_train: Processed training features.
        y_train: Training target vector.
        config (dict): Project configuration dictionary.

    Returns:
        Trained sklearn estimator.
    """

    model = get_model(
        model_name=model_name,
        config=config,
    )

    model.fit(X_train, y_train)

    return model 