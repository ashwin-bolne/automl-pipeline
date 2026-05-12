import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)


def get_scaler(scale_method: str) -> object:
    """
    Return scaler object based on configuration.

    Args:
        scale_method (str): Scaling method name.
    
    Returns:
        sklearn scaler object.
    """
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }

    if scale_method not in scalers:
        raise ValueError(
            f"Unsupported scale_method: {scale_method}"
        )
    
    return scalers[scale_method]


def get_encoder(encode_method: str) -> object:
    """
    Return encoder object based on configuration.

    Args:
        encode_method (str): Encoding method name.

    Returns:
        sklearn encoder object.
    """

    encoders = {
        "onehot": OneHotEncoder(
            handle_unknown="ignore"
        ),
    }

    if encode_method not in encoders:
        raise ValueError(
            f"Unsupported encode_method: {encode_method}"
        )
    
    return encoders[encode_method]


def make_preprocessor(
        X: pd.DataFrame,
        config: dict
) -> ColumnTransformer:
    """
    Create configurable preprocessing pipeline.

    Args:
        X (pd.DataFrame): Feature dataframe.
        config (dict): Project configuration dictionary.

    Returns:
        ColumnTransformer: Configured preprocessing object.
    """
    numeric_cols = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include="object"
    ).columns.tolist()

    numeric_config = config["preprocessing"]["numeric"]

    categorical_config = config["preprocessing"]["categorical"]

    numeric_impute_strategy = numeric_config["impute_strategy"]

    scale_method = numeric_config["scale_method"]

    categorical_impute_strategy = categorical_config["impute_strategy"]

    encode_method = categorical_config["encode_method"]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                strategy=numeric_impute_strategy
                ),
            ),
            (
                "scaler",
                get_scaler(scale_method),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=categorical_impute_strategy
                ),
            ),
            (
                "encoder",
                get_encoder(encode_method),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_cols,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_cols,
            ),
        ]
    )

    return preprocessor