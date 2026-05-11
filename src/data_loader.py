import pandas as pd 

from sklearn.model_selection import train_test_split

def load_dataset(config: dict) -> pd.DataFrame:
    """
    Load dataset using configuration dictionary.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        pd.DataFrame: Loaded dataset.
    """

    dataset_path = config["dataset"]["path"]

    df = pd.read_csv(dataset_path)

    return df 

def get_features_and_target(df: pd.DataFrame, target_col: str):
    """
    Separate features and target from dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Target column name.

    Returns:
        tuple: Feature matrix (X) abd target vector (y).
    """
    X = df.drop(columns=[target_col])

    y = df[target_col]

    return X, y

def split_dataset(
        X,
        y,
        test_size: float,
        random_state: int,
        stratify=None
):
    """
    Split dataset into train and test sets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size (float): Test split ratio.
        random_state (int): Random seed for reproducibility.
        stratify: Stratification target for classification.
    
    Returns:
        tuple: Split datasets.    
     """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    return X_train, X_test, y_train, y_test