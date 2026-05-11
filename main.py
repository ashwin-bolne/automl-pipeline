import yaml

from src.data_loader import (
    load_dataset, 
    get_features_and_target,
    split_dataset
)


def main():
    """
    Main workflow entrypoint.
    """

    with open("configs/regression.yaml", "r") as file:
        config = yaml.safe_load(file)

    df = load_dataset(config)

    target_col = config["dataset"]["target"]

    X, y = get_features_and_target(
        df,
        target_col
    )

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"]
    )


    print("\nDataset Loaded successfully.")

    print(f"\nDataset Shape: {df.shape}")

    print(f"\nFeature Matrix shape: {X.shape}")

    print(f"\nTarget vector shape: {y.shape}")
    
    print(f"\nTrain Feature shape: {X_train.shape}")

    print(f"\nTest Feature shape: {X_test.shape}")

    print(f"\ny_train shape: {y_train.shape}")

    print(f"\ny_test shape: {y_test.shape}")

if __name__ == "__main__":
    main()