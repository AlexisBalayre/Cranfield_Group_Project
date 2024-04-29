import pandas as pd
from sklearn.model_selection import train_test_split

# Parameters
input_file_path = "/Users/alexis/Cranfield/Group Project/Repos1/data/oden_db/channel/combined_channel.csv"
output_folder_path = (
    "/Users/alexis/Cranfield/Group Project/Repos1/data/oden_db/channel/"
)

TEST_SIZE = 0.2  # Split the dataset into 80% for training and 20% for testing
VALIDATION_SIZE = (
    0.15  # Split the training dataset into 85% for training and 15% for validation
)

statify_column = "Re_tau"  # The column to stratify the dataset on
random_state = None  # Random state for reproducibility


if __name__ == "__main__":

    # Load the dataset
    df = pd.read_csv(input_file_path)

    # Split the dataset into training and temporary datasets (temporary will be further split into validation and test)
    temp_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[statify_column], random_state=random_state
    )

    # Split the temporary dataset into validation and test datasets
    train_df, val_df = train_test_split(
        temp_df,
        test_size=VALIDATION_SIZE,
        stratify=temp_df[statify_column],
        random_state=random_state,
    )

    # Save the datasets to new CSV files
    train_df.to_csv(output_folder_path + "train_dataset.csv", index=False)
    val_df.to_csv(output_folder_path + "val_dataset.csv", index=False)
    test_df.to_csv(output_folder_path + "test_dataset.csv", index=False)
