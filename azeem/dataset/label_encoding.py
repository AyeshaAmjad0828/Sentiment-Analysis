# Importing necessary libraries
import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import pyarrow as pa

# Main function for instantiating label encoder
@hydra.main(version_base=None, config_path=".", config_name="config")
def instantiate_label_encoder(cfg: DictConfig):

    # Informing about label encoding process initiation
    print('Generating label encoding...')

    # Getting absolute paths to the data files
    file_path = to_absolute_path(cfg.dataset.file_path)
    
    # Reading training and testing data from CSV files into Pandas DataFrames
    df_train = pd.read_csv(file_path + '\\' + cfg.dataset.file_name[0])
    df_test = pd.read_csv(file_path + '\\' + cfg.dataset.file_name[1])

    # Extracting labels from DataFrames and converting to lists
    train_labels = df_train[cfg.dataset.columns[1]].tolist()
    test_labels = df_test[cfg.dataset.columns[1]].tolist()

    # Combining labels from training and testing data
    labels = train_labels + test_labels

    # Instantiating the LabelEncoder using the provided configuration
    label_encoder = instantiate(cfg.label_encoder)
    
    # Fitting and transforming the label encoder on the combined labels
    encoded_labels = label_encoder.fit_transform(labels)

    # Separating encoded labels back into training and testing sets
    train_labels = encoded_labels[:len(train_labels)]
    test_labels = encoded_labels[len(train_labels):]

    # Converting the encoded labels into Pandas DataFrames
    train_labels_df = pd.DataFrame(train_labels)
    test_labels_df = pd.DataFrame(test_labels)

    # Writing the encoded training labels to a Parquet file
    with tqdm(total=len(train_labels_df), desc="Writing Train Parquet") as pbar:
        table = pa.Table.from_pandas(train_labels_df)
        pq.write_table(table, cfg.dataset.output.path + cfg.dataset.output.file_name[0])
        pbar.update(len(train_labels_df))
    
    # Writing the encoded testing labels to a Parquet file
    with tqdm(total=len(test_labels_df), desc="Writing Test Parquet") as pbar:
        table = pa.Table.from_pandas(test_labels_df)
        pq.write_table(table, cfg.dataset.output.path + cfg.dataset.output.file_name[1])
        pbar.update(len(test_labels_df))

    # Displaying the encoded training and testing labels
    print("Encoded train labels:", train_labels)
    print("Encoded test labels:", test_labels)

# Entry point of the script
if __name__ == "__main__":
    instantiate_label_encoder()
