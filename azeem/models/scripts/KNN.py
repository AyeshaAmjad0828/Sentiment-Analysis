import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score

@hydra.main(version_base=None, config_path="../configs", config_name="KN")
def knn(cfg: DictConfig):
    # Instantiate KNN model
    model = instantiate(cfg.model)

    # Get absolute paths for data and label files
    data_path = to_absolute_path(cfg.dataset.path)
    label_path = to_absolute_path(cfg.labels.path)

    # Read training data and labels from Parquet files
    X_train = pq.read_table(data_path + '\\' + cfg.dataset.file_name[0]).to_pandas()
    y_train = pq.read_table(label_path + '\\' + cfg.labels.file_name[0]).to_pandas()

    # Read testing data and labels from Parquet files
    X_test = pq.read_table(data_path + '\\' + cfg.dataset.file_name[1]).to_pandas()
    y_test = pq.read_table(label_path + '\\' + cfg.labels.file_name[1]).to_pandas()

    # Fit the model on training data
    model.fit(X_train, y_train.values.ravel())

    # Predict labels for testing data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print the accuracy
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    knn()
