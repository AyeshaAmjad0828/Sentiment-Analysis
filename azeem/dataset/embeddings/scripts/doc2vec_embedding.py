# Importing necessary libraries
import gensim
import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

# Main function for Doc2Vec embedding
@hydra.main(version_base=None, config_path="../config", config_name="d2v_config")
def doc2vec_embedding(cfg: DictConfig):

    # Getting absolute path to the data file
    file_path = to_absolute_path(cfg.data.path)
    
    # Reading training and testing data from Parquet files into Pandas DataFrames
    df_train = pq.read_table(file_path + '\\' + cfg.data.filename[0]).to_pandas()
    df_test = pq.read_table(file_path + '\\' + cfg.data.filename[1]).to_pandas()

    # Extracting text reviews from DataFrames and converting to lists
    train_reviews = df_train[cfg.data.column].tolist()
    test_reviews = df_test[cfg.data.column].tolist()

    # Preparing TaggedDocument objects for training and testing data
    tagged_train_reviews = [TaggedDocument(words=gensim.utils.simple_preprocess(review), tags=[str(i)]) for i, review in tqdm(enumerate(train_reviews))]
    tagged_test_reviews = [TaggedDocument(words=gensim.utils.simple_preprocess(review), tags=[str(i)]) for i, review in tqdm(enumerate(test_reviews))]

    # Instantiating the Doc2Vec model using the provided configuration
    model = instantiate(cfg.model)
    
    # Building vocabulary for the model using training data
    model.build_vocab(tagged_train_reviews)
    
    # Training the Doc2Vec model
    model.train(tagged_train_reviews, total_examples=model.corpus_count, epochs=10)

    # Initializing lists to store vectors for training and testing data
    train_vectors = []
    test_vectors = []

    # Generating vectors for training data
    print("Generating train vectors:")
    for review in tqdm(tagged_train_reviews):
        train_vectors.append(model.infer_vector(review.words))

    # Generating vectors for testing data
    print("Generating test vectors:")
    for review in tqdm(tagged_test_reviews):
        test_vectors.append(model.infer_vector(review.words))

    # Converting the generated vectors into Pandas DataFrames
    train_vectors_df = pd.DataFrame(train_vectors)
    test_vectors_df = pd.DataFrame(test_vectors)

    # Writing the training vectors to a Parquet file
    with tqdm(total=len(train_vectors_df), desc="Writing Train Parquet") as pbar:
        table = pa.Table.from_pandas(train_vectors_df)
        pq.write_table(table, cfg.data.output.path + cfg.data.output.filename[0])
        pbar.update(len(train_vectors_df))
    
    # Writing the testing vectors to a Parquet file
    with tqdm(total=len(test_vectors_df), desc="Writing Test Parquet") as pbar:
        table = pa.Table.from_pandas(test_vectors_df)
        pq.write_table(table, cfg.data.output.path + cfg.data.output.filename[1])
        pbar.update(len(test_vectors_df))

# Entry point of the script
if __name__ == "__main__":
    doc2vec_embedding()
