# Importing necessary libraries
import gensim
import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from hydra.utils import instantiate, to_absolute_path
from nltk.tokenize import word_tokenize
from omegaconf import DictConfig
from tqdm import tqdm

# Main function for FastText embedding
@hydra.main(version_base=None, config_path="../config", config_name="fasttext")
def fasttext_embedding(cfg: DictConfig):

    # Getting absolute path to the data file
    file_path = to_absolute_path(cfg.data.path)
    
    # Reading training and testing data from Parquet files into Pandas DataFrames
    df_train = pq.read_table(file_path + '\\' + cfg.data.filename[0]).to_pandas()
    df_test = pq.read_table(file_path + '\\' + cfg.data.filename[1]).to_pandas()

    # Tokenizing reviews in training and testing data
    train_tokenized_reviews = [word_tokenize(review) for review in df_train['review']]
    test_tokenized_reviews = [word_tokenize(review) for review in df_test['review']]

    # Instantiating the FastText model using the provided configuration
    model = instantiate(cfg.model)
    
    # Building vocabulary for the model using both training and testing data
    model.build_vocab(train_tokenized_reviews + test_tokenized_reviews)
    
    # Training the FastText model
    model.train(train_tokenized_reviews + test_tokenized_reviews, total_examples=model.corpus_count, epochs=10)

    # Function to get FastText embeddings for a sentence
    def get_embedding(sentence):
        tokens = word_tokenize(sentence)
        return model.wv[tokens].mean(axis=0) if tokens else None

    # Initializing lists to store vectors for training and testing data
    train_vectors = []
    test_vectors = []

    # Generating vectors for training data
    print("Generating train vectors:")
    for review in tqdm(df_train['review']):
        train_vectors.append(get_embedding(review))

    # Generating vectors for testing data
    print("Generating test vectors:")
    for review in tqdm(df_test['review']):
        test_vectors.append(get_embedding(review))

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
    fasttext_embedding()
