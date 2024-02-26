# Importing necessary libraries
import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np
import nltk

# Downloading NLTK data
nltk.download('punkt')

# Main function for Word2Vec embedding
@hydra.main(version_base=None, config_path="../config", config_name="w2v_config")
def word2vec_embedding(cfg: DictConfig):

    # Getting absolute path to the data file
    file_path = to_absolute_path(cfg.data.path)
    
    # Reading training and testing data from Parquet files into Pandas DataFrames
    df_train = pq.read_table(file_path + '\\' + cfg.data.filename[0]).to_pandas()
    df_test = pq.read_table(file_path + '\\' + cfg.data.filename[1]).to_pandas()

    # Extracting text reviews from DataFrames and converting to lists
    train_reviews = df_train[cfg.data.column].tolist()
    test_reviews = df_test[cfg.data.column].tolist()

    # Tokenizing sentences in training and testing data
    train_tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in train_reviews]
    test_tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in test_reviews]

    # Instantiating the Word2Vec model using the provided configuration
    model = instantiate(cfg.model)
    
    # Building vocabulary for the model using training data
    model.build_vocab(train_tokenized_sentences)
    
    # Training the Word2Vec model
    model.train(train_tokenized_sentences, total_examples=len(train_tokenized_sentences), epochs=10)

    # Function to calculate sentence embeddings by averaging word vectors
    def get_sentence_embedding(sentence, model):
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if len(word_vectors) == 0:
            return np.zeros(model.vector_size)  # return zero vector if no word vectors found
        return np.mean(word_vectors, axis=0)

    # Initializing lists to store vectors for training and testing data
    train_vectors = []
    test_vectors = []

    # Generating vectors for training data
    print("Generating train vectors:")
    for review in tqdm(train_tokenized_sentences):
        train_vectors.append(get_sentence_embedding(review, model))

    # Generating vectors for testing data
    print("Generating test vectors:")
    for review in tqdm(test_tokenized_sentences):
        test_vectors.append(get_sentence_embedding(review, model))

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
    word2vec_embedding()
