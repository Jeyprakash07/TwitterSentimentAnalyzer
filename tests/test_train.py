import unittest
from app.train import train_model, get_train_data, build_w2v_model, encode_and_reshape_result_labels, tokenize_inputs, build_embed_layer, create_and_fit_model, decode_sentiment
from unittest.mock import patch, MagicMock
import pandas as pd
from keras.models import Sequential
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from consts.consts import DATASET_ENCODING, DATASET_COLUMNS, POSITIVE, NEGATIVE, TRAIN_SIZE, W2V_SIZE
import numpy as np
from sklearn.model_selection import train_test_split
from data.preprocessing import TextPreprocessor
import kagglehub

# Mocked behavior of kagglehub.load_dataset to simulate loading dataset from kaggle
def mock_kagglehub_load_dataset_behavior(adapter=None, handle=None , path=None, pandas_kwargs=None):
    data = {
        "target": ['0', '4', '0', '4', '0'],
        "ids": ["1468000743", "1468000744", "1468000745", "1468000746", "1468000747"],
        "date": ["Mon Apr 06 23:11:53 PDT 2009", "Mon Apr 06 23:11:53 PDT 2010", "Mon Apr 06 23:11:53 PDT 2011", "Mon Apr 06 23:11:53 PDT 2012", "Mon Apr 06 23:11:53 PDT 2013"],
        "flag": ["A", "B", "C", "A", "B"],
        "user": ["user1", "user2", "user3", "user4", "user5"],
        "text": [
            "This is a sample text for the first entry.",
            "Another text here with different content.",
            "Yet another sample for testing purposes.",
            "Text example number four, with more data.",
            "The fifth and final entry in this dataset."
        ]
    }

    df = pd.DataFrame(data)
    return df

class TestTrain(unittest.TestCase):

    @patch('app.train.kagglehub.load_dataset', new_callable=MagicMock())  # Mocking kagglehub.load_dataset
    def test_get_train_data(self, mock_kagglehub_load_dataset):
        # Mock the behavior of loading data from kaggle
        mock_kagglehub_load_dataset.side_effect = mock_kagglehub_load_dataset_behavior

        # Test case 1: Verifying the data returned by get_train_data
        df = get_train_data()

        self.assertEqual(df.loc[0, 'target'], NEGATIVE)  # Ensure correct target mapping
        self.assertEqual(df.loc[1, 'target'], POSITIVE)  
        self.assertEqual(df.loc[2, 'target'], NEGATIVE)  
        self.assertEqual(df.loc[3, 'target'], POSITIVE)  
        self.assertEqual(df.loc[4, 'target'], NEGATIVE)  

        self.assertEqual(df.loc[0, 'text'], 'sample text first entry')  # Ensure correct text processing
        self.assertEqual(df.loc[1, 'text'], 'another text different content')  
        self.assertEqual(df.loc[2, 'text'], 'yet another sample testing purposes')  
        self.assertEqual(df.loc[3, 'text'], 'text example number four data')  
        self.assertEqual(df.loc[4, 'text'], 'fifth final entry dataset')  

        # Test case 2: Handle file read error gracefully
        mock_kagglehub_load_dataset.side_effect = Exception('Unknown')
        with self.assertRaises(Exception) as context1:
            df = get_train_data()
        self.assertEqual(str(context1.exception), "Unknown")

        # Test case 3: Handle FileNotFoundError
        mock_kagglehub_load_dataset.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            df = get_train_data()

    def test_build_w2v_model(self):
        # Test Word2Vec model building
        df = get_train_data(reduce=0.01)
        
        w2v_model = build_w2v_model(df)
        self.assertEqual(hasattr(w2v_model, 'wv'), True)  # Ensure model has word vectors
        self.assertEqual(len(w2v_model.wv) > 0, True)  # Ensure word vectors are non-empty
        self.assertEqual(len(w2v_model.wv.index_to_key) > 0, True)  # Ensure word index is populated
        self.assertEqual(hasattr(w2v_model, 'vector_size'), True)  # Ensure vector size exists
        self.assertNotEqual(w2v_model.wv['love'].size, 0)  # Ensure 'love' word vector exists

    @patch('app.train.kagglehub.load_dataset', new_callable=MagicMock())  # Mocking kagglehub.load_dataset
    def test_encode_and_reshape_result_labels(self, mock_kagglehub_load_dataset):
        # Test encoding and reshaping labels for training
        mock_kagglehub_load_dataset.side_effect = mock_kagglehub_load_dataset_behavior
        df = get_train_data()
        df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)

        y_train, y_test, encoder = encode_and_reshape_result_labels(df_train, df_test)
        self.assertNotEqual(len(y_train), 0)  # Ensure y_train is non-empty
        self.assertNotEqual(len(y_train[0]), 0)  # Ensure y_train entries are non-empty
        self.assertNotEqual(len(y_test), 0)  # Ensure y_test is non-empty
        self.assertNotEqual(len(y_test[0]), 0)  # Ensure y_test entries are non-empty
        self.assertEqual(type(encoder), LabelEncoder)  # Ensure encoder is of LabelEncoder type

    @patch('app.train.kagglehub.load_dataset', new_callable=MagicMock())  # Mocking kagglehub.load_dataset
    def test_tokenize_inputs(self, mock_kagglehub_load_dataset):
        # Test tokenization of input text
        mock_kagglehub_load_dataset.side_effect = mock_kagglehub_load_dataset_behavior
        df = get_train_data()
        df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)

        x_train, x_test, tokenizer = tokenize_inputs(df_train, df_test)
        self.assertNotEqual(len(x_train), 0)  # Ensure x_train is non-empty
        self.assertNotEqual(len(x_train[0]), 0)  # Ensure x_train entries are non-empty
        self.assertNotEqual(len(x_test), 0)  # Ensure x_test is non-empty
        self.assertNotEqual(len(x_test[0]), 0)  # Ensure x_test entries are non-empty
        self.assertEqual(type(tokenizer), Tokenizer)  # Ensure tokenizer is of Tokenizer type

    def test_build_embed_layer(self):
        # Test embedding layer creation
        df = get_train_data(reduce=0.01)
        df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)
        w2v_model = build_w2v_model(df_train)

        _, _, tokenizer = tokenize_inputs(df_train, df_test)
        vocab_size = len(tokenizer.word_index) + 1  # Get vocab size from tokenizer

        embedding_layer = build_embed_layer(w2v_model, tokenizer, vocab_size)
        self.assertEqual(embedding_layer.built, True)  # Ensure embedding layer is built
        self.assertEqual(embedding_layer.input_dim, vocab_size)  # Ensure input dimension matches vocab size
        self.assertEqual(embedding_layer.output_dim, W2V_SIZE)  # Ensure output dimension matches word2vec size

    def test_create_and_fit_model(self):
        # Test model creation and fitting
        df = get_train_data(reduce=0.001)
        df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)

        w2v_model = build_w2v_model(df_train)
        y_train, y_test, encoder = encode_and_reshape_result_labels(df_train, df_test)
        x_train, x_test, tokenizer = tokenize_inputs(df_train, df_test)

        vocab_size = len(tokenizer.word_index) + 1
        embedding_layer = build_embed_layer(w2v_model, tokenizer, vocab_size)
        model = create_and_fit_model(x_train, y_train, embedding_layer)

        self.assertEqual(model.input_shape[1:], (300,))  # Ensure input shape is (300,)
        self.assertEqual(model.output_shape[1:], (1,))  # Ensure output shape is (1,)
        self.assertEqual(model.optimizer.get_config()['name'], 'adam')  # Ensure optimizer is 'adam'
        self.assertEqual(model.loss, 'binary_crossentropy')  # Ensure loss function is binary_crossentropy
        self.assertEqual(model.get_config()['name'], 'sequential')  # Ensure model type is sequential

    @patch('app.train.get_train_data', new_callable=MagicMock())  # Mocking get_train_data
    def test_train_model(self, mock_get_train_data):
        # Test the entire training pipeline
        def mock_get_train_data_behavior(reduce):
            df = kagglehub.load_dataset(
                    adapter=kagglehub.KaggleDatasetAdapter.PANDAS,
                    handle='kazanova/sentiment140',
                    path='training.1600000.processed.noemoticon.csv',
                    pandas_kwargs={"encoding": DATASET_ENCODING, "names": DATASET_COLUMNS})

            df = df.sample(frac=0.001, random_state=42)
            df.target = df.target.apply(lambda x: decode_sentiment(x))

            preprocessor = TextPreprocessor()
            df.text = df.text.apply(lambda x: preprocessor._preprocess_text(x, False))
            return df
        
        mock_get_train_data.side_effect = mock_get_train_data_behavior

        model, w2v_model, tokenizer, encoder = train_model()
        self.assertEqual(model.get_config()['name'], "sequential")  # Ensure model is of type 'sequential'
        self.assertEqual(type(w2v_model), gensim.models.word2vec.Word2Vec)  # Ensure Word2Vec model is returned
        self.assertEqual(type(tokenizer), Tokenizer)  # Ensure Tokenizer is returned
        self.assertEqual(type(encoder), LabelEncoder)  # Ensure LabelEncoder is returned
