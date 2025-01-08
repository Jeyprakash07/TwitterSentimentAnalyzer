import pandas as pd
from sklearn.preprocessing import LabelEncoder
from consts.consts import (
    TRAINING_DATA_PATH, DATASET_ENCODING, DATASET_COLUMNS, LABEL_MAP,
    TRAIN_SIZE, W2V_MIN_COUNT, W2V_SIZE, W2V_WINDOW, W2V_EPOCH, SEQUENCE_LENGTH,
    NEUTRAL, BATCH_SIZE, EPOCHS
)
from sklearn.model_selection import train_test_split
from .evaluate import evaluate_model

from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from data.preprocessing import TextPreprocessor

import gensim
import tensorflow as tf
import kagglehub
from . import logger

# Decode sentiment label from integer to string using LABEL_MAP
def decode_sentiment(label):
    """
    Decodes an integer label to its corresponding sentiment string.

    Args:
        label (int): Encoded sentiment label.

    Returns:
        str: Decoded sentiment label.
    """
    return LABEL_MAP[int(label)]

# Train the sentiment analysis model
def train_model():
    """
    Train the model using training data, build the Word2Vec model, 
    tokenize inputs, and evaluate the trained model.

    Returns:
        tuple: Trained model, Word2Vec model, tokenizer, and label encoder.
    """
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices) > 0:
        logger.info(f"GPUs found: {len(physical_devices)}")
    else:
        logger.info("No GPU found, TensorFlow will use the CPU.")

    try:
        df = get_train_data(reduce=0.01)
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

    # Split data into training and testing sets
    df_train, df_test = train_test_split(df, test_size=1 - TRAIN_SIZE, random_state=42)
    logger.info(f"Training data size: {len(df_train)}")
    logger.info(f"Testing data size: {len(df_test)}")

    # Build Word2Vec model
    w2v_model = build_w2v_model(df_train)

    # Encode and reshape result labels
    y_train, y_test, encoder = encode_and_reshape_result_labels(df_train, df_test)

    # Tokenize input data
    x_train, x_test, tokenizer = tokenize_inputs(df_train, df_test)
    vocab_size = len(tokenizer.word_index) + 1
    logger.info(f"Vocabulary size: {vocab_size}")

    # Build embedding layer
    embedding_layer = build_embed_layer(w2v_model, tokenizer, vocab_size)

    # Create, compile, and fit the model
    model = create_and_fit_model(x_train, y_train, embedding_layer)

    # Evaluate the trained model
    evaluate_model(model, x_test, y_test)

    return model, w2v_model, tokenizer, encoder

# Load and preprocess training data
def get_train_data(reduce=0):
    """
    Loads training data, applies optional reduction, and preprocesses text.

    Args:
        reduce (float): Fraction of the data to sample (for reducing dataset size).

    Returns:
        pd.DataFrame: Preprocessed training data.
    """
    try:
        df = kagglehub.load_dataset(
            adapter=kagglehub.KaggleDatasetAdapter.PANDAS,
            handle='kazanova/sentiment140',
            path='training.1600000.processed.noemoticon.csv',
            pandas_kwargs={"encoding": DATASET_ENCODING, "names": DATASET_COLUMNS})
        
        # df = pd.read_csv(TRAINING_DATA_PATH, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
        logger.info("Training data loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Training data not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise

    if reduce != 0:
        df = df.sample(frac=reduce, random_state=42)
        logger.info(f"Training data reduced to {reduce * 100}% of original size.")

    # Decode sentiment labels
    df.target = df.target.apply(lambda x: decode_sentiment(x))

    # Preprocess text
    preprocessor = TextPreprocessor()
    df.text = df.text.apply(lambda x: preprocessor._preprocess_text(x, False))
    logger.info("Training data preprocessed successfully.")

    return df

# Build Word2Vec model
def build_w2v_model(df_train: pd.DataFrame):
    """
    Builds a Word2Vec model from training data.

    Args:
        df_train (pd.DataFrame): Training data.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    documents = [_text.split() for _text in df_train.text]

    w2v_model = gensim.models.word2vec.Word2Vec(
        vector_size=W2V_SIZE, window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT, workers=4
    )
    w2v_model.build_vocab(documents)
    logger.info(f"Word2Vec vocabulary size: {len(w2v_model.wv.index_to_key)}")

    w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
    logger.info("Word2Vec model training completed.")

    return w2v_model

# Encode labels and reshape them for model compatibility
def encode_and_reshape_result_labels(df_train, df_test):
    """
    Encodes and reshapes sentiment labels for training and testing.

    Args:
        df_train (pd.DataFrame): Training data.
        df_test (pd.DataFrame): Testing data.

    Returns:
        tuple: Encoded training labels, testing labels, and the label encoder.
    """
    encoder = LabelEncoder()
    encoder.fit(df_train.target.tolist())

    y_train = encoder.transform(df_train.target.tolist()).reshape(-1, 1)
    y_test = encoder.transform(df_test.target.tolist()).reshape(-1, 1)

    logger.info(f"Training labels shape: {y_train.shape}")
    logger.info(f"Testing labels shape: {y_test.shape}")

    return y_train, y_test, encoder

# Tokenize input text data
def tokenize_inputs(df_train, df_test):
    """
    Tokenizes input text data for model training.

    Args:
        df_train (pd.DataFrame): Training data.
        df_test (pd.DataFrame): Testing data.

    Returns:
        tuple: Tokenized training data, testing data, and tokenizer object.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train.text)

    x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
    x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)
    logger.info("Input data tokenized successfully.")

    return x_train, x_test, tokenizer

# Build embedding layer from Word2Vec model
def build_embed_layer(w2v_model, tokenizer, vocab_size):
    """
    Builds an embedding layer using the Word2Vec model.

    Args:
        w2v_model: Word2Vec model.
        tokenizer: Tokenizer object.
        vocab_size (int): Size of the vocabulary.

    Returns:
        Embedding: Keras embedding layer.
    """
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")

    return Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], trainable=False)

# Create and train the model
def create_and_fit_model(x_train, y_train, embedding_layer):
    """
    Creates, compiles, and trains a sentiment analysis model.

    Args:
        x_train: Tokenized training data.
        y_train: Encoded training labels.
        embedding_layer: Pre-built embedding layer.

    Returns:
        Sequential: Trained Keras model.
    """
    model = Sequential(name="sequential")
    model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.build(input_shape=(BATCH_SIZE, SEQUENCE_LENGTH))
    model.summary()

    model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
    
    logger.info("Model architecture created and compiled.")

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
        EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)
    ]

    history = model.fit(
        x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_split=0.1, verbose=1, callbacks=callbacks
    )
    logger.info("Model training completed.")

    return model
