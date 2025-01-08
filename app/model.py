import tensorflow as tf
from . import logger
from consts.consts import MODEL_PATH, LABEL_ENCODER_PATH, W2V_MODEL_PATH, TOKENIZER_PATH
from .train import train_model
import os
import pickle
import gensim
import keras

class SAModel():    
    def __init__(self):
        self.model = None
        self.w2v_model = None
        self.tokenizer = None
        self.encoder = None

    def initialize_models(self):
        """
        Load the trained model if present otherwise raise relevant exception.
        """
        try:
            self.model = self.load_model()
            self.w2v_model = self.load_w2v_model()
            self.tokenizer = self.load_tokenizer()
            self.encoder = self.load_encoder()
        except FileNotFoundError as e:
            logger.info("Model not found. Starting model training phase.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during model initialization: {e}")
            raise

    def train_and_persist_model(self):
        """
        Train the W2V and LSTM sequential model and save it to disk
        """
        try:
            self.model, self.w2v_model, self.tokenizer, self.encoder = train_model()
        except Exception as e:
            logger.error(f"Unexpected error while training the model: {e}")
            raise
        
        try:
            self.save_model()
            self.save_w2v_model()
            self.save_tokenizer()
            self.save_encoder()
        except Exception as e:
            logger.error(f"Unexpected error while persisting the model: {e}")
            raise


    # Function to load model
    def load_model(self):
        """
        Load the trained model from disk.

        Returns:
            model (LSTM): The trained model.
        """
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        try:
            model = keras.models.load_model(MODEL_PATH, compile=True)
            logger.info("successfully loaded the model")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_w2v_model(self):
        """
        Load the trained w2v model from disk.

        Returns:
            model (Word2Vec): The trained model.
        """
        if not os.path.exists(W2V_MODEL_PATH):
            logger.error(f"W2V Model file not found: {W2V_MODEL_PATH}")
            raise FileNotFoundError(f"W2V Model file not found: {W2V_MODEL_PATH}")

        try:
            model = gensim.models.Word2Vec.load(W2V_MODEL_PATH)
            logger.info("successfully loaded the w2v model")
            return model
        except Exception as e:
            logger.error(f"Error loading w2v model: {e}")
            raise

    def load_tokenizer(self):
        """
        Load the tokenizer from disk.

        Returns:
            tokenizer: Text to token converter.
        """
        if not os.path.exists(TOKENIZER_PATH):
            logger.error(f"Tokenizer file not found: {TOKENIZER_PATH}")
            raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")

        try:
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
            logger.info("successfully loaded tokenizer")
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise

    def load_encoder(self):
        """
        Load the encoder from disk.

        Returns:
            Label encoder: Convert categorical variables into numerical format.
        """
        if not os.path.exists(LABEL_ENCODER_PATH):
            logger.error(f"W2V Model file not found: {LABEL_ENCODER_PATH}")
            raise FileNotFoundError(f"W2V Model file not found: {LABEL_ENCODER_PATH}")

        try:
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                encoder = pickle.load(f)
            logger.info("successfully loaded encoder")
            return encoder
        except Exception as e:
            logger.error(f"Error loading encoder: {e}")
            raise


    def save_model(self):
        """
        Save the trained model to disk.
        """
        try:
            self.model.save(MODEL_PATH)
            logger.info("Model saved.")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def save_w2v_model(self):
        """
        Save the trained Word2Vec model to disk.
        """
        try:
            self.w2v_model.save(W2V_MODEL_PATH)
            logger.info("W2V model saved.")
        except Exception as e:
            logger.error(f"Error saving w2v model: {e}")
            raise

    def save_tokenizer(self):
        """
        Save the tokenizer to disk.
        """
        try:
            pickle.dump(self.tokenizer, open(TOKENIZER_PATH, "wb"), protocol=0)
            logger.info("Tokenizer saved.")
        except Exception as e:
            logger.error(f"Error saving tokenizer {e}")
            raise

    def save_encoder(self):
        """
        Save the Encoder to disk.
        """
        try:
            pickle.dump(self.encoder, open(LABEL_ENCODER_PATH, "wb"), protocol=0)
            logger.info("Encoder saved.")
        except Exception as e:
            logger.error(f"Error saving encoder {e}")
            raise



















