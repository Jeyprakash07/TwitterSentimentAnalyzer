import re
from sklearn.base import BaseEstimator, TransformerMixin
from app import logger
from consts.consts import TEXT_CLEANING_RE, CUSTOM_NLTK_DATA_PATH
import os

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Custom transformer class to apply text preprocessing to a column
class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom text preprocessing transformer to clean, tokenize, and optionally stem text data.
    """

    def __init__(self):
        """
        Initialize the TextPreprocessor and ensure required NLTK resources are available.
        """
        self._ensure_nltk_resources()
        logger.info("TextPreprocessor initialized successfully.")

    def _get_stemmer(self, lang):
        """
        Get the Snowball stemmer for the specified language.

        Args:
            lang (str): Language for the stemmer.

        Returns:
            SnowballStemmer: The stemmer for the specified language.
        """
        logger.debug(f"Getting SnowballStemmer for language: {lang}")
        return SnowballStemmer(lang)
    
    def _get_stopwords(self, lang):
        """
        Get the stopwords for the specified language.

        Args:
            lang (str): Language for the stopwords.

        Returns:
            list: List of stopwords for the specified language.
        """
        logger.debug(f"Getting stopwords for language: {lang}")
        return stopwords.words(lang)

    def _preprocess_text(self, text, stem):
        """
        Clean, tokenize, and optionally stem the given text.

        Args:
            text (str): Input text to preprocess.
            stem (bool): Whether to apply stemming to the tokens.

        Returns:
            str: Preprocessed text.
        """
        logger.debug("Starting text preprocessing.")
        stop_words = self._get_stopwords("english")
        stemmer = self._get_stemmer("english")

        # Remove links, users, and special characters using the provided regex pattern
        text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
        logger.debug(f"Text after cleaning: {text}")

        tokens = []
        for token in text.split():
            if token not in stop_words:
                if stem:
                    stemmed_token = stemmer.stem(token)
                    tokens.append(stemmed_token)
                    logger.debug(f"Token stemmed: {token} -> {stemmed_token}")
                else:
                    tokens.append(token)
                    logger.debug(f"Token added without stemming: {token}")
        preprocessed_text = " ".join(tokens)
        logger.debug(f"Preprocessed text: {preprocessed_text}")
        return preprocessed_text
    
    def _ensure_nltk_resources(self):
        """
        Ensure necessary NLTK resources are downloaded and available.
        """
        logger.info("Checking for required NLTK resources.")
        resources = ['stopwords']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                logger.debug(f"NLTK resource '{resource}' is already available.")
            except LookupError:
                logger.warning(f"NLTK resource '{resource}' not found. Downloading...")
                nltk.download(resource)
                logger.info(f"Downloaded NLTK resource: {resource}")

        # # Add custom NLTK data path to ensure it is recognized
        # nltk.data.path.append(CUSTOM_NLTK_DATA_PATH)
        # logger.debug(f"Custom NLTK data path added: {CUSTOM_NLTK_DATA_PATH}")
