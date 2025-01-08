import time

from consts.consts import POSITIVE, NEUTRAL, NEGATIVE, SENTIMENT_THRESHOLDS, SEQUENCE_LENGTH
from keras.preprocessing.sequence import pad_sequences
from consts.errors import ValidationError
from . import logger

# Global variable to hold the sentiment analysis model
sa_model = None

# Initialize the inference model
def init_inference(model):
    """
    Initializes the global sentiment analysis model.

    Args:
        model: The pre-trained sentiment analysis model.
    """
    global sa_model
    sa_model = model
    logger.info("Sentiment analysis model initialized successfully.")

# Decode the sentiment score into a label
def decode_sentiment(score, include_neutral=True):
    """
    Decodes the sentiment score into a sentiment label.

    Args:
        score (float): The sentiment score.
        include_neutral (bool): Whether to include neutral sentiment in decoding.

    Returns:
        str: Sentiment label (POSITIVE, NEUTRAL, or NEGATIVE).
    """
    if include_neutral:
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        logger.info(f"Score {score} decoded to label '{label}' with neutral inclusion.")
        return label
    else:
        label = NEGATIVE if score < 0.5 else POSITIVE
        logger.info(f"Score {score} decoded to label '{label}' without neutral inclusion.")
        return label

# Perform sentiment prediction
def predict(request, include_neutral=True):
    """
    Performs sentiment prediction based on the input request.

    Args:
        request: The request object containing the text input.
        include_neutral (bool): Whether to include neutral sentiment in prediction.

    Returns:
        str: Predicted sentiment label.
    
    Raises:
        ValidationError: If the text input is missing in the request.
    """
    if request.text == None:
        logger.error("Validation error: Text input is not found in the request.")
        raise ValidationError("Text input is not found in request")

    start_at = time.time()
    logger.info(f"Starting inference for request: {request}")
    
    # Tokenize text
    x_predict = pad_sequences(sa_model.tokenizer.texts_to_sequences([request.text]), maxlen=SEQUENCE_LENGTH)
    logger.info(f"Text tokenized successfully: {request.text}")

    # Predict
    score = sa_model.model.predict([x_predict])[0]
    logger.info(f"Prediction score computed: {float(score)}")

    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)
    elapsed_time = time.time() - start_at
    logger.info(f"Inference completed. Label: {label}, Score: {float(score)}, Elapsed time: {elapsed_time:.4f} seconds")

    return label
