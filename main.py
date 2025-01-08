# main.py

from app import logger
from app.model import SAModel
from app.server import serve
from app.inference import init_inference

def main():
    """
    Entry point for the application. Initializes the sentiment analysis model and starts the server.
    """
    logger.info("Starting application...")

    # Instantiate the sentiment analysis model
    logger.info("Initializing sentiment analysis model.")
    sa_model = SAModel()
    try:
        # Attempt to load pre-trained models
        logger.info("Attempting to load pre-trained models.")
        sa_model.initialize_models()
        logger.info("Models loaded successfully.")
    except FileNotFoundError as e:
        # Train and persist the model if no pre-trained model is found
        logger.warning(f"Pre-trained models not found: {e}. Training a new model...")
        sa_model.train_and_persist_model()
        logger.info("New model trained and persisted successfully.")
    except Exception as e:
        # Log any unexpected exceptions and terminate the program
        logger.error(f"Unexpected error during model initialization: {e}")
        exit(1)

    # Initialize the inference module with the trained or loaded model
    logger.info("Initializing inference module with the sentiment analysis model.")
    init_inference(sa_model)

    # Start the server to serve predictions
    logger.info("Starting the server.")
    serve()
    logger.info("Application is up and running.")

if __name__ == '__main__':
    main()
