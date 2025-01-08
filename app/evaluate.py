from consts.consts import BATCH_SIZE
# Matplot
from . import logger

def evaluate_model(model, x_test, y_test):
    
    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    logger.info(f"Model - ACCURACY: {score[1]}")
    logger.info(f"Model - LOSS: {score[0]}")

    #TODO: tensorflow/MLflow serving for evaluation
