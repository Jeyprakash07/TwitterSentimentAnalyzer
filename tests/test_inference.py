import unittest
from app.inference import decode_sentiment, predict
from consts.consts import POSITIVE, NEUTRAL, NEGATIVE
from unittest.mock import patch, MagicMock
from app.proto.twitter_sentiment_analyzer_pb2 import PredictSentimentRequest
import numpy as np
from consts.errors import ValidationError

class TestInference(unittest.TestCase):
    
    # Test the 'decode_sentiment' function with various scores and expected sentiment labels
    def test_decode_sentiment(self):
        scores = [
            [0.1, True, NEGATIVE], 
            [0.3, False, NEGATIVE], 
            [0.45, True, NEUTRAL], 
            [0.7, True, POSITIVE], 
            [0.499, False, NEGATIVE],
            [0.5, False, POSITIVE],
            [0.8, False, POSITIVE],
            [0.9, True, POSITIVE]]

        # Iterate through the predefined scores to validate the function's output
        for score in scores:
            sentiment = decode_sentiment(score[0], include_neutral=score[1])
            self.assertEqual(score[2], sentiment)

    # Test the 'predict' function by mocking necessary components
    @patch('app.inference.sentiment_analyzer_model')
    @patch('app.inference.pad_sequences', new_callable=MagicMock)
    def test_predict(self, mock_app_inference_pad_sequences, 
                     mock_app_inference_sentiment_analyzer_model):
        """
        Test the 'predict' function by mocking the pad_sequences and sentiment_analyzer_model components.
        The mock values simulate the behavior of these components.
        """
        # Define mock behavior for pad_sequences
        def mock_app_inference_pad_sequences_behavior(sequences, maxlen):
            return np.array([[0, 0, 0, 10, 12, 15]])
        
        # Mock behavior for tokenizer's texts_to_sequences
        def mock_app_inference_sa_model_tokenizer_texts_to_seq_behavior(textArr):
            return None
        
        # Mock behavior for the model's predict function
        def mock_app_inference_sa_model_model_predict_behavior(arr):
            return [0.7]
        
        # Setup the mocks
        mock_app_inference_sentiment_analyzer_model.tokenizer.texts_to_sequences.side_effect = mock_app_inference_sa_model_tokenizer_texts_to_seq_behavior
        mock_app_inference_sentiment_analyzer_model.model.predict = mock_app_inference_sa_model_model_predict_behavior
        mock_app_inference_pad_sequences.side_effect = mock_app_inference_pad_sequences_behavior

        # Create a test request with sample text
        request = PredictSentimentRequest(text="I feel great")
        
        # Call the predict function and assert that it returns the correct sentiment label
        label = predict(request, include_neutral=True)
        self.assertEqual(label, POSITIVE)

    # Test the 'predict' function with additional mocks to ensure all components work as expected
    @patch('app.inference.pad_sequences', new_callable=MagicMock)  # Mock pad sequences
    @patch('app.inference.sentiment_analyzer_model', new_callable=MagicMock)  # Mock tokenizer
    @patch('app.inference.decode_sentiment')  # Mock the decode_sentiment function
    def test_predict_success(self, mock_decode_sentiment, mock_inference_sa_model, mock_inference_pad_sequences):
        """
        Test successful prediction using mocked components for pad_sequences, tokenizer, and decode_sentiment.
        Ensures that the 'predict' function works correctly with these mocked components.
        """
        # Setup mock values for the request and return values for various components
        request = MagicMock()
        request.text = "This is a positive review"

        # Mock the return values of decode_sentiment, tokenizer, pad_sequences, and model prediction
        mock_decode_sentiment.return_value = POSITIVE  # Mock sentiment decoding
        mock_inference_sa_model.tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
        mock_inference_pad_sequences.return_value = [[0, 0, 1, 2, 4]]
        mock_inference_sa_model.model.predict.return_value = [0.7]

        # Call the predict function with the mocked request
        result = predict(request, include_neutral=True)

        # Assertions to ensure the correct function calls and values
        mock_inference_sa_model.tokenizer.texts_to_sequences.assert_called_once()  # Ensure tokenizer was called
        mock_inference_pad_sequences.assert_called_once()  # Ensure pad_sequences was called
        mock_inference_sa_model.model.predict.assert_called_once_with([[[0, 0, 1, 2, 4]]])
        mock_decode_sentiment.assert_called_once_with(0.7, include_neutral=True)  # Ensure decode_sentiment was called with correct arguments

        # Assert the result returned by the predict function is correct
        self.assertEqual(result, POSITIVE)

    # Test failure case for 'predict' when the request does not have text input
    def test_predict_failure(self):
        """
        Test the 'predict' function to ensure it raises a ValidationError when no text input is provided.
        """
        # Setup a request with no text
        request = MagicMock()
        request.text = None

        # Assert that the predict function raises a ValidationError with the expected message
        with self.assertRaises(ValidationError) as context:
            predict(request, include_neutral=True)

        # Check that the exception message is correct
        self.assertEqual(str(context.exception), "Text input is not found in request") 
