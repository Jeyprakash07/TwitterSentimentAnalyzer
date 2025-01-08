import unittest
from unittest.mock import patch, mock_open, MagicMock
from app.model import SAModel
from consts.consts import MODEL_PATH, W2V_MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH
import gensim
from keras.models import Sequential

class TestSAModel(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open)  # Mock open for file reading
    @patch('pickle.load', new_callable=MagicMock)  # Mock pickle.load
    @patch('keras.models.load_model', new_callable=MagicMock)  # Mock loading Keras models
    @patch('gensim.models.Word2Vec.load', new_callable=MagicMock)  # Mock loading Word2Vec model
    def test_initialize_model(self, 
                                mock_gensim_models_Word2Vec_load, 
                                mock_keras_models_load_model,
                                mock_pickle_load, mocked_open):
        """
        Test the initialization of SAModel, ensuring models and objects are correctly loaded.
        Mocks the loading of models, tokenizer, and encoder.
        """
        sa_model = SAModel()

        mock_model = "I'm the trained model"
        mock_keras_models_load_model.return_value = "I'm the trained model"

        mock_w2v_model = "I'm the trained W2V model"
        mock_gensim_models_Word2Vec_load.return_value = "I'm the trained W2V model"

        tokenizer = mock_open(read_data="mocked content for tokenizer").return_value
        label_encoder = mock_open(read_data="mocked content for label encoder").return_value

        def mock_open_side_effect(file_path, mode='r', *args, **kwargs):
            if file_path == TOKENIZER_PATH:
                return tokenizer
            elif file_path == LABEL_ENCODER_PATH:
                return label_encoder

        def mock_pickle_load_behavior(input):
            if input == tokenizer:
                return tokenizer.read()
            elif input == label_encoder:
                return label_encoder.read()

        mocked_open.side_effect  = mock_open_side_effect
        mock_pickle_load.side_effect = mock_pickle_load_behavior

        self.assertEqual(sa_model.model, None)  # Ensure model is not loaded initially
        self.assertEqual(sa_model.w2v_model, None)  # Ensure W2V model is not loaded initially
        self.assertEqual(sa_model.tokenizer, None)  # Ensure tokenizer is not loaded initially
        self.assertEqual(sa_model.encoder, None)  # Ensure encoder is not loaded initially

        # Initialize models and objects
        sa_model.initialize_models()

        # Check if the models and objects were correctly loaded
        self.assertEqual(sa_model.model, mock_model)
        self.assertEqual(sa_model.w2v_model, mock_w2v_model)
        self.assertEqual(sa_model.tokenizer, "mocked content for tokenizer")
        self.assertEqual(sa_model.encoder, "mocked content for label encoder")


    @patch('keras.models.load_model', side_effect=Exception("load model failed"))
    @patch('gensim.models.Word2Vec.load', side_effect=Exception("load w2v model failed"))
    @patch('pickle.load', side_effect=Exception("failed to load with pickle"))
    def test_load_model_exception(self, mock_pickle_load, 
                                  mock_gensim_models_Word2Vec_model, 
                                  mock_keras_models_load_model):
        """
        Test to raise exception on loading trained ML models. This includes checking failures for Word2Vec model as well.
        """
        sa_model = SAModel()

        with self.assertRaises(Exception) as context1:
            model = sa_model.load_model()

        print(str(context1.exception))
        self.assertEqual(str(context1.exception), "load model failed")

        with self.assertRaises(Exception) as context2:
            w2v_model = sa_model.load_w2v_model()

        self.assertEqual(str(context2.exception), "load w2v model failed")

        with self.assertRaises(Exception) as context3:
            tokenizer = sa_model.load_tokenizer()

        self.assertEqual(str(context3.exception), "failed to load with pickle")

        with self.assertRaises(Exception) as context4:
            encoder = sa_model.load_encoder()

        self.assertEqual(str(context4.exception), "failed to load with pickle")


    @patch('os.path.exists', new_callable=MagicMock)
    def test_load_model_file_not_found_exception(self, mock_os_path_exists):
        """
        Test to raise FileNotFoundError on model load functions if the model files do not exist.
        Applicable for Word2Vec model, tokenizer, and label encoder.
        """
        sa_model = SAModel()
        
        mock_os_path_exists.return_value = False  # Simulate that the files don't exist

        # Test if FileNotFoundError is raised for each model, tokenizer, and encoder
        with self.assertRaises(FileNotFoundError):
            model = sa_model.load_model()

        with self.assertRaises(FileNotFoundError):
            w2v_model = sa_model.load_w2v_model()

        with self.assertRaises(FileNotFoundError):
            tokenizer = sa_model.load_tokenizer()

        with self.assertRaises(FileNotFoundError):
            encoder = sa_model.load_encoder()
    
    @patch('app.model.train_model', new_callable=MagicMock)
    @patch('builtins.open', new_callable=MagicMock)  # Mock 'open' function
    @patch('pickle.dump')  # Mock 'pickle.dump'
    @patch('gensim.models.word2vec.Word2Vec.save', new_callable=MagicMock)
    @patch.object(Sequential, 'save')  # Mock Keras model save
    def test_save_models_and_pickling_objects(self, 
                                              mock_keras_seq_model, 
                                              mock_gensim_w2v_model_save,
                                              mock_pickle_dump, mock_open, mock_train_model):
        """
        Test to ensure models and objects are saved properly, including pickling the tokenizer and encoder.
        """
        sa_model = SAModel()

        def create_model():
            return Sequential()

        mock_model = create_model()
        mock_w2v_model = gensim.models.word2vec.Word2Vec()
        mock_tokenizer = "I'm the tokenizer"
        mock_encoder = "I'm the encoder"

        def mock_train():
            return mock_model, mock_w2v_model, mock_tokenizer, mock_encoder
        
        mock_train_model.side_effect = mock_train

        mock_file = mock_open(read_data="").return_value
        mock_open.return_value = mock_file

        # Train the model and persist
        sa_model.train_and_persist_model()

        # Check if the model and W2V model are saved correctly
        mock_keras_seq_model.assert_called_once_with(MODEL_PATH)
        mock_gensim_w2v_model_save.assert_called_once_with(W2V_MODEL_PATH)

        # Check if tokenizer and encoder are pickled correctly
        mock_open.assert_any_call(TOKENIZER_PATH, "wb")
        mock_pickle_dump.assert_any_call(mock_tokenizer, mock_file, protocol=0)

        mock_open.assert_called_with(LABEL_ENCODER_PATH, "wb")
        mock_pickle_dump.assert_called_with(mock_encoder, mock_file, protocol=0)

    @patch('pickle.dump', side_effect=Exception("Failed to dump file"))  # Mock 'pickle.dump'
    @patch('gensim.models.word2vec.Word2Vec.save', side_effect=Exception("Failed to save W2V model file"))
    @patch.object(Sequential, 'save')
    def test_save_models_and_pickling_objects_exception(self, 
                                                        mock_keras_model_save, 
                                                        mock_gensim_model_save, 
                                                        mock_pickle_dump):
        """
        Test to raise exception during saving models and pickling objects, ensuring exceptions are handled properly.
        """
        sa_model = SAModel()
        sa_model.model = Sequential()

        mock_keras_model_save.side_effect = Exception("Failed to save model")

        # Test saving model failure
        with self.assertRaises(Exception) as context1:
            sa_model.save_model()

        self.assertEqual(str(context1.exception), "Failed to save model")

        sa_model.w2v_model = gensim.models.word2vec.Word2Vec()

        # Test saving W2V model failure
        with self.assertRaises(Exception) as context2:
            sa_model.save_w2v_model()

        self.assertEqual(str(context2.exception), "Failed to save W2V model file")

        sa_model.tokenizer = MagicMock()
        
        # Test pickling tokenizer failure
        with self.assertRaises(Exception) as context3:
            sa_model.save_tokenizer()

        self.assertEqual(str(context3.exception), "Failed to dump file")

        sa_model.encoder = MagicMock()
        
        # Test pickling encoder failure
        with self.assertRaises(Exception) as context4:
            sa_model.save_encoder()

        self.assertEqual(str(context4.exception), "Failed to dump file")
