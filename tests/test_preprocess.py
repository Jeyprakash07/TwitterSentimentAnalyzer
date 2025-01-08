import unittest
from data.preprocessing import TextPreprocessor
from unittest.mock import patch, MagicMock
from consts.consts import CUSTOM_NLTK_DATA_PATH, TEXT_CLEANING_RE

class TestTextPreprocessor(unittest.TestCase):

    @patch('nltk.data.find')  # Mock nltk.data.find to simulate finding NLTK resources
    @patch('nltk.download')  # Mock nltk.download to simulate downloading NLTK resources
    @patch('nltk.data.path', new_callable=list)  # Mock the list of paths where NLTK data is stored
    def test_ensure_nltk_resources_download(self, mock_data_path, mock_download, mock_find):
        # Test case for when the resource is NOT found, so it should trigger a download.

        mock_find.side_effect = LookupError  # Simulate resource not found

        # Instantiate the TextPreprocessor object
        obj = TextPreprocessor()

        # Check if 'find' and 'download' were called correctly
        mock_find.assert_called_once_with('tokenizers/stopwords')  # Ensure find was called for the stopwords resource
        mock_download.assert_called_once_with('stopwords', download_dir=CUSTOM_NLTK_DATA_PATH)  # Ensure download was called for the missing resource
        self.assertIn(CUSTOM_NLTK_DATA_PATH, mock_data_path)  # Ensure the NLTK data path was included

    @patch('nltk.data.find')  # Mock nltk.data.find
    @patch('nltk.download')  # Mock nltk.download
    @patch('nltk.data.path', new_callable=list)  # Mock the list of paths where NLTK data is stored
    def test_ensure_nltk_resources_already_present(self, mock_data_path, mock_download, mock_find):
        # Test case for when the resource is already present, so no download should occur.

        mock_find.side_effect = None  # Simulate resource found, so no exception is raised

        # Instantiate the TextPreprocessor object
        obj = TextPreprocessor()

        # Check if 'find' was called and 'download' was not called
        mock_find.assert_called_once_with('tokenizers/stopwords')  # Ensure find was called for the stopwords resource
        mock_download.assert_not_called()  # Ensure download was NOT called
        self.assertIn(CUSTOM_NLTK_DATA_PATH, mock_data_path)  # Ensure the NLTK data path was included

    @patch('nltk.data.find')  # Mock nltk.data.find
    @patch('re.sub')  # Mock re.sub for text cleaning (TEXT_CLEANING_RE)
    def test_preprocess_text(self, mock_re_sub, mock_nltk_find):
        # Test case to ensure text is processed correctly with stopword removal and stemming

        mock_nltk_find.side_effect = None  # Simulate the resource is found

        # Mock re.sub (TEXT_CLEANING_RE) to simulate the cleaned text
        mock_re_sub.return_value = 'the quick brown fox jumps over the lazy dog'

        # Instantiate the TextPreprocessor object
        obj = TextPreprocessor()

        with patch.object(obj, '_get_stopwords', return_value=['the', 'is', 'in', 'a']) as mock_get_stopwords:
            with patch.object(obj, '_get_stemmer', new_callable=MagicMock) as mock_get_stemmer:

                def mock_stem_behavior(token):
                    if token == 'jumps':
                        return 'jump'  # Stem the word 'jumps' to 'jump'
                    return token

                mock_get_stemmer.return_value.stem.side_effect = mock_stem_behavior
                # Test case 1: When stem is True
                text = "The quick brown fox jumps over the lazy dog"
                stem = True
                result = obj._preprocess_text(text, stem)

                # Expected output after stopword removal and stemming
                expected_result = "quick brown fox jump over lazy dog"
                self.assertEqual(result, expected_result)

                # Test case 2: When stem is False
                text = "The quick brown fox jumps over the lazy dog"
                stem = False
                result = obj._preprocess_text(text, stem)

                # Expected output after stopword removal without stemming
                expected_result = "quick brown fox jumps over lazy dog"
                self.assertEqual(result, expected_result)

                # Ensure the proper methods were called
                mock_get_stopwords.assert_called_with("english")  # Ensure stopwords were fetched
                mock_get_stemmer.assert_called_with("english")  # Ensure stemmer was initialized

                # Ensure re.sub was called to clean the text
                mock_re_sub.assert_called_with(TEXT_CLEANING_RE, ' ', str(text).lower())

    @patch('nltk.data.find')  # Mock nltk.data.find
    @patch('re.sub')  # Mock re.sub for text cleaning (TEXT_CLEANING_RE)
    def test_preprocess_text_with_special_characters(self, mock_re_sub, mock_nltk_find):
        # Test case for when the text contains special characters and needs to be cleaned.

        mock_nltk_find.side_effect = None  # Simulate the resource is found

        # Mock re.sub to simulate the cleaned text
        mock_re_sub.return_value = 'the quick brown fox'

        # Instantiate the TextPreprocessor object
        obj = TextPreprocessor()

        with patch.object(obj, '_get_stopwords', return_value=['the', 'is', 'in', 'a']) as mock_get_stopwords:
            with patch.object(obj, '_get_stemmer', new_callable=MagicMock) as mock_get_stemmer:

                def mock_stem_behavior(token):
                    return token  # No stemming for this test

                mock_get_stemmer.return_value.stem.side_effect = mock_stem_behavior
                
                # Test case 3: When special characters are included in the text
                text = "Check out this link: http://example.com! The quick brown fox..."
                stem = False  # Do not apply stemming
                result = obj._preprocess_text(text, stem)

                # Expected output after removing special characters and stopwords
                expected_result = "quick brown fox"
                self.assertEqual(result, expected_result)

                # Ensure the proper methods were called
                mock_get_stopwords.assert_called_with("english")  # Ensure stopwords were fetched
                mock_get_stemmer.assert_called_with("english")  # Ensure stemmer was initialized

                # Ensure re.sub was called to clean the text
                mock_re_sub.assert_called_with(TEXT_CLEANING_RE, ' ', str(text).lower())

    @patch('nltk.data.find')  # Mock nltk.data.find
    @patch('re.sub')  # Mock re.sub for text cleaning (TEXT_CLEANING_RE)
    def test_preprocess_text_with_empty_string(self, mock_re_sub, mock_nltk_find):
        # Test case when the input text is an empty string.

        mock_nltk_find.side_effect = None  # Simulate the resource is found

        # Mock re.sub to simulate the cleaned text
        mock_re_sub.return_value = 'the quick brown fox'

        # Instantiate the TextPreprocessor object
        obj = TextPreprocessor()

        with patch.object(obj, '_get_stopwords', return_value=['the', 'is', 'in', 'a']) as mock_get_stopwords:
            with patch.object(obj, '_get_stemmer', new_callable=MagicMock) as mock_get_stemmer:

                def mock_stem_behavior(token):
                    return token  # No stemming for this test

                mock_get_stemmer.return_value.stem.side_effect = mock_stem_behavior

                # Setup mock for re.sub to simulate an empty string after cleaning
                mock_re_sub.return_value = ''  # Empty string after cleaning

                # Test case 4: When the input text is an empty string
                text = ""
                stem = True  # Apply stemming
                result = obj._preprocess_text(text, stem)

                # Expected output should be an empty string
                self.assertEqual(result, "")

                # Ensure the proper methods were called
                mock_get_stopwords.assert_called_with("english")  # Ensure stopwords were fetched
                mock_get_stemmer.assert_called_with("english")  # Ensure stemmer was initialized

                # Ensure re.sub was called with the correct arguments
                mock_re_sub.assert_called_once_with(TEXT_CLEANING_RE, ' ', str(text).lower())
