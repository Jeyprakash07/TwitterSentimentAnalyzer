# Constants for file paths
MODEL_PATH = 'models/model.h5'
LABEL_ENCODER_PATH = 'models/encoder.pkl'
W2V_MODEL_PATH = 'models/model.w2v'
TOKENIZER_PATH = 'models/tokenizer.pkl'

# TRAINING_DATA_PATH = 'data/sentimentdataset.csv'  # Preprocessed and vectorized training data
TRAINING_DATA_PATH = 'data/sentiment140.csv'

# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 64

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"

LABEL_MAP = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

# (TODO: MOVE TO ENV FILE) Specify the custom directory where you want to download NLTK data
CUSTOM_NLTK_DATA_PATH = '/data/sa-lstm-engine/data/nltk_data'

