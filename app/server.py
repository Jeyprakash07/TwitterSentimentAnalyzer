import grpc
from concurrent import futures
from .proto import twitter_sentiment_analyzer_pb2, twitter_sentiment_analyzer_pb2_grpc
from .inference import predict
from . import logger

class MLService(twitter_sentiment_analyzer_pb2_grpc.SentimentAnalysisServiceServicer):
    def PredictSentiment(self, request, context):     
        # Make the prediction
        prediction = predict(request=request)
        
        # Return the prediction as a response
        return twitter_sentiment_analyzer_pb2.PredictSentimentResponse(sentiment=prediction)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    twitter_sentiment_analyzer_pb2_grpc.add_SentimentAnalysisServiceServicer_to_server(MLService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Server started, listening on port 50051...")
    server.wait_for_termination()
