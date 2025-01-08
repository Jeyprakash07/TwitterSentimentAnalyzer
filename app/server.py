import grpc
from concurrent import futures
from .proto import sa_lstm_engine_pb2, sa_lstm_engine_pb2_grpc
from .inference import predict

class MLService(sa_lstm_engine_pb2_grpc.SentimentAnalysisServiceServicer):
    def PredictSentiment(self, request, context):     
        # Make the prediction
        prediction = predict(request=request)
        
        # Return the prediction as a response
        return sa_lstm_engine_pb2.PredictSentimentResponse(sentiment=prediction)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sa_lstm_engine_pb2_grpc.add_SentimentAnalysisServiceServicer_to_server(MLService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server started, listening on port 50051...")
    server.start()
    server.wait_for_termination()
