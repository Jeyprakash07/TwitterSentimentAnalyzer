import grpc
from app.proto.sa_lstm_engine_pb2_grpc import SentimentAnalysisServiceStub
from app.proto.sa_lstm_engine_pb2 import PredictSentimentRequest
import time
import asyncio
from app import logger

async def run():
    # Connect to the server
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = SentimentAnalysisServiceStub(channel)
        
        current_time = time.strftime("%a %b %d %H:%M:%S %Z %Y", time.localtime())

        # Make a request
        response = await stub.PredictSentiment(PredictSentimentRequest(id='1234567890', date=current_time, flag='', user='T_Power', text="I feel grateful"))
        logger.info(f"Sentiment-Analysis client received: {response}")

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(run())