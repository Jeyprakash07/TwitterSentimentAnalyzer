# Use an official Python runtime as the base image
FROM python:3.10.16

# Set environment variable for NLTK data directory
ENV NLTK_DATA=/app/nltk_data

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . /app

# Install system dependencies for grpc_tools
RUN apt-get update && apt-get install -y \
    build-essential \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Generate the Python gRPC code from the .proto file
RUN python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. ./app/proto/sa_lstm_engine.proto

# Expose the gRPC port
EXPOSE 50051

# Start the gRPC server
CMD ["python", "main.py"]
