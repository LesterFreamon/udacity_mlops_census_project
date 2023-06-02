# Use the base image from the root directory
FROM choya/my-base-image:latest

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Run mlflow_server.py when the container launches
CMD ["python", "mlflow_server.py"]