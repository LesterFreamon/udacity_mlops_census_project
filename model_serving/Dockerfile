# Use the root image as the base
FROM choya/my-base-image:latest

# Set the working directory
WORKDIR /app

# Copy the model serving component files
COPY model_serving/MLproject .
COPY model_serving/app ./app
COPY model_serving/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run your application
CMD ["python", "-u", "./app/main.py"]