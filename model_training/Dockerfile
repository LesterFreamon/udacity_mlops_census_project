# Use the base image from the root directory
FROM choya/my-base-image:latest

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any additional needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run train_model.py when the container launches
CMD ["python", "train_model.py"]