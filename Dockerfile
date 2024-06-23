# Use the official Python image from the Docker Hub
FROM python:3.11.0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable
ENV FLASK_APP=api.py

# Expose any ports the app is expected to run on
EXPOSE 5000  

# Command to run your script
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5000"]
