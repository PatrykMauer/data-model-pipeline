# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files into the container
COPY data_pipeline/requirements.txt /app/data_pipeline_requirements.txt
COPY flask_app/requirements.txt /app/flask_app_requirements.txt
COPY model_training/requirements.txt /app/model_training_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r data_pipeline_requirements.txt
RUN pip install --no-cache-dir -r flask_app_requirements.txt
RUN pip install --no-cache-dir -r model_training_requirements.txt

# Copy the application code into the container
COPY data_pipeline/ /app/data_pipeline/
COPY flask_app/ /app/flask_app/
COPY model_training/ /app/model_training/

# Change permissions to make the script executable
RUN chmod +x data_pipeline/data_processing.sh

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define environment variables
ENV PYTHONUNBUFFERED=1
ENV SMTP_SENDER="email"
ENV SMTP_RECEIVER="email"
ENV SMTP_SERVER="smtp.example.com"
ENV SMTP_PORT=587
ENV SMTP_USER="user"
ENV SMTP_PASSWORD="pass"

# Run the Flask app
CMD ["bash", "-c", "python3 flask_app/app.py"]
