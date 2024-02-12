# Stage 1: Build environment
FROM python:3.10.12-slim

WORKDIR /app

COPY image-captioning/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip cache purge

COPY . .

EXPOSE 8080

# Run the Python script
CMD ["python", "image-captioning/server.py"]