# Stage 1: Build environment
FROM python:3.10.12-slim

WORKDIR /app

COPY image-captioning/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# # Stage 2: Runtime environment
# FROM python:3.9-slim

# WORKDIR /app

# # Copy the installed dependencies from the builder stage
# COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# # Copy the Python script
# COPY your_script.py .

EXPOSE 8080

# Run the Python script
CMD ["python", "server.py"]
