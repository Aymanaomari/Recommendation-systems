# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them in a virtual environment
COPY requirement.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirement.txt

# Stage 2: Create runtime image
FROM python:3.12-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Ensure venv Python is used
ENV PATH="/opt/venv/bin:$PATH"

# Copy app source code
COPY app/ ./app/
COPY run.py ./

# Expose port (default Flask port)
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=run.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV MONGO_URI="mongodb://localhost:27017/"

# Run the app
CMD ["flask", "run"]
