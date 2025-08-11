# Use a slim, modern Python image
FROM python:3.11-slim

# Install essential build tools and system dependencies for scientific/graphical packages
# build-essential: provides compilers like gcc needed for numpy, pandas, etc.
# libgomp1: often required by matplotlib and other scientific libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy and install Python dependencies first to leverage Docker's build cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# IMPORTANT: Ensure your python file is named "main.py"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]