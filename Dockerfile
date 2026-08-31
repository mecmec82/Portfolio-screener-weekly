# Use the official lightweight Python image
FROM python:3.11-slim

# Set environment variables for stdout buffering
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory
WORKDIR /app

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY . .

# Expose port and start Functions Framework
EXPOSE 8080
CMD ["functions-framework", "--target=run_rebalance_function"]
