# Use Python 3.12 as base image (matching pyproject.toml requirements)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (matching local version)
RUN pip install --no-cache-dir poetry==2.2.1

# Configure Poetry: Don't create virtual environment, install dependencies to system Python
RUN poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose Chainlit default port
EXPOSE 8000

# Health check - Chainlit serves on root, so we check the main endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run Chainlit with --host 0.0.0.0 for Docker compatibility
# Use -w flag for watch mode (auto-reload on code changes) - remove in production if needed
# Add --root-path flag if deploying to a subpath (e.g., --root-path /chainlit)
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]

