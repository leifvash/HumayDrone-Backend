FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies for OpenCV and PostgreSQL client
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU first to keep image lightweight (<200MB vs >2GB GPU build)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app/

# Ensure media and static directories exist
RUN mkdir -p /app/media/captures /app/media/masks /app/staticfiles

EXPOSE 8000

# Apply migrations, collect static files, and launch gunicorn (1 worker + 4 threads fits Render $25/mo Standard cleanly without duplicating ML RAM)
CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py collectstatic --noinput && gunicorn config.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120"]
