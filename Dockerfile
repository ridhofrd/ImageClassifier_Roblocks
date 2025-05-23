# Gunakan base image yang cocok untuk ML
FROM python:3.11-slim

# Install dependency sistem (biar OpenCV dan TensorFlow bisa jalan)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja
WORKDIR /app

# Salin requirement dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file dari project
COPY . .

# Jalankan aplikasi dengan Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]