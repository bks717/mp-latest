FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (rasterio, geopandas, GDAL, GEOS, PROJ, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_VERSION=3.10.3
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

COPY requirements.txt .

# Install PyTorch CPU first, then other dependencies
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# Set AWS_NO_SIGN_REQUEST for public Sentinel-1 GRD data access
ENV AWS_NO_SIGN_REQUEST=YES

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
