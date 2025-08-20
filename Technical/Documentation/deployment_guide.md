# Real Estate Market Intelligence Platform
## Deployment Guide

### Version 2.0.0 Enterprise
### Author: DevOps Engineering Team
### Date: August 2025

---

## Overview

This deployment guide covers the Real Estate Market Intelligence Platform with focus on geospatial data processing, property image analysis, and high-volume valuation services.

## Prerequisites

### System Requirements

**Production Requirements**:
- CPU: 16 cores (Intel Xeon recommended)
- RAM: 128GB DDR4 ECC memory
- Storage: 2TB NVMe SSD + 10TB for property images
- Network: 10Gbps connectivity
- GPU: Optional NVIDIA V100 for image processing

### Software Dependencies

- **Runtime**: Python 3.11+, PostGIS, GDAL
- **Databases**: PostgreSQL 15+ with PostGIS, MongoDB 6.0+
- **Geospatial**: PostGIS, GeoPandas, Shapely, GDAL
- **ML Stack**: Scikit-learn, XGBoost, OpenCV

## Local Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/enterprise/real-estate-intelligence.git
cd real-estate-intelligence

# Create virtual environment
python3.11 -m venv realestate_env
source realestate_env/bin/activate

# Install geospatial dependencies
sudo apt-get install gdal-bin libgdal-dev libspatialindex-dev

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-geospatial.txt
```

### 2. PostGIS Database Setup

```bash
# Install PostgreSQL with PostGIS
sudo apt install postgresql-15 postgresql-15-postgis-3
sudo systemctl start postgresql

# Create real estate database with PostGIS
sudo -u postgres createdb real_estate_intelligence
sudo -u postgres psql real_estate_intelligence
```

```sql
-- Enable PostGIS extensions
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_topology;
CREATE EXTENSION fuzzystrmatch;
CREATE EXTENSION postgis_tiger_geocoder;

-- Create real estate user
CREATE USER realestate WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE real_estate_intelligence TO realestate;
```

### 3. Environment Configuration

```bash
# Create .env file
cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql://realestate:password@localhost/real_estate_intelligence
MONGODB_URL=mongodb://localhost:27017/real_estate
REDIS_URL=redis://localhost:6379

# Geospatial Configuration
POSTGIS_URL=postgresql://realestate:password@localhost/real_estate_intelligence
GDAL_DATA=/usr/share/gdal
PROJ_LIB=/usr/share/proj

# Real Estate Data APIs
MLS_API_KEY=your-mls-api-key
ZILLOW_API_KEY=your-zillow-key
REALTOR_API_KEY=your-realtor-key
GOOGLE_MAPS_API_KEY=your-google-maps-key

# Model Configuration
MODEL_PATH=./models
PROPERTY_IMAGES_PATH=./data/images
VALUATION_CACHE_TTL=3600

# Security
JWT_SECRET_KEY=your-jwt-secret-32-characters
ENCRYPTION_KEY=your-encryption-key-32-chars

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
EOF
```

### 4. Start Development Environment

```bash
# Start geospatial services
docker-compose -f docker-compose.geospatial.yml up -d

# Run property data ingestion
python scripts/ingest_property_data.py

# Start application
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Verify geospatial functionality
curl http://localhost:8000/health
curl http://localhost:8000/properties/valuation -X POST -d '{...}'
```

## Docker Deployment

### 1. Geospatial Dockerfile

```dockerfile
FROM osgeo/gdal:ubuntu-small-3.7.1 AS geospatial-base

# Install Python and geospatial dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    postgresql-client \
    libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-geospatial.txt ./
RUN pip3.11 install --no-cache-dir -r requirements.txt
RUN pip3.11 install --no-cache-dir -r requirements-geospatial.txt

FROM geospatial-base AS production

RUN groupadd -r realestate && useradd -r -g realestate realestate

COPY --chown=realestate:realestate Files/ ./Files/
COPY --chown=realestate:realestate models/ ./models/
COPY --chown=realestate:realestate *.py ./

ENV PYTHONPATH=/app
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

USER realestate

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2. Docker Compose for Real Estate Platform

```yaml
version: '3.8'

services:
  realestate-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://realestate:password@postgis:5432/real_estate_intelligence
      - MONGODB_URL=mongodb://mongo:27017/real_estate
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - property_images:/app/data/images
    depends_on:
      - postgis
      - mongo
      - redis
    restart: unless-stopped

  postgis:
    image: postgis/postgis:15-3.3-alpine
    environment:
      POSTGRES_DB: real_estate_intelligence
      POSTGRES_USER: realestate
      POSTGRES_PASSWORD: password
    volumes:
      - postgis_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  mongo:
    image: mongo:6.0
    environment:
      MONGO_INITDB_DATABASE: real_estate
    volumes:
      - mongo_data:/data/db
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 4gb
    volumes:
      - redis_data:/data
    restart: unless-stopped

  elasticsearch:
    image: elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    restart: unless-stopped

volumes:
  postgis_data:
  mongo_data:
  redis_data:
  elasticsearch_data:
  property_images:
```

## Kubernetes Deployment

### 1. Real Estate Application Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: realestate-app
  namespace: real-estate-intelligence
spec:
  replicas: 3
  selector:
    matchLabels:
      app: realestate-app
  template:
    metadata:
      labels:
        app: realestate-app
    spec:
      containers:
      - name: realestate-app
        image: enterprise/real-estate-intelligence:2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: realestate-secrets
              key: database-url
        - name: GOOGLE_MAPS_API_KEY
          valueFrom:
            secretKeyRef:
              name: realestate-secrets
              key: google-maps-key
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
        volumeMounts:
        - name: property-models
          mountPath: /app/models
        - name: property-images
          mountPath: /app/data/images
      volumes:
      - name: property-models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: property-images
        persistentVolumeClaim:
          claimName: images-pvc
```

### 2. PostGIS Database Deployment

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgis
  namespace: real-estate-intelligence
spec:
  serviceName: postgis
  replicas: 1
  selector:
    matchLabels:
      app: postgis
  template:
    metadata:
      labels:
        app: postgis
    spec:
      containers:
      - name: postgis
        image: postgis/postgis:15-3.3
        env:
        - name: POSTGRES_DB
          value: "real_estate_intelligence"
        - name: POSTGRES_USER
          value: "realestate"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: realestate-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgis-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
  volumeClaimTemplates:
  - metadata:
      name: postgis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 500Gi
```

## Performance Optimization

### 1. PostGIS Optimization

```sql
-- PostGIS performance tuning
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';

-- Spatial indexing
CREATE INDEX CONCURRENTLY idx_properties_geom ON properties USING GIST (geom);
CREATE INDEX CONCURRENTLY idx_properties_location ON properties (city, state, zip_code);

-- Analyze tables for query optimization
ANALYZE properties;
ANALYZE sales_comparables;
```

### 2. Application Tuning

```python
# Optimized valuation configuration
VALUATION_CONFIG = {
    'cache_valuations': True,
    'cache_ttl': 3600,
    'batch_processing': True,
    'max_batch_size': 1000,
    'parallel_processing': True,
    'num_workers': 4
}
```

## Monitoring Real Estate Models

### 1. Property Valuation Metrics

```yaml
# Prometheus real estate metrics
- name: property_valuation_accuracy
  help: Accuracy of property valuations
  type: gauge
  
- name: valuation_request_duration
  help: Time to complete property valuation
  type: histogram
  
- name: market_data_freshness
  help: Age of market data used in valuations
  type: gauge
```

### 2. Real Estate Alerts

```yaml
groups:
- name: realestate-alerts
  rules:
  - alert: ValuationAccuracyDrop
    expr: property_valuation_accuracy < 0.90
    for: 30m
    labels:
      severity: warning
    annotations:
      summary: "Property valuation accuracy below 90%"
      
  - alert: MarketDataStale
    expr: market_data_freshness > 86400
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Market data older than 24 hours"
```

This deployment guide provides comprehensive instructions for deploying the Real Estate Market Intelligence Platform with specialized focus on geospatial data processing, property image handling, and high-volume valuation services.