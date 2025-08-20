# 🏢 Real Estate Intelligence Platform - Installation Guide

## Prerequisites

### System Requirements
- **Python 3.9+** - Primary development environment
- **16GB+ RAM** - Required for geospatial data processing
- **SSD Storage** - Recommended for property database operations
- **Stable Internet** - For real-time MLS and market data feeds

### Required API Access
- **MLS Data**: Multiple Listing Service APIs
- **Property Data**: Zillow, Rentals.com, RentSpree APIs
- **Geographic Data**: Google Maps, Census Bureau APIs
- **Market Intelligence**: CoreLogic, RealtyTrac APIs

## Quick Installation (5 Minutes)

### 1. Clone Repository
```bash
git clone <repository-url>
cd Real-Estate-Market-Intelligence-Platform
```

### 2. Docker Setup (Recommended)
```bash
# Build and start real estate analytics environment
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 3. Access Platform
- **Property Dashboard**: http://localhost:8080
- **API Documentation**: http://localhost:8080/api/docs
- **Live Demo**: Open `interactive_demo.html` in browser

## Detailed Installation

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r Technical/Deployment/requirements.txt
```

### 2. Configuration
Required environment variables:
```env
# Real Estate Data APIs
MLS_API_KEY=your_mls_api_key
ZILLOW_API_KEY=your_zillow_api_key
RENTALS_API_KEY=your_rentals_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_key
CENSUS_API_KEY=your_census_api_key

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/realestate_db
MONGODB_URL=mongodb://localhost:27017/realestate_listings
REDIS_URL=redis://localhost:6379

# Investment Parameters
DEFAULT_INVESTMENT_BUDGET=2000000
RISK_TOLERANCE=moderate
TARGET_ROI_THRESHOLD=0.15
GEOGRAPHIC_DIVERSIFICATION_MIN=0.20
```

### 3. Database Setup
```bash
# Start database services
docker-compose up -d postgres mongodb redis

# Initialize database schema
python Technical/Source_Code/data_manager.py --init-db

# Load property and market data
python Technical/Source_Code/data_manager.py --load-property-data

# Import MLS historical data
python Technical/Source_Code/data_manager.py --import-mls --years=3
```

### 4. Validation & Testing
```bash
# Run system validation
python Technical/Source_Code/real_estate_main.py --validate

# Test valuation models
python Technical/Source_Code/real_estate_main.py --test-valuations

# Run investment analysis backtesting
python Technical/Source_Code/real_estate_main.py --backtest --portfolio-size=50M
```

---

**⚠️ Important**: Always validate property valuations with local market experts before making significant investment decisions. This platform provides analytical insights but real estate decisions should consider local market conditions and professional advice.