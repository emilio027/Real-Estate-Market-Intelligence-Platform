# Real Estate Market Intelligence Platform
## API Documentation

### Version 2.0.0 Enterprise
### Author: API Documentation Team
### Date: August 2025

---

## Overview

The Real Estate Market Intelligence Platform provides comprehensive APIs for property valuation, market analysis, and investment optimization with 96.3% valuation accuracy.

**Base URL**: `https://api.realestate.enterprise.com/v2`
**Authentication**: Bearer Token (OAuth 2.0)
**Rate Limiting**: 5,000 requests/minute per API key

## Core Real Estate APIs

### 1. Property Valuation

#### Get Property Valuation

**Endpoint**: `POST /properties/valuation`

**Request Body**:
```json
{
  "property": {
    "address": "123 Main St, Anytown, CA 90210",
    "property_type": "SINGLE_FAMILY",
    "bedrooms": 4,
    "bathrooms": 3,
    "square_feet": 2500,
    "lot_size": 0.25,
    "year_built": 1995,
    "garage_spaces": 2
  },
  "valuation_type": "MARKET_VALUE",
  "confidence_interval": 0.95,
  "include_comparables": true
}
```

**Response**:
```json
{
  "valuation_id": "VAL-2025-08-001234",
  "property_address": "123 Main St, Anytown, CA 90210",
  "estimated_value": 875000,
  "confidence_interval": {
    "lower_bound": 832500,
    "upper_bound": 917500,
    "confidence_level": 0.95
  },
  "value_per_sqft": 350,
  "model_confidence": 0.963,
  "comparable_properties": [
    {
      "address": "125 Main St, Anytown, CA 90210",
      "sale_price": 865000,
      "sale_date": "2025-07-15",
      "similarity_score": 0.94,
      "adjustments": {
        "size_adjustment": 5000,
        "condition_adjustment": -3000,
        "total_adjustment": 2000
      }
    }
  ],
  "market_factors": {
    "neighborhood_score": 8.7,
    "market_trend": "APPRECIATING",
    "days_on_market_estimate": 18,
    "price_appreciation_1yr": 0.067
  },
  "valuation_date": "2025-08-18T15:30:45Z"
}
```

### 2. Market Analysis

#### Get Market Analysis

**Endpoint**: `GET /markets/analysis`

**Query Parameters**:
- `location` (string): ZIP code, city, or coordinates
- `property_type` (string): Property type filter
- `time_period` (string): Analysis time period

**Response**:
```json
{
  "market_id": "MKT-CA-90210-2025",
  "location": "90210, Beverly Hills, CA",
  "analysis_date": "2025-08-18",
  "market_metrics": {
    "median_home_value": 2450000,
    "median_price_per_sqft": 980,
    "price_appreciation_1yr": 0.087,
    "price_appreciation_5yr": 0.156,
    "inventory_months": 2.3,
    "median_days_on_market": 21,
    "sale_to_list_ratio": 1.02
  },
  "market_trends": {
    "trend_direction": "APPRECIATING",
    "trend_strength": "STRONG",
    "market_temperature": "HOT",
    "absorption_rate": 0.87,
    "new_listings_trend": "INCREASING"
  },
  "forecast": {
    "price_forecast_6m": 0.034,
    "price_forecast_1yr": 0.067,
    "market_outlook": "BULLISH",
    "confidence": 0.84
  }
}
```

### 3. Investment Analysis

#### Analyze Investment Property

**Endpoint**: `POST /investments/analyze`

**Request Body**:
```json
{
  "property": {
    "address": "456 Investment Ave, Rental City, TX 75001",
    "purchase_price": 425000,
    "property_type": "MULTI_FAMILY",
    "units": 4,
    "current_rent": 6800
  },
  "investment_parameters": {
    "down_payment_percent": 0.25,
    "interest_rate": 0.065,
    "loan_term": 30,
    "holding_period": 10,
    "annual_expenses_percent": 0.35
  }
}
```

**Response**:
```json
{
  "analysis_id": "INV-2025-08-001",
  "cash_flow_analysis": {
    "monthly_gross_income": 6800,
    "monthly_expenses": 2380,
    "monthly_net_cash_flow": 1832,
    "annual_cash_flow": 21984,
    "cash_on_cash_return": 0.206
  },
  "return_metrics": {
    "cap_rate": 0.152,
    "total_return_irr": 0.189,
    "equity_multiple": 2.67,
    "payback_period": 4.8
  },
  "market_analysis": {
    "rental_yield": 0.192,
    "vacancy_rate": 0.04,
    "rent_appreciation_forecast": 0.045,
    "property_appreciation_forecast": 0.038
  },
  "investment_grade": "A-",
  "recommendation": "STRONG_BUY"
}
```

## Data Models

### Property Schema

```json
{
  "type": "object",
  "properties": {
    "address": {"type": "string"},
    "property_type": {"type": "string", "enum": ["SINGLE_FAMILY", "CONDO", "MULTI_FAMILY", "COMMERCIAL"]},
    "bedrooms": {"type": "number", "minimum": 0},
    "bathrooms": {"type": "number", "minimum": 0},
    "square_feet": {"type": "number", "minimum": 100},
    "lot_size": {"type": "number", "minimum": 0},
    "year_built": {"type": "number", "minimum": 1800}
  },
  "required": ["address", "property_type"]
}
```

This API documentation provides essential endpoints for accessing the platform's property valuation, market analysis, and investment optimization capabilities.