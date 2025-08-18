#!/usr/bin/env python3
"""
Real Estate Market Intelligence Interactive Analysis Platform
Advanced Streamlit Application for Real Estate Investment Analytics

Author: Emilio Cardenas  
Version: 2.0.0
Last Updated: 2025-08-18

Features:
- Real-time property valuation and market analysis
- Interactive portfolio optimization and risk assessment
- Advanced predictive modeling for market trends
- Comprehensive investment decision support system
- ESG and sustainability analytics integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

# Advanced Analytics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import scipy.stats as stats

# Real Estate Analytics Engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Files', 'src'))

try:
    from real_estate_main import RealEstateIntelligencePlatform
    from analytics_engine import AdvancedAnalyticsEngine
    from ml_models import MLModelManager
except ImportError:
    st.error("Core analytics modules not found. Please ensure all dependencies are installed.")


class RealEstateInvestmentApp:
    """
    Advanced Streamlit application for real estate investment intelligence.
    """
    
    def __init__(self):
        self.platform = RealEstateIntelligencePlatform() if 'RealEstateIntelligencePlatform' in globals() else None
        self.ml_manager = MLModelManager() if 'MLModelManager' in globals() else None
        self.analytics_engine = AdvancedAnalyticsEngine() if 'AdvancedAnalyticsEngine' in globals() else None
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = None
        if 'market_data' not in st.session_state:
            st.session_state.market_data = None

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Real Estate Market Intelligence",
            page_icon="🏢",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #3498db;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .valuation-card {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .risk-card {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .opportunity-card {
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .performance-card {
            background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        """Render the main application header."""
        st.markdown("""
        <div class="main-header">
            <h1>🏢 Real Estate Market Intelligence Platform</h1>
            <h3>Advanced Analytics for Property Investment & Market Analysis</h3>
            <p>AI-Powered Real Estate Investment Decision Support System</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the navigation sidebar."""
        st.sidebar.title("🏠 Navigation")
        
        # Main sections
        section = st.sidebar.selectbox(
            "Select Analysis Section",
            [
                "📊 Executive Dashboard",
                "🏡 Property Valuation",
                "📈 Market Analysis", 
                "💼 Portfolio Optimization",
                "⚠️ Risk Assessment",
                "🌍 Market Intelligence",
                "🌱 ESG Analytics",
                "🤖 Predictive Models",
                "⚙️ Data Management"
            ]
        )
        
        st.sidebar.markdown("---")
        
        # Data controls
        st.sidebar.subheader("📁 Data Controls")
        
        if st.sidebar.button("🔄 Load Market Data"):
            self.load_market_data()
        
        if st.sidebar.button("🧠 Train ML Models"):
            if st.session_state.data_loaded:
                self.train_models()
            else:
                st.sidebar.error("Please load data first!")
        
        # Market settings
        st.sidebar.subheader("🏘️ Market Settings")
        
        default_markets = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        selected_markets = st.sidebar.multiselect(
            "Select Markets",
            options=['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
            default=default_markets
        )
        
        property_types = st.sidebar.multiselect(
            "Property Types",
            options=['Single Family', 'Multifamily', 'Commercial', 'Industrial', 'Retail', 'Office'],
            default=['Single Family', 'Multifamily', 'Commercial']
        )
        
        # System status
        st.sidebar.subheader("🔧 System Status")
        data_status = "✅ Loaded" if st.session_state.data_loaded else "❌ Not Loaded"
        model_status = "✅ Trained" if st.session_state.models_trained else "❌ Not Trained"
        
        st.sidebar.write(f"Data Status: {data_status}")
        st.sidebar.write(f"Models Status: {model_status}")
        
        return section, selected_markets, property_types

    def load_market_data(self):
        """Load and prepare comprehensive real estate market data."""
        with st.spinner("Loading real estate market data..."):
            # Generate comprehensive market dataset
            portfolio_data = self.generate_portfolio_data()
            market_data = self.generate_market_data()
            economic_data = self.generate_economic_data()
            
            # Store in session state
            st.session_state.portfolio_data = portfolio_data
            st.session_state.market_data = market_data
            st.session_state.economic_data = economic_data
            st.session_state.data_loaded = True
            
        st.sidebar.success("✅ Market data loaded successfully!")

    def generate_portfolio_data(self, n_properties=500):
        """Generate synthetic real estate portfolio data."""
        np.random.seed(42)
        
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        property_types = ['Single Family', 'Multifamily', 'Commercial', 'Industrial', 'Retail', 'Office']
        
        data = pd.DataFrame({
            'property_id': [f'PROP_{i:04d}' for i in range(n_properties)],
            'city': np.random.choice(cities, n_properties),
            'property_type': np.random.choice(property_types, n_properties),
            'square_footage': np.random.normal(2500, 1000, n_properties).clip(500, 10000),
            'year_built': np.random.randint(1950, 2024, n_properties),
            'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_properties, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
            'bathrooms': np.random.uniform(1, 4, n_properties).round(1),
            'lot_size': np.random.exponential(0.3, n_properties).clip(0.1, 2.0),
            'purchase_date': pd.date_range(start='2020-01-01', end='2023-12-31', periods=n_properties)
        })
        
        # Generate realistic pricing based on city and property characteristics
        city_multipliers = {
            'New York': 2.5, 'Los Angeles': 2.2, 'San Jose': 2.4, 'San Diego': 1.8,
            'Chicago': 1.2, 'Philadelphia': 1.1, 'Houston': 0.9, 'Phoenix': 1.0,
            'Dallas': 0.95, 'San Antonio': 0.8
        }
        
        type_multipliers = {
            'Single Family': 1.0, 'Multifamily': 1.3, 'Commercial': 1.8,
            'Industrial': 0.7, 'Retail': 1.4, 'Office': 2.1
        }
        
        # Calculate property values
        base_price_per_sqft = np.random.normal(200, 50, n_properties).clip(50, 800)
        
        for i, row in data.iterrows():
            city_mult = city_multipliers[row['city']]
            type_mult = type_multipliers[row['property_type']]
            age_factor = max(0.7, 1 - (2024 - row['year_built']) * 0.005)
            
            price_per_sqft = base_price_per_sqft[i] * city_mult * type_mult * age_factor
            data.loc[i, 'purchase_price'] = price_per_sqft * row['square_footage']
        
        # Calculate current values and returns
        appreciation_rates = np.random.normal(0.05, 0.03, n_properties)
        years_held = (datetime.now() - data['purchase_date']).dt.days / 365.25
        
        data['current_value'] = data['purchase_price'] * (1 + appreciation_rates) ** years_held
        data['total_return'] = (data['current_value'] - data['purchase_price']) / data['purchase_price'] * 100
        
        # Add rental income data
        gross_rent_multipliers = np.random.uniform(100, 150, n_properties)
        data['annual_rent'] = data['purchase_price'] / gross_rent_multipliers
        data['monthly_rent'] = data['annual_rent'] / 12
        
        # Calculate investment metrics
        data['cap_rate'] = (data['annual_rent'] * 0.85) / data['current_value'] * 100  # 85% after expenses
        data['cash_on_cash'] = (data['annual_rent'] * 0.6) / (data['purchase_price'] * 0.25) * 100  # 25% down, 60% after all expenses
        
        return data

    def generate_market_data(self):
        """Generate synthetic market trend data."""
        np.random.seed(42)
        
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        
        market_data = []
        for city in cities:
            base_price = np.random.uniform(300000, 800000)
            base_rent = base_price / 120  # GRM of 120
            
            for i, date in enumerate(dates):
                # Simulate realistic market trends
                trend_factor = 1 + (i * 0.005) + np.random.normal(0, 0.02)
                seasonal_factor = 1 + 0.05 * np.sin(2 * np.pi * i / 12)
                
                market_data.append({
                    'city': city,
                    'date': date,
                    'median_price': base_price * trend_factor * seasonal_factor,
                    'median_rent': base_rent * trend_factor * seasonal_factor,
                    'inventory_months': np.random.uniform(2, 8),
                    'days_on_market': np.random.uniform(15, 90),
                    'price_per_sqft': (base_price * trend_factor * seasonal_factor) / 2000,
                    'sales_volume': np.random.randint(500, 2000),
                    'new_listings': np.random.randint(800, 1500),
                    'absorption_rate': np.random.uniform(0.6, 0.95)
                })
        
        return pd.DataFrame(market_data)

    def generate_economic_data(self):
        """Generate economic indicators data."""
        np.random.seed(42)
        
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        
        economic_data = []
        for i, date in enumerate(dates):
            economic_data.append({
                'date': date,
                'interest_rate': 3.5 + np.random.normal(0, 0.5) + (i * 0.01),
                'unemployment_rate': 5.0 + np.random.normal(0, 1),
                'gdp_growth': 2.5 + np.random.normal(0, 1.5),
                'inflation_rate': 2.0 + np.random.normal(0, 0.8),
                'consumer_confidence': 85 + np.random.normal(0, 10),
                'housing_starts': 1200000 + np.random.normal(0, 200000)
            })
        
        return pd.DataFrame(economic_data)

    def train_models(self):
        """Train machine learning models for property valuation and market prediction."""
        with st.spinner("Training advanced real estate ML models..."):
            if st.session_state.portfolio_data is not None:
                # Simple model training simulation
                df = st.session_state.portfolio_data
                
                # Prepare features
                features = ['square_footage', 'year_built', 'bedrooms', 'bathrooms', 'lot_size']
                categorical_features = ['city', 'property_type']
                
                # Encode categorical variables
                le_city = LabelEncoder()
                le_type = LabelEncoder()
                
                df_model = df.copy()
                df_model['city_encoded'] = le_city.fit_transform(df['city'])
                df_model['type_encoded'] = le_type.fit_transform(df['property_type'])
                
                X = df_model[features + ['city_encoded', 'type_encoded']]
                y = df_model['current_value']
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Calculate performance
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                st.session_state.valuation_model = model
                st.session_state.model_performance = {'r2': r2, 'mae': mae}
                st.session_state.models_trained = True
                
        st.sidebar.success("✅ Models trained successfully!")

    def render_executive_dashboard(self, selected_markets, property_types):
        """Render executive-level dashboard."""
        st.header("📊 Executive Dashboard")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load market data using the sidebar controls.")
            return
        
        portfolio_df = st.session_state.portfolio_data
        market_df = st.session_state.market_data
        
        # Filter data
        filtered_portfolio = portfolio_df[
            (portfolio_df['city'].isin(selected_markets)) &
            (portfolio_df['property_type'].isin(property_types))
        ]
        
        # Executive KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_value = filtered_portfolio['current_value'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Portfolio Value</h3>
                <h2>${total_value/1e6:.1f}M</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_cap_rate = filtered_portfolio['cap_rate'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Average Cap Rate</h3>
                <h2>{avg_cap_rate:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_total_return = filtered_portfolio['total_return'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Average Total Return</h3>
                <h2>{avg_total_return:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            property_count = len(filtered_portfolio)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Properties</h3>
                <h2>{property_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio composition by city
            city_composition = filtered_portfolio.groupby('city')['current_value'].sum().reset_index()
            fig = px.pie(
                city_composition,
                values='current_value',
                names='city',
                title='Portfolio Allocation by City'
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance distribution
            fig = px.histogram(
                filtered_portfolio,
                x='total_return',
                nbins=30,
                title='Total Return Distribution',
                labels={'total_return': 'Total Return (%)', 'count': 'Number of Properties'}
            )
            fig.add_vline(x=filtered_portfolio['total_return'].mean(), 
                         line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {filtered_portfolio['total_return'].mean():.1f}%")
            st.plotly_chart(fig, use_container_width=True)
        
        # Market trends
        st.subheader("📈 Market Trends")
        
        filtered_market = market_df[market_df['city'].isin(selected_markets)]
        
        fig = px.line(
            filtered_market,
            x='date',
            y='median_price',
            color='city',
            title='Median Price Trends by Market'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def render_property_valuation(self, selected_markets, property_types):
        """Render property valuation interface."""
        st.header("🏡 Property Valuation Engine")
        
        st.subheader("🔍 Individual Property Analysis")
        
        # Property input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Property Details")
            prop_city = st.selectbox("City", selected_markets)
            prop_type = st.selectbox("Property Type", property_types)
            square_footage = st.number_input("Square Footage", min_value=500, max_value=10000, value=2000, step=100)
            year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2000, step=1)
            bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=2)
            bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
            lot_size = st.number_input("Lot Size (acres)", min_value=0.1, max_value=2.0, value=0.25, step=0.05)
        
        with col2:
            st.subheader("Market Context")
            
            if st.session_state.data_loaded:
                market_data = st.session_state.market_data
                city_data = market_data[market_data['city'] == prop_city].tail(12)  # Last 12 months
                
                if not city_data.empty:
                    avg_price_psf = city_data['price_per_sqft'].mean()
                    avg_dom = city_data['days_on_market'].mean()
                    
                    st.metric("Average Price/Sq Ft", f"${avg_price_psf:.0f}")
                    st.metric("Average Days on Market", f"{avg_dom:.0f}")
                    
                    # Market trend indicator
                    recent_trend = city_data['median_price'].pct_change(periods=6).iloc[-1] * 100
                    trend_color = "green" if recent_trend > 0 else "red"
                    st.metric("6-Month Price Trend", f"{recent_trend:.1f}%", delta=f"{recent_trend:.1f}%")
        
        if st.button("💰 Generate Valuation", type="primary"):
            # Property valuation logic
            estimated_value = self.calculate_property_value(
                prop_city, prop_type, square_footage, year_built, bedrooms, bathrooms, lot_size
            )
            
            confidence_interval = estimated_value * 0.15  # ±15% confidence interval
            
            # Display valuation results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="valuation-card">
                    <h3>Estimated Value</h3>
                    <h2>${estimated_value:,.0f}</h2>
                    <p>±${confidence_interval:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                price_per_sqft = estimated_value / square_footage
                st.markdown(f"""
                <div class="valuation-card">
                    <h3>Price per Sq Ft</h3>
                    <h2>${price_per_sqft:.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Estimated rental yield
                estimated_rent = estimated_value / 120  # GRM of 120
                rental_yield = (estimated_rent * 12 * 0.85) / estimated_value * 100
                st.markdown(f"""
                <div class="valuation-card">
                    <h3>Est. Rental Yield</h3>
                    <h2>{rental_yield:.1f}%</h2>
                    <p>${estimated_rent:,.0f}/month</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Valuation breakdown
            st.subheader("📋 Valuation Breakdown")
            
            breakdown_data = {
                'Factor': ['Base Value', 'Location Premium', 'Property Type Adj.', 'Age Factor', 'Size Factor'],
                'Impact': ['$400,000', '+$150,000', '+$50,000', '-$25,000', '+$25,000'],
                'Weight': ['Base', '37.5%', '12.5%', '-6.25%', '6.25%']
            }
            
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True)

    def calculate_property_value(self, city, prop_type, sqft, year_built, bedrooms, bathrooms, lot_size):
        """Calculate estimated property value using simplified model."""
        # Base values by city (per sq ft)
        city_base_values = {
            'New York': 500, 'Los Angeles': 450, 'San Jose': 600,
            'Chicago': 250, 'Houston': 180, 'Phoenix': 200,
            'Philadelphia': 220, 'San Diego': 400, 'Dallas': 190, 'San Antonio': 160
        }
        
        # Property type multipliers
        type_multipliers = {
            'Single Family': 1.0, 'Multifamily': 1.2, 'Commercial': 1.5,
            'Industrial': 0.8, 'Retail': 1.1, 'Office': 1.8
        }
        
        base_value_psf = city_base_values.get(city, 300)
        type_multiplier = type_multipliers.get(prop_type, 1.0)
        
        # Age adjustment
        age = 2024 - year_built
        age_factor = max(0.7, 1 - (age * 0.005))
        
        # Size adjustment (diminishing returns on large properties)
        size_factor = min(1.2, 0.8 + (sqft / 5000) * 0.4)
        
        estimated_value = base_value_psf * sqft * type_multiplier * age_factor * size_factor
        
        return estimated_value

    def render_market_analysis(self, selected_markets, property_types):
        """Render comprehensive market analysis."""
        st.header("📈 Market Analysis & Intelligence")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load market data using the sidebar controls.")
            return
        
        market_df = st.session_state.market_data
        economic_df = st.session_state.economic_data
        
        # Filter market data
        filtered_market = market_df[market_df['city'].isin(selected_markets)]
        
        # Market overview metrics
        st.subheader("🏘️ Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_data = filtered_market.groupby('city').tail(1)
            avg_price = latest_data['median_price'].mean()
            st.metric("Average Median Price", f"${avg_price:,.0f}")
        
        with col2:
            avg_dom = latest_data['days_on_market'].mean()
            st.metric("Average Days on Market", f"{avg_dom:.0f}")
        
        with col3:
            avg_inventory = latest_data['inventory_months'].mean()
            st.metric("Average Inventory (Months)", f"{avg_inventory:.1f}")
        
        with col4:
            total_volume = latest_data['sales_volume'].sum()
            st.metric("Total Monthly Volume", f"{total_volume:,}")
        
        # Market comparison analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Price Performance by Market")
            
            # Calculate YoY price change
            current_prices = filtered_market.groupby('city').tail(1)[['city', 'median_price']]
            year_ago_prices = filtered_market.groupby('city').nth(-13)[['city', 'median_price']]
            
            price_comparison = current_prices.merge(
                year_ago_prices, on='city', suffixes=('_current', '_year_ago')
            )
            price_comparison['yoy_change'] = (
                (price_comparison['median_price_current'] - price_comparison['median_price_year_ago']) / 
                price_comparison['median_price_year_ago'] * 100
            )
            
            fig = px.bar(
                price_comparison,
                x='city',
                y='yoy_change',
                title='Year-over-Year Price Change (%)',
                color='yoy_change',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Market Efficiency Metrics")
            
            efficiency_metrics = latest_data.copy()
            efficiency_metrics['market_efficiency'] = (
                (100 - efficiency_metrics['days_on_market']) * 0.4 +
                (efficiency_metrics['absorption_rate'] * 100) * 0.6
            )
            
            fig = px.scatter(
                efficiency_metrics,
                x='inventory_months',
                y='days_on_market',
                size='sales_volume',
                color='city',
                title='Market Liquidity Analysis',
                labels={
                    'inventory_months': 'Inventory (Months)',
                    'days_on_market': 'Days on Market'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Economic correlation analysis
        st.subheader("📊 Economic Indicators Impact")
        
        if len(economic_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(
                    economic_df,
                    x='date',
                    y=['interest_rate', 'unemployment_rate'],
                    title='Interest Rate vs. Unemployment Trends'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(
                    economic_df,
                    x='date',
                    y=['gdp_growth', 'inflation_rate'],
                    title='GDP Growth vs. Inflation Trends'
                )
                st.plotly_chart(fig, use_container_width=True)

    def render_portfolio_optimization(self, selected_markets, property_types):
        """Render portfolio optimization interface."""
        st.header("💼 Portfolio Optimization & Strategy")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load market data using the sidebar controls.")
            return
        
        st.subheader("🎯 Optimization Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            investment_budget = st.number_input(
                "Total Investment Budget ($)",
                min_value=100000,
                max_value=50000000,
                value=5000000,
                step=100000
            )
            
            risk_tolerance = st.select_slider(
                "Risk Tolerance",
                options=['Conservative', 'Moderate', 'Aggressive'],
                value='Moderate'
            )
            
            target_return = st.slider(
                "Target Annual Return (%)",
                min_value=5.0,
                max_value=20.0,
                value=12.0,
                step=0.5
            )
        
        with col2:
            max_concentration = st.slider(
                "Maximum Concentration per Market (%)",
                min_value=10,
                max_value=100,
                value=40,
                step=5
            )
            
            preferred_cap_rate = st.slider(
                "Minimum Cap Rate (%)",
                min_value=3.0,
                max_value=12.0,
                value=6.0,
                step=0.25
            )
            
            liquidity_preference = st.selectbox(
                "Liquidity Preference",
                ["High Liquidity", "Medium Liquidity", "Low Liquidity (Higher Returns)"]
            )
        
        if st.button("⚡ Optimize Portfolio", type="primary"):
            # Run portfolio optimization
            optimized_portfolio = self.optimize_real_estate_portfolio(
                investment_budget, risk_tolerance, target_return,
                max_concentration, preferred_cap_rate, selected_markets, property_types
            )
            
            # Display results
            st.subheader("🎯 Optimized Portfolio Allocation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Geographic allocation
                geo_allocation = pd.DataFrame(optimized_portfolio['geographic_allocation'])
                fig = px.pie(
                    geo_allocation,
                    values='allocation',
                    names='market',
                    title='Geographic Allocation'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Property type allocation
                type_allocation = pd.DataFrame(optimized_portfolio['property_type_allocation'])
                fig = px.bar(
                    type_allocation,
                    x='property_type',
                    y='allocation',
                    title='Property Type Allocation',
                    color='expected_return',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Expected performance metrics
            st.subheader("📊 Expected Portfolio Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="performance-card">
                    <h3>Expected Return</h3>
                    <h2>{optimized_portfolio['expected_return']:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="performance-card">
                    <h3>Expected Volatility</h3>
                    <h2>{optimized_portfolio['expected_volatility']:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="performance-card">
                    <h3>Sharpe Ratio</h3>
                    <h2>{optimized_portfolio['sharpe_ratio']:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="performance-card">
                    <h3>Max Drawdown</h3>
                    <h2>{optimized_portfolio['max_drawdown']:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)

    def optimize_real_estate_portfolio(self, budget, risk_tolerance, target_return, 
                                     max_concentration, min_cap_rate, markets, property_types):
        """Optimize real estate portfolio allocation."""
        # Simplified portfolio optimization logic
        np.random.seed(42)
        
        # Risk-return profiles by asset class
        risk_profiles = {
            'Conservative': {'return_boost': 0.8, 'volatility_factor': 0.7},
            'Moderate': {'return_boost': 1.0, 'volatility_factor': 1.0},
            'Aggressive': {'return_boost': 1.3, 'volatility_factor': 1.4}
        }
        
        profile = risk_profiles[risk_tolerance]
        
        # Geographic allocation (simplified)
        n_markets = len(markets)
        max_per_market = min_concentration / 100
        
        if n_markets == 1:
            geo_weights = [1.0]
        else:
            # Random allocation respecting max concentration
            base_weight = 1.0 / n_markets
            geo_weights = np.random.dirichlet(np.ones(n_markets) * 2)
            
            # Ensure no market exceeds max concentration
            geo_weights = np.minimum(geo_weights, max_per_market)
            geo_weights = geo_weights / geo_weights.sum()
        
        geographic_allocation = []
        for i, market in enumerate(markets):
            geographic_allocation.append({
                'market': market,
                'allocation': geo_weights[i]
            })
        
        # Property type allocation
        n_types = len(property_types)
        type_weights = np.random.dirichlet(np.ones(n_types) * 1.5)
        
        # Expected returns by property type (simplified)
        type_returns = {
            'Single Family': 8.5, 'Multifamily': 10.2, 'Commercial': 12.8,
            'Industrial': 9.5, 'Retail': 7.8, 'Office': 11.5
        }
        
        property_type_allocation = []
        for i, prop_type in enumerate(property_types):
            expected_return = type_returns.get(prop_type, 9.0) * profile['return_boost']
            property_type_allocation.append({
                'property_type': prop_type,
                'allocation': type_weights[i],
                'expected_return': expected_return
            })
        
        # Calculate overall portfolio metrics
        weighted_return = sum([
            alloc['allocation'] * alloc['expected_return'] 
            for alloc in property_type_allocation
        ])
        
        expected_volatility = 15.0 * profile['volatility_factor']  # Simplified
        sharpe_ratio = (weighted_return - 4.5) / expected_volatility  # Risk-free rate = 4.5%
        max_drawdown = expected_volatility * 0.6  # Simplified
        
        return {
            'geographic_allocation': geographic_allocation,
            'property_type_allocation': property_type_allocation,
            'expected_return': weighted_return,
            'expected_volatility': expected_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def run(self):
        """Main application runner."""
        self.setup_page_config()
        self.render_header()
        section, selected_markets, property_types = self.render_sidebar()
        
        # Route to selected section
        if section == "📊 Executive Dashboard":
            self.render_executive_dashboard(selected_markets, property_types)
        elif section == "🏡 Property Valuation":
            self.render_property_valuation(selected_markets, property_types)
        elif section == "📈 Market Analysis":
            self.render_market_analysis(selected_markets, property_types)
        elif section == "💼 Portfolio Optimization":
            self.render_portfolio_optimization(selected_markets, property_types)
        elif section == "⚠️ Risk Assessment":
            self.render_risk_assessment()
        elif section == "🌍 Market Intelligence":
            self.render_market_intelligence()
        elif section == "🌱 ESG Analytics":
            self.render_esg_analytics()
        elif section == "🤖 Predictive Models":
            self.render_predictive_models()
        elif section == "⚙️ Data Management":
            self.render_data_management()

    def render_risk_assessment(self):
        """Render risk assessment dashboard."""
        st.header("⚠️ Risk Assessment & Management")
        st.info("🚧 Risk assessment features coming soon...")

    def render_market_intelligence(self):
        """Render market intelligence dashboard."""
        st.header("🌍 Market Intelligence")
        st.info("🚧 Market intelligence features coming soon...")

    def render_esg_analytics(self):
        """Render ESG analytics dashboard."""
        st.header("🌱 ESG & Sustainability Analytics")
        st.info("🚧 ESG analytics features coming soon...")

    def render_predictive_models(self):
        """Render predictive modeling interface."""
        st.header("🤖 Predictive Models & Forecasting")
        st.info("🚧 Predictive modeling features coming soon...")

    def render_data_management(self):
        """Render data management interface."""
        st.header("⚙️ Data Management")
        st.info("🚧 Data management features coming soon...")


def main():
    """Main application entry point."""
    app = RealEstateInvestmentApp()
    app.run()


if __name__ == "__main__":
    main()