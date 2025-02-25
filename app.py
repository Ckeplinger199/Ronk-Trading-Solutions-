import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import logging
from datetime import datetime, timedelta, date
import requests
import time
import os

st.set_page_config(page_title="Options Evaluation Program", layout="wide")

# Configure logging
log_file = 'options_analysis.log'
if os.path.exists(log_file):
    try:
        # Backup old log file if it exists and is too large
        if os.path.getsize(log_file) > 5 * 1024 * 1024:  # 5MB
            backup_file = f'options_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            os.rename(log_file, backup_file)
    except Exception as e:
        print(f"Error managing log file: {str(e)}")

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)
logger.info("Starting Options Analysis Program")

# Constants
TRADIER_BASE_URL = "https://sandbox.tradier.com/v1"

# Safely load secrets with defaults
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except KeyError:
        if default is not None:
            return default
        st.warning(f" {key} not found in secrets. Some features may be limited.")
        return ""

# Load API keys
TRADIER_TOKEN = get_secret("tradier_token")
ALPHA_VANTAGE_API_KEY = get_secret("alpha_vantage_api_key")
NEWS_API_KEY = get_secret("news_api_key")

# Validate API keys
if not TRADIER_TOKEN:
    logger.error("Tradier API key not found in secrets!")
    st.error("Tradier API key not found in secrets!")
    st.stop()

# Initialize global headers for API requests
HEADERS = {
    "Authorization": f"Bearer {TRADIER_TOKEN}",
    "Accept": "application/json"
}

# Bloomberg Terminal Theme Colors
BLOOMBERG_COLORS = {
    "terminal_green": "#00FF00",
    "terminal_amber": "#FFA500",
    "alert_red": "#FF0000",
    "positive_green": "#00CC00",
    "negative_red": "#CC0000",
    "grid_gray": "#333333",
    "axis_white": "#FFFFFF"
}

# Bloomberg Terminal Plotly Template
BLOOMBERG_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#000000",
        "plot_bgcolor": "#1A1A1A",
        "font": {"color": "#00FF00", "family": "Consolas, 'Courier New', monospace"},
        "xaxis": {
            "gridcolor": "#333333",
            "linecolor": "#FFFFFF",
            "tickfont": {"color": "#00FF00"},
            "title_font": {"color": "#FFA500"}
        },
        "yaxis": {
            "gridcolor": "#333333",
            "linecolor": "#FFFFFF",
            "tickfont": {"color": "#00FF00"},
            "title_font": {"color": "#FFA500"}
        },
        "colorway": ["#FFA500", "#00FF00", "#FF0000", "#FFFFFF"]
    }
}

# Set default Plotly template
pio.templates["bloomberg"] = go.layout.Template(BLOOMBERG_TEMPLATE)
pio.templates.default = "bloomberg"

# Apply Bloomberg Terminal styling
def apply_bloomberg_theme():
    st.markdown("""
        <style>
            /* Main theme colors */
            .stApp {
                background-color: #000000;
            }
            
            /* Dataframe styling */
            .stDataFrame, .stTable {
                background-color: #1A1A1A !important;
                border: 1px solid #333333 !important;
            }
            .dataframe th {
                background-color: #000000 !important;
                color: #FFA500 !important;
                font-family: Consolas, 'Courier New', monospace !important;
            }
            .dataframe td {
                color: #00FF00 !important;
                font-family: Consolas, 'Courier New', monospace !important;
            }
            
            /* Metric styling */
            .stMetric label {
                color: #FFA500 !important;
            }
            .stMetric .value {
                color: #00FF00 !important;
            }
            
            /* Button styling */
            .stButton button {
                background-color: #1A1A1A !important;
                color: #FFA500 !important;
                border: 1px solid #333333 !important;
            }
            .stButton button:hover {
                border-color: #FFA500 !important;
            }
            
            /* Input fields */
            .stTextInput input {
                background-color: #1A1A1A !important;
                color: #00FF00 !important;
                border-color: #333333 !important;
            }
            
            /* Selectbox */
            .stSelectbox select {
                background-color: #1A1A1A !important;
                color: #00FF00 !important;
                border-color: #333333 !important;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: #1A1A1A !important;
            }
            .stTabs [data-baseweb="tab"] {
                color: #FFA500 !important;
            }
            .stTabs [aria-selected="true"] {
                color: #00FF00 !important;
            }
            
            /* Headers */
            h1, h2, h3, h4, h5, h6 {
                color: #FFA500 !important;
                font-family: Consolas, 'Courier New', monospace !important;
            }
            
            /* Text */
            p, div {
                color: #00FF00 !important;
                font-family: Consolas, 'Courier New', monospace !important;
            }
        </style>
    """, unsafe_allow_html=True)

apply_bloomberg_theme()

# API Configuration
TRADIER_BASE_URL = "https://sandbox.tradier.com/v1"

# Retry configuration with exponential backoff
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def exponential_backoff(attempt):
    """Calculate exponential backoff time"""
    return RETRY_DELAY * (2 ** attempt)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_options_chain(ticker, expiration_date, min_oi=100, min_volume=50, max_iv=90, option_type="Both"):
    """Fetch options chain with proper error handling"""
    logger.debug(f"Fetching options chain for {ticker}, expiry: {expiration_date}")
    
    try:
        # Use expiration date as is - it should already be in YYYY-MM-DD format
        exp_date_str = expiration_date
        logger.debug(f"Using expiration date: {exp_date_str}")
        
        # Make API request with retries
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1} to fetch options chain")
                response = requests.get(
                    f"{TRADIER_BASE_URL}/markets/options/chains",
                    params={
                        "symbol": ticker,
                        "expiration": exp_date_str,
                        "greeks": "true"
                    },
                    headers=HEADERS
                )
                response.raise_for_status()
                data = response.json()
                logger.debug(f"API Response: {data}")
                
                # Check for valid response structure
                if not isinstance(data, dict):
                    logger.warning(f"Invalid response format from API: {data}")
                    continue
                    
                if 'options' not in data:
                    logger.warning(f"No 'options' key in response for {ticker}")
                    return pd.DataFrame()
                    
                # Handle case where options is None or empty
                if data['options'] is None:
                    logger.warning(f"No options data available for {ticker}")
                    return pd.DataFrame()
                    
                # Handle case where options is a dict but has no 'option' key
                if isinstance(data['options'], dict) and 'option' not in data['options']:
                    logger.warning(f"No 'option' key in options data for {ticker}")
                    return pd.DataFrame()
                    
                options_list = data['options'].get('option', [])
                if not options_list:
                    logger.warning(f"Empty options list for {ticker}")
                    return pd.DataFrame()
                    
                logger.debug(f"Found {len(options_list)} raw options")
                
                df = pd.DataFrame(options_list)
                
                # Apply filters with logging
                original_len = len(df)
                logger.debug(f"Starting with {original_len} options")
                
                if option_type != "Both":
                    df = df[df['option_type'].str.lower() == option_type.lower()]
                    logger.debug(f"After option type filter: {len(df)} options")
                    
                # Filter by minimum OI and volume
                df = df[
                    (df['open_interest'] >= min_oi) &
                    (df['volume'] >= min_volume)
                ]
                logger.debug(f"After OI/Volume filter: {len(df)} options")
                
                if 'greeks' in df.columns:
                    # Apply IV threshold filter
                    df['iv'] = df['greeks'].apply(lambda x: float(x.get('mid_iv', 0)) if isinstance(x, dict) else 0)
                    df = df[df['iv'] <= max_iv]
                    logger.debug(f"After IV filter: {len(df)} options")
                    
                # Filter based on bid-ask spread
                filtered_options = []
                for opt in df.to_dict(orient='records'):
                    try:
                        bid = float(opt.get('bid', 0))
                        ask = float(opt.get('ask', 0))
                        
                        if ask > 0:
                            spread_percentage = (ask - bid) / ask * 100
                            if spread_percentage > 5:  # Max 5% spread
                                logger.debug(f"Option rejected due to high spread: {spread_percentage:.2f}%")
                                continue

                        # Validate premium is reasonable (typically < 20% of strike for near-the-money options)
                        if 'strike' in opt:
                            strike = float(opt['strike'])
                            premium = (bid + ask) / 2
                            if premium > (strike * 0.20):
                                logger.debug(f"Option rejected due to unusually high premium: ${premium:.2f} for strike ${strike:.2f}")
                                continue

                        # Validate delta values
                        if 'greeks' in opt and isinstance(opt['greeks'], dict):
                            delta = float(opt['greeks'].get('delta', 0))
                            option_type = opt.get('option_type', '').lower()
                            
                            # Delta should be between 0 and 1 for calls, -1 and 0 for puts
                            if option_type == 'call' and (delta < 0 or delta > 1):
                                logger.debug(f"Call option rejected due to invalid delta: {delta}")
                                continue
                            elif option_type == 'put' and (delta > 0 or delta < -1):
                                logger.debug(f"Put option rejected due to invalid delta: {delta}")
                                continue

                        filtered_options.append(opt)
                    except Exception as e:
                        logger.warning(f"Error processing option {opt}: {str(e)}")
                        continue
                        
                logger.info(f"Found {len(filtered_options)} valid options after all filters")
                return pd.DataFrame(filtered_options)
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch options chain after {max_retries} attempts: {str(e)}")
                    return pd.DataFrame()
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Retrying options fetch in {wait_time} seconds...")
                time.sleep(wait_time)
                
    except Exception as e:
        logger.error(f"Error fetching options chain: {str(e)}", exc_info=True)
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_historical_data(ticker, start_date=None, end_date=None):
    """
    Fetch historical price data with caching and proper error handling
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (datetime, optional): Start date for historical data
        end_date (datetime, optional): End date for historical data
        
    Returns:
        pd.DataFrame: Historical price data with technical indicators
    """
    logger.debug(f"Fetching historical data for {ticker}")
    try:
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=90)  # 3 months of data
            
        logger.debug(f"Date range: {start_date} to {end_date}")
        
        # Ensure proper date format
        start_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else start_date
        end_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else end_date
        
        response = requests.get(
            f"{TRADIER_BASE_URL}/markets/history",
            params={
                "symbol": ticker,
                "start": start_str,
                "end": end_str,
                "interval": "daily"
            },
            headers=HEADERS
        )
        response.raise_for_status()
        data = response.json()
        
        # Log the API response for debugging
        logger.debug(f"API Response: {data.keys()}")
        
        if 'history' not in data:
            logger.error(f"No history data in response for {ticker}: {data}")
            st.error(f"Could not fetch historical data for {ticker}. API response missing 'history' key.")
            return pd.DataFrame()
            
        if 'day' not in data['history'] or not data['history']['day']:
            logger.error(f"No daily data in history for {ticker}")
            st.error(f"No historical data available for {ticker} in the specified date range.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data['history']['day'])
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Convert and set date index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate technical indicators
        if not df.empty:
            try:
                df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
                df['SMA_50'] = df['close'].rolling(window=50, min_periods=1).mean()
                df['RSI'] = calculate_rsi(df['close'])
                df['MACD'] = df['close'].ewm(span=12, adjust=False, min_periods=1).mean() - df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
                rolling_mean = df['close'].rolling(window=20, min_periods=1).mean()
                rolling_std = df['close'].rolling(window=20, min_periods=1).std()
                df['BB_middle'] = rolling_mean
                df['BB_upper'] = rolling_mean + (rolling_std * 2)
                df['BB_lower'] = rolling_mean - (rolling_std * 2)
                logger.debug(f"Successfully calculated technical indicators for {ticker}")
            except Exception as e:
                logger.error(f"Error calculating technical indicators: {str(e)}", exc_info=True)
                st.warning("Could not calculate some technical indicators. Basic price data still available.")
        
        logger.info(f"Successfully fetched {len(df)} days of data for {ticker}")
        return df
        
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed for {ticker}: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return pd.DataFrame()
    except Exception as e:
        error_msg = f"Error fetching historical data for {ticker}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return pd.DataFrame()

def fetch_expirations(ticker):
    """Fetch available expiration dates for options"""
    url = f"{TRADIER_BASE_URL}/markets/options/expirations"
    params = {"symbol": ticker, "includeAllRoots": "true"}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'expirations' in data and 'date' in data['expirations']:
                # Filter dates based on requirements (1 week to 6 months)
                dates = data['expirations']['date']
                valid_dates = []
                now = datetime.now()
                for date_str in dates:
                    exp_date = datetime.strptime(date_str, '%Y-%m-%d')
                    days_to_expiry = (exp_date - now).days
                    if 7 <= days_to_expiry <= 180:
                        valid_dates.append(date_str)
                return valid_dates
            
            return []
            
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                logging.error(f"Failed to fetch expirations after {MAX_RETRIES} attempts: {str(e)}")
                raise
            time.sleep(RETRY_DELAY * (attempt + 1))

def validate_ticker(ticker):
    """Validate if a ticker symbol exists"""
    url = f"{TRADIER_BASE_URL}/markets/quotes"
    params = {"symbols": ticker}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'quotes' in data and 'quote' in data['quotes']:
                quote = data['quotes']['quote']
                return quote is not None and 'symbol' in quote
            
            return False
            
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                logging.error(f"Failed to validate ticker after {MAX_RETRIES} attempts: {str(e)}")
                raise
            time.sleep(RETRY_DELAY * (attempt + 1))

# Technical Analysis and Scoring Functions
def calculate_rsi(data, periods=14):
    """
    Calculate RSI using a simplified and robust method
    """
    try:
        # Convert data to float
        data = pd.to_numeric(data, errors='coerce')
        
        # Handle empty or invalid data
        if data.empty or data.isna().all():
            logger.warning("Empty or invalid data for RSI calculation")
            return pd.Series(50, index=data.index)

        # Calculate price changes
        delta = data.diff()

        # Handle all NaN deltas
        if delta.isna().all():
            logger.warning("No valid price changes for RSI calculation")
            return pd.Series(50, index=data.index)

        # Separate gains and losses
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)

        # Calculate average gain and loss
        avg_gain = gain.ewm(com=periods-1, min_periods=periods).mean()
        avg_loss = loss.ewm(com=periods-1, min_periods=periods).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        # Handle edge cases
        rsi = rsi.fillna(50)  # Fill NaN with neutral value
        rsi = rsi.clip(0, 100)  # Ensure values are between 0 and 100

        return rsi

    except Exception as e:
        logger.error(f"Error in RSI calculation: {str(e)}", exc_info=True)
        return pd.Series(50, index=data.index)

def calculate_macd(data):
    """Calculate MACD, Signal Line, and Histogram"""
    try:
        # Convert data to float
        data = pd.to_numeric(data, errors='coerce')
        
        # Handle empty or invalid data
        if data.empty or data.isna().all():
            logger.warning("Empty or invalid data for MACD calculation")
            return pd.DataFrame({
                'MACD': pd.Series(0, index=data.index),
                'Signal': pd.Series(0, index=data.index),
                'Histogram': pd.Series(0, index=data.index)
            })

        # Calculate EMAs
        exp1 = data.ewm(span=12, adjust=False, min_periods=1).mean()
        exp2 = data.ewm(span=26, adjust=False, min_periods=1).mean()
        
        # Calculate MACD line
        macd = exp1 - exp2
        
        # Calculate signal line
        signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
        
        # Calculate histogram
        hist = macd - signal

        # Fill NaN values
        macd = macd.fillna(0)
        signal = signal.fillna(0)
        hist = hist.fillna(0)

        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal,
            'Histogram': hist
        })

    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}", exc_info=True)
        return pd.DataFrame({
            'MACD': pd.Series(0, index=data.index),
            'Signal': pd.Series(0, index=data.index),
            'Histogram': pd.Series(0, index=data.index)
        })

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    try:
        # Convert data to float
        data = pd.to_numeric(data, errors='coerce')
        
        # Handle empty or invalid data
        if data.empty or data.isna().all():
            logger.warning("Empty or invalid data for Bollinger Bands calculation")
            return pd.DataFrame({
                'Upper': pd.Series(data.iloc[-1] if not data.empty else 0, index=data.index),
                'Middle': pd.Series(data.iloc[-1] if not data.empty else 0, index=data.index),
                'Lower': pd.Series(data.iloc[-1] if not data.empty else 0, index=data.index)
            })

        # Calculate middle band (simple moving average)
        middle = data.rolling(window=window, min_periods=1).mean()
        
        # Calculate standard deviation
        std = data.rolling(window=window, min_periods=1).std()
        
        # Calculate upper and lower bands
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        # Forward fill then backward fill NaN values
        middle = middle.ffill().bfill()
        upper = upper.ffill().bfill()
        lower = lower.ffill().bfill()

        return pd.DataFrame({
            'Upper': upper,
            'Middle': middle,
            'Lower': lower
        })

    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}", exc_info=True)
        return pd.DataFrame({
            'Upper': pd.Series(data.iloc[-1] if not data.empty else 0, index=data.index),
            'Middle': pd.Series(data.iloc[-1] if not data.empty else 0, index=data.index),
            'Lower': pd.Series(data.iloc[-1] if not data.empty else 0, index=data.index)
        })

def calculate_technical_indicators(df):
    """Calculate technical indicators for analysis"""
    try:
        if df.empty:
            logger.error("Empty DataFrame provided for technical analysis")
            return {}
            
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Ensure we have OHLCV data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error("Missing required OHLCV columns for technical analysis")
            return {}
            
        # Convert price columns to numeric, replacing errors with NaN
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Calculate technical indicators
        indicators = {}
        
        try:
            # Calculate RSI
            rsi = calculate_rsi(df['close'])
            indicators['RSI'] = {
                'value': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
                'data': rsi.tolist()
            }
            
            # Calculate MACD
            macd_data = calculate_macd(df['close'])
            indicators['MACD'] = {
                'value': macd_data['MACD'].iloc[-1] if not pd.isna(macd_data['MACD'].iloc[-1]) else 0,
                'signal': macd_data['Signal'].iloc[-1] if not pd.isna(macd_data['Signal'].iloc[-1]) else 0,
                'histogram': macd_data['Histogram'].iloc[-1] if not pd.isna(macd_data['Histogram'].iloc[-1]) else 0,
                'data': macd_data['MACD'].tolist(),
                'signal_data': macd_data['Signal'].tolist(),
                'histogram_data': macd_data['Histogram'].tolist()
            }
            
            # Calculate Bollinger Bands
            bb_data = calculate_bollinger_bands(df['close'])
            indicators['BB'] = {
                'upper': bb_data['Upper'].iloc[-1] if not pd.isna(bb_data['Upper'].iloc[-1]) else df['close'].iloc[-1],
                'middle': bb_data['Middle'].iloc[-1] if not pd.isna(bb_data['Middle'].iloc[-1]) else df['close'].iloc[-1],
                'lower': bb_data['Lower'].iloc[-1] if not pd.isna(bb_data['Lower'].iloc[-1]) else df['close'].iloc[-1],
                'upper_data': bb_data['Upper'].tolist(),
                'middle_data': bb_data['Middle'].tolist(),
                'lower_data': bb_data['Lower'].tolist()
            }
            
            # Calculate Moving Averages
            sma_20 = calculate_sma(df['close'], 20)
            sma_50 = calculate_sma(df['close'], 50)
            
            indicators['SMA_20'] = {
                'value': sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else df['close'].iloc[-1],
                'data': sma_20.tolist()
            }
            indicators['SMA_50'] = {
                'value': sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else df['close'].iloc[-1],
                'data': sma_50.tolist()
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error processing indicator values: {str(e)}")
            return {}
            
    except Exception as e:
        logger.error(f"Technical indicators calculation failed: {str(e)}", exc_info=True)
        return {}

def get_indicator_description(key):
    """Get description for technical indicators"""
    try:
        descriptions = {
            'SMA_20': '20-day Simple Moving Average',
            'SMA_50': '50-day Simple Moving Average',
            'RSI': 'Relative Strength Index',
            'MACD': 'Moving Average Convergence Divergence',
            'BB': 'Bollinger Bands'
        }
        return descriptions.get(key, 'Technical Indicator')
    except Exception as e:
        logger.error(f"Error getting indicator description: {str(e)}")
        return 'Technical Indicator'

def calculate_volatility_risk(historical_data):
    """
    Calculate historical volatility risk score
    
    Args:
        historical_data (pd.DataFrame): DataFrame with historical price data
        
    Returns:
        float: Volatility risk score between 0 and 1
    """
    try:
        if historical_data.empty:
            logger.warning("Historical data is empty, cannot calculate volatility risk")
            return 0.5  # Return neutral score
            
        # Calculate daily returns
        historical_data['returns'] = np.log(
            historical_data['close'] / historical_data['close'].shift(1)
        )
        
        # Calculate 30-day historical volatility (annualized)
        rolling_std = historical_data['returns'].rolling(window=30).std()
        hist_vol = rolling_std.iloc[-1] * np.sqrt(252) * 100  # Annualize and convert to percentage
        
        if np.isnan(hist_vol):
            logger.warning("Could not calculate historical volatility, using neutral score")
            return 0.5
            
        # Normalize to 0-1 scale with reasonable bounds
        # Most stocks have historical volatility between 15% and 100%
        vol_score = 1 - min(max(hist_vol - 15, 0) / 85, 1)
        
        logger.debug(f"Calculated volatility risk score: {vol_score:.2f} (Historical Vol: {hist_vol:.1f}%)")
        return vol_score
        
    except Exception as e:
        logger.error(f"Error calculating volatility risk: {str(e)}", exc_info=True)
        return 0.5  # Return neutral score on error

def analyze_greeks_risk(option, risk_appetite):
    """
    Analyze option Greeks to determine risk score
    
    Args:
        option (pd.Series): Option data including Greeks
        risk_appetite (int): Risk appetite score (1-10)
        
    Returns:
        float: Greeks-based risk score between 0 and 1
    """
    try:
        # Extract Greeks from the option data
        greeks = option.get('greeks', {})
        if not isinstance(greeks, dict):
            logger.warning("Greeks data not available or invalid")
            return 0.5  # Return neutral score
            
        # Get individual Greeks
        delta = abs(float(greeks.get('delta', 0)))
        theta = abs(float(greeks.get('theta', 0)))
        vega = abs(float(greeks.get('vega', 0)))
        gamma = abs(float(greeks.get('gamma', 0)))
        
        # Normalize theta and vega relative to option price
        premium = (float(option['bid']) + float(option['ask'])) / 2
        if premium > 0:
            theta_norm = min(abs(theta / premium), 1)  # Daily theta as % of premium
            vega_norm = min(abs(vega / premium), 1)   # Vega as % of premium
        else:
            theta_norm = 1
            vega_norm = 1
            
        # Normalize gamma
        gamma_norm = min(gamma * 100, 1)  # Scale gamma to 0-1
        
        # Weight Greeks based on risk appetite
        if risk_appetite <= 3:  # Conservative
            weights = {
                'delta': 0.2,   # Less directional exposure
                'theta': 0.4,   # Focus on time decay risk
                'vega': 0.3,    # Watch volatility risk
                'gamma': 0.1    # Less concerned with gamma risk
            }
        elif risk_appetite <= 7:  # Moderate
            weights = {
                'delta': 0.3,   # Balanced directional exposure
                'theta': 0.3,   # Balanced time decay risk
                'vega': 0.2,    # Moderate volatility risk concern
                'gamma': 0.2    # Moderate gamma risk concern
            }
        else:  # Aggressive
            weights = {
                'delta': 0.4,   # More directional exposure
                'theta': 0.2,   # Less concerned with time decay
                'vega': 0.2,    # Less concerned with volatility risk
                'gamma': 0.2    # Equal gamma risk concern
            }
            
        # Calculate weighted Greek scores
        # Invert some scores so that 1 is always better
        delta_score = 1 - delta  # Lower delta is safer
        theta_score = 1 - theta_norm  # Lower theta is safer
        vega_score = 1 - vega_norm   # Lower vega is safer
        gamma_score = 1 - gamma_norm  # Lower gamma is safer
        
        # Combine scores using weights
        greek_score = (
            delta_score * weights['delta'] +
            theta_score * weights['theta'] +
            vega_score * weights['vega'] +
            gamma_score * weights['gamma']
        )
        
        logger.debug(
            f"Greeks risk analysis - Delta: {delta_score:.2f}, Theta: {theta_score:.2f}, "
            f"Vega: {vega_score:.2f}, Gamma: {gamma_score:.2f}, Final: {greek_score:.2f}"
        )
        
        return greek_score
        
    except Exception as e:
        logger.error(f"Error analyzing Greeks risk: {str(e)}", exc_info=True)
        return 0.5  # Return neutral score on error

def score_options(options_df, indicators, current_price, budget, risk_appetite, sentiment=None, historical_data=None):
    """Score options based on technical, fundamental, and Greek factors"""
    try:
        if options_df.empty:
            return pd.DataFrame()
            
        # Add current price to the DataFrame
        options_df['underlying_price'] = current_price
        
        # Create a copy to avoid modifying the original
        scored_options = options_df.copy()
        
        # Initialize score column
        scored_options['score'] = 0.0
        
        # Calculate volatility risk score if historical data is available
        vol_risk_score = calculate_volatility_risk(historical_data) if historical_data is not None else 0.5
        logger.info(f"Overall volatility risk score: {vol_risk_score:.2f}")
        
        for idx, option in scored_options.iterrows():
            try:
                # Basic score components
                price_score = 0
                technical_score = 0
                risk_score = 0
                
                # Price based scoring
                strike = float(option['strike'])
                if option['option_type'].lower() == 'call':
                    price_score = 1 - (abs(strike - current_price) / current_price)
                else:  # put
                    price_score = 1 - (abs(current_price - strike) / current_price)
                    
                # Technical analysis scoring
                if indicators:
                    rsi = indicators.get('RSI', {}).get('value', 50)
                    macd = indicators.get('MACD', {}).get('value', 0)
                    
                    # RSI scoring
                    if option['option_type'].lower() == 'call':
                        if 30 < rsi < 70:  # Neutral
                            technical_score += 0.5
                        elif rsi <= 30:  # Oversold - good for calls
                            technical_score += 1
                    else:  # put
                        if 30 < rsi < 70:  # Neutral
                            technical_score += 0.5
                        elif rsi >= 70:  # Overbought - good for puts
                            technical_score += 1
                            
                    # MACD scoring
                    if option['option_type'].lower() == 'call':
                        technical_score += 1 if macd > 0 else 0
                    else:  # put
                        technical_score += 1 if macd < 0 else 0
                        
                # Risk scoring based on volume, open interest, and Greeks
                volume = float(option['volume'])
                open_interest = float(option['open_interest'])
                
                volume_score = min(volume / 1000, 1)  # Scale volume score
                oi_score = min(open_interest / 1000, 1)  # Scale OI score
                
                # Get Greeks-based risk score
                greek_score = analyze_greeks_risk(option, risk_appetite)
                
                # Combine all risk metrics
                risk_score = (volume_score + oi_score + vol_risk_score + greek_score) / 4
                
                # Adjust for bid-ask spread
                bid = float(option['bid'])
                ask = float(option['ask'])
                spread = (ask - bid) / ((bid + ask) / 2)
                spread_penalty = max(0, 1 - spread)
                
                # Calculate final score with more weight on Greeks for conservative strategies
                if risk_appetite <= 3:
                    base_score = (price_score * 0.2 + technical_score * 0.3 + risk_score * 0.5)
                elif risk_appetite <= 7:
                    base_score = (price_score * 0.3 + technical_score * 0.4 + risk_score * 0.3)
                else:
                    base_score = (price_score * 0.4 + technical_score * 0.4 + risk_score * 0.2)
                    
                final_score = base_score * spread_penalty
                
                # Adjust for sentiment if available
                if sentiment is not None:
                    sentiment_factor = sentiment / 100  # Convert to 0-1 scale
                    if option['option_type'].lower() == 'call':
                        final_score *= (1 + sentiment_factor) / 2
                    else:  # put
                        final_score *= (2 - sentiment_factor) / 2
                        
                # Adjust for risk appetite (1-10 scale)
                risk_factor = risk_appetite / 5  # Convert to 0.2-2 scale
                final_score *= risk_factor
                
                # Store the score
                scored_options.at[idx, 'score'] = final_score * 100  # Convert to 0-100 scale
                
                # Store individual component scores for analysis
                scored_options.at[idx, 'greek_score'] = greek_score * 100
                scored_options.at[idx, 'volatility_score'] = vol_risk_score * 100
                scored_options.at[idx, 'technical_score'] = technical_score * 100
                
            except Exception as e:
                logger.warning(f"Error scoring option {option}: {str(e)}")
                scored_options.at[idx, 'score'] = 0
                
        logger.info(f"Scored {len(scored_options)} options successfully")
        return scored_options
        
    except Exception as e:
        logger.error(f"Error in score_options: {str(e)}", exc_info=True)
        return pd.DataFrame()

def analyze_pmcc_strategy(options_chain, current_price, budget, indicators):
    """
    Analyze Poor Man's Covered Call strategy opportunities
    """
    logger.debug(f"Analyzing PMCC strategy with current price: {current_price}, budget: {budget}")
    
    try:
        # Convert indicators DataFrame to required format if needed
        if isinstance(indicators, pd.DataFrame):
            logger.debug("Converting indicators DataFrame to dictionary format")
            indicators_dict = {}
            for col in ['RSI', 'MACD', 'BB_upper', 'BB_lower']:
                if col in indicators.columns:
                    key = col.lower().replace('_upper', '_high').replace('_lower', '_low')
                    indicators_dict[key] = indicators[col]
                else:
                    logger.warning(f"Missing indicator column: {col}")
                    indicators_dict[col.lower().replace('_upper', '_high').replace('_lower', '_low')] = pd.Series()
        else:
            indicators_dict = indicators
            
        # Validate indicators
        required_indicators = ['rsi', 'macd', 'bb_high', 'bb_low']
        if not indicators_dict or not all(k in indicators_dict for k in required_indicators):
            missing = [k for k in required_indicators if k not in indicators_dict]
            logger.error(f"Missing required technical indicators: {missing}")
            st.warning("Unable to analyze PMCC strategy: Missing required technical indicators")
            return None
            
        # Technical Analysis Validation
        latest_rsi = indicators_dict['rsi'].iloc[-1] if not indicators_dict['rsi'].empty else 50
        if latest_rsi > 80 or latest_rsi < 20:
            logger.warning(f"RSI {latest_rsi} is outside acceptable range (20-80)")
            st.info(f"PMCC not recommended: RSI ({latest_rsi:.1f}) indicates overbought/oversold conditions")
            return None
            
        # Check if price is within Bollinger Bands
        if not indicators_dict['bb_high'].empty and not indicators_dict['bb_low'].empty:
            bb_high = indicators_dict['bb_high'].iloc[-1]
            bb_low = indicators_dict['bb_low'].iloc[-1]
            if current_price > bb_high or current_price < bb_low:
                logger.warning(f"Price ({current_price}) is outside Bollinger Bands ({bb_low:.2f} - {bb_high:.2f})")
                st.info("PMCC not recommended: Price is outside Bollinger Bands indicating high volatility")
                return None
                
        # Verify MACD trend is bullish for PMCC
        if not indicators_dict['macd'].empty:
            macd_value = indicators_dict['macd'].iloc[-1]
            if macd_value <= 0:
                logger.warning(f"MACD ({macd_value:.3f}) indicates bearish trend")
                st.info("PMCC not recommended: MACD indicates bearish trend")
                return None
                
        # Filter for LEAPS (at least 1 year expiration)
        one_year_out = datetime.now().date() + timedelta(days=365)
        leaps_options = options_chain[
            (options_chain['option_type'] == 'call') & 
            (pd.to_datetime(options_chain['expiration_date']).dt.date >= one_year_out)
        ]
        
        if leaps_options.empty:
            logger.warning("No suitable LEAPS options found (expiration >= 1 year)")
            st.warning("No LEAPS options found with expiration date >= 1 year")
            return None
            
        # Filter for short-term calls (less than 45 days)
        short_calls = options_chain[
            (options_chain['option_type'] == 'call') & 
            (pd.to_datetime(options_chain['expiration_date']).dt.date <= datetime.now().date() + timedelta(days=45))
        ]
        
        if short_calls.empty:
            logger.warning("No suitable short-term calls found (expiration <= 45 days)")
            st.warning("No short-term call options found with expiration <= 45 days")
            return None
            
        # Find best LEAPS candidate
        leaps_candidates = []
        for _, leap in leaps_options.iterrows():
            try:
                # Validate required fields
                required_fields = ['strike', 'ask', 'greeks']
                if not all(field in leap for field in required_fields):
                    missing = [f for f in required_fields if f not in leap]
                    logger.warning(f"Missing required fields in LEAPS option: {missing}")
                    continue
                    
                leap_cost = float(leap['ask']) * 100
                if leap_cost > budget * 0.8:  # LEAPS should not use more than 80% of budget
                    logger.debug(f"LEAPS cost ({leap_cost}) exceeds 80% of budget ({budget})")
                    continue
                    
                # Calculate intrinsic and extrinsic value
                intrinsic = max(0, current_price - float(leap['strike']))
                extrinsic = float(leap['ask']) - intrinsic
                
                if 'greeks' in leap and 'delta' in leap['greeks']:
                    leap_delta = float(leap['greeks']['delta'])
                    if leap_delta < 0.7:  # LEAPS should be deep ITM
                        logger.debug(f"LEAPS delta ({leap_delta}) < 0.7, skipping")
                        continue
                        
                    leaps_candidates.append({
                        'option': leap,
                        'cost': leap_cost,
                        'delta': leap_delta,
                        'extrinsic': extrinsic,
                        'intrinsic': intrinsic
                    })
            except Exception as e:
                logger.warning(f"Error processing LEAPS option: {str(e)}")
                continue
                
        if not leaps_candidates:
            logger.warning("No valid LEAPS candidates found after filtering")
            st.warning("No suitable LEAPS options found matching criteria (delta >= 0.7, within budget)")
            return None
            
        # Sort LEAPS by delta and lowest extrinsic value
        leaps_candidates.sort(key=lambda x: (-x['delta'], x['extrinsic']))
        best_leap = leaps_candidates[0]['option']
        logger.info(f"Selected best LEAPS: Strike={best_leap['strike']}, Delta={leaps_candidates[0]['delta']:.2f}, Cost=${leaps_candidates[0]['cost']:.2f}")
        
        # Find matching short call
        short_candidates = []
        for _, short in short_calls.iterrows():
            try:
                required_fields = ['strike', 'bid', 'greeks']
                if not all(field in short for field in required_fields):
                    missing = [f for f in required_fields if f not in short]
                    logger.warning(f"Missing required fields in short call: {missing}")
                    continue
                    
                if float(short['strike']) <= float(best_leap['strike']):
                    logger.debug(f"Short call strike ({short['strike']}) <= LEAPS strike ({best_leap['strike']})")
                    continue  # Short call strike must be higher than LEAPS
                    
                if 'greeks' in short and 'delta' in short['greeks']:
                    short_delta = float(short['greeks']['delta'])
                    if short_delta >= float(best_leap['greeks']['delta']):
                        logger.debug(f"Short call delta ({short_delta}) >= LEAPS delta ({best_leap['greeks']['delta']})")
                        continue  # Short call delta must be lower than LEAPS
                        
                    short_candidates.append({
                        'option': short,
                        'premium': float(short['bid']) * 100,
                        'delta': short_delta,
                        'strike': float(short['strike'])
                    })
            except Exception as e:
                logger.warning(f"Error processing short call option: {str(e)}")
                continue
                
        if not short_candidates:
            logger.warning("No valid short call candidates found")
            st.warning("No suitable short call options found with appropriate strike/delta")
            return None
            
        # Sort short calls by highest premium and appropriate delta
        short_candidates.sort(key=lambda x: (x['premium'], -abs(x['delta'] - 0.3)))
        best_short = short_candidates[0]['option']
        logger.info(f"Selected best short call: Strike={best_short['strike']}, Delta={short_candidates[0]['delta']:.2f}, Premium=${short_candidates[0]['premium']:.2f}")
        
        st.success("Found valid PMCC strategy combination!")
        return (best_leap, best_short)
        
    except Exception as e:
        logger.error(f"Error in analyze_pmcc_strategy: {str(e)}", exc_info=True)
        st.error("An error occurred while analyzing PMCC strategy. Please check the logs.")
        return None

# Streamlit UI Setup

# Sidebar Configuration
st.sidebar.header("Options Evaluation Program")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_current_price(ticker):
    """
    Fetch current stock price with caching and proper error handling
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        float: Current stock price or None if error occurs
    """
    logger.debug(f"Fetching current price for {ticker}")
    try:
        response = requests.get(
            f"{TRADIER_BASE_URL}/markets/quotes",
            params={"symbols": ticker},
            headers=HEADERS
        )
        response.raise_for_status()
        data = response.json()
        
        if 'quotes' in data and 'quote' in data['quotes']:
            return float(data['quotes']['quote']['last'])
        else:
            logger.error(f"Invalid response format for {ticker}: {data}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching price for {ticker}: {str(e)}", exc_info=True)
        return None

def test_api_connection():
    """
    Test the Tradier API connection with proper error handling and validation
    Verifies both market status and quote endpoints are working
    Displays current market status and next market event
    
    Returns:
        bool: True if test passes, False otherwise
    """
    logger.info("Testing API connection...")
    test_ticker = "SPY"  # Using a reliable ticker for testing
    
    try:
        # Test market status endpoint first
        status_url = f"{TRADIER_BASE_URL}/markets/clock"
        logger.debug("Testing market status endpoint")
        
        response = requests.get(
            status_url, 
            headers=HEADERS
        )
        response.raise_for_status()
        
        status_data = response.json()
        if 'clock' not in status_data:
            logger.error("Invalid response format from market status endpoint")
            st.error("API test failed: Invalid response format")
            return False
            
        # Test quote endpoint
        quote_url = f"{TRADIER_BASE_URL}/markets/quotes"
        logger.debug(f"Testing quote endpoint with {test_ticker}")
        
        response = requests.get(
            quote_url, 
            params={"symbols": test_ticker}, 
            headers=HEADERS
        )
        response.raise_for_status()
        
        quote_data = response.json()
        if 'quotes' not in quote_data or 'quote' not in quote_data['quotes']:
            logger.error("Invalid response format from quote endpoint")
            st.error("API test failed: Invalid response format")
            return False
            
        # All tests passed
        logger.info("API connection test successful")
        st.success(" API connection test successful!")
        
        # Display current market status
        market_status = status_data['clock']['state']
        next_market_event = status_data['clock']['next_change']
        st.info(f"Market Status: {market_status}\nNext Market Event: {next_market_event}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        error_msg = f"API connection test failed: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return False
    except Exception as e:
        error_msg = f"Unexpected error during API test: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return False

def get_date_ranges():
    """
    Generate date ranges for options analysis
    Returns a dict with human-readable labels and their corresponding date ranges
    All ranges start from 1 week out and extend to their respective end dates
    """
    today = datetime.now()
    min_date = today + timedelta(days=7)  # Start from 1 week out
    
    ranges = {
        "1 Week": (min_date, today + timedelta(days=14)),
        "1 Month": (min_date, today + timedelta(days=30)),
        "3 Months": (min_date, today + timedelta(days=90)),
        "6 Months": (min_date, today + timedelta(days=180)),
        "1 Year": (min_date, today + timedelta(days=365)),
        "2 Years": (min_date, today + timedelta(days=730))
    }
    return ranges

def fetch_options_for_range(ticker, start_date, end_date, min_oi=100, min_volume=50, max_iv=90, option_type="Both"):
    """
    Fetch and analyze options for a given date range
    
    Args:
        ticker (str): Stock ticker
        start_date (datetime): Start of date range (minimum 1 week out)
        end_date (datetime): End of date range (maximum 6 months out)
        min_oi (int): Minimum open interest
        min_volume (int): Minimum volume
        max_iv (float): Maximum implied volatility
        option_type (str): Option type (Call, Put, Both)
    
    Returns:
        pd.DataFrame: Filtered and scored options chain
    """
    logger.debug(f"Fetching options for {ticker} between {start_date} and {end_date}")
    logger.debug(f"Type of start_date: {type(start_date)}, Type of end_date: {type(end_date)}")
    
    try:
        # Convert datetime objects to date objects
        if isinstance(start_date, datetime):
            start_date = start_date.date()
            logger.debug(f"Converted start_date to date object: {start_date}")
        if isinstance(end_date, datetime):
            end_date = end_date.date()
            logger.debug(f"Converted end_date to date object: {end_date}")
            
        # Ensure we have date objects
        if not isinstance(start_date, (datetime, date)):
            start_date = datetime.strptime(str(start_date), "%Y-%m-%d").date()
            logger.debug(f"Parsed start_date string to date object: {start_date}")
        if not isinstance(end_date, (datetime, date)):
            end_date = datetime.strptime(str(end_date), "%Y-%m-%d").date()
            logger.debug(f"Parsed end_date string to date object: {end_date}")
        
        # Get all available expiration dates
        logger.debug("Requesting expiration dates from Tradier API...")
        response = requests.get(
            f"{TRADIER_BASE_URL}/markets/options/expirations",
            params={"symbol": ticker},
            headers=HEADERS
        )
        response.raise_for_status()
        data = response.json()
        
        logger.debug(f"API Response for expirations: {data}")
        
        if 'expirations' not in data or 'date' not in data['expirations']:
            logger.error(f"No expiration dates found for {ticker}. API Response: {data}")
            return pd.DataFrame()
            
        # Filter expiration dates within our range
        valid_dates = []
        for exp_date in data['expirations']['date']:
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d').date()
            logger.debug(f"Comparing expiry {exp_datetime} with range {start_date} to {end_date}")
            if start_date <= exp_datetime <= end_date:
                valid_dates.append(exp_date)  # Store the original date string
                logger.debug(f"Found valid expiration date: {exp_date}")
        
        if not valid_dates:
            logger.warning(f"No expiration dates found between {start_date} and {end_date}")
            logger.debug(f"Available dates: {data['expirations']['date']}")
            st.warning(f"No option expiration dates found between {start_date} and {end_date}")
            return pd.DataFrame()
            
        # Sort dates for chronological processing
        valid_dates.sort()
        
        # Fetch options for each valid date
        all_options = []
        with st.spinner(f"Fetching options for {len(valid_dates)} expiration dates..."):
            for exp_date in valid_dates:
                logger.debug(f"Fetching options chain for {exp_date}")
                options_df = fetch_options_chain(ticker, exp_date, min_oi, min_volume, max_iv, option_type)
                
                if not options_df.empty:
                    logger.debug(f"Found {len(options_df)} options for {exp_date}")
                    all_options.append(options_df)
                else:
                    logger.debug(f"No valid options found for {exp_date}")
                    
        if not all_options:
            logger.warning("No valid options found in the specified date range")
            st.warning("No valid options found that meet the minimum criteria (volume, open interest, etc.)")
            return pd.DataFrame()
            
        # Combine all options and sort by expiration date
        combined_options = pd.concat(all_options, ignore_index=True)
        combined_options['expiration_date'] = pd.to_datetime(combined_options['expiration_date'])
        combined_options = combined_options.sort_values('expiration_date')
        
        logger.info(f"Successfully fetched total of {len(combined_options)} options")
        return combined_options
        
    except Exception as e:
        logger.error(f"Error fetching options for date range: {str(e)}", exc_info=True)
        st.error(f"Error fetching options: {str(e)}")
        return pd.DataFrame()

def display_pmcc_analysis(pmcc_recommendations):
    """
    Display Poor Man's Covered Call analysis
    
    Args:
        pmcc_recommendations (tuple): Tuple containing (LEAPS, short call) recommendations
    """
    if not pmcc_recommendations:
        return
        
    leap, short = pmcc_recommendations
    
    try:
        # Create expander for PMCC Strategy Details
        with st.expander("PMCC Strategy Details", expanded=True):
            # Calculate key metrics
            leap_cost = float(leap['ask']) * 100
            short_premium = float(short['bid']) * 100
            net_debit = leap_cost - short_premium
            max_profit = (float(short['strike']) - float(leap['strike'])) * 100 + short_premium
            breakeven = float(leap['strike']) + net_debit/100
            
            # Display strategy overview
            st.subheader("Strategy Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**LEAPS Call (Long)**")
                st.write(f"Strike: ${float(leap['strike']):.2f}")
                st.write(f"Expiration: {leap['expiration_date']}")
                st.write(f"Ask: ${float(leap['ask']):.2f}")
                st.write(f"Total Cost: ${leap_cost:.2f}")
                if 'greeks' in leap:
                    st.write(f"Delta: {float(leap['greeks'].get('delta', 0)):.3f}")
                    st.write(f"Theta: {float(leap['greeks'].get('theta', 0)):.3f}")
                
            with col2:
                st.markdown("**Short Call**")
                st.write(f"Strike: ${float(short['strike']):.2f}")
                st.write(f"Expiration: {short['expiration_date']}")
                st.write(f"Bid: ${float(short['bid']):.2f}")
                st.write(f"Premium: ${short_premium:.2f}")
                if 'greeks' in short:
                    st.write(f"Delta: {float(short['greeks'].get('delta', 0)):.3f}")
                    st.write(f"Theta: {float(short['greeks'].get('theta', 0)):.3f}")
            
            # Display key metrics
            st.subheader("Key Metrics")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Net Debit", f"${net_debit:.2f}")
            with metrics_col2:
                st.metric("Max Profit", f"${max_profit:.2f}")
            with metrics_col3:
                st.metric("Breakeven", f"${breakeven:.2f}")
            
            # Strategy explanation
            st.subheader("Strategy Analysis")
            st.markdown("""
            **Key Points:**
            - The LEAPS call provides leveraged upside exposure with reduced time decay
            - The short call generates income and reduces the cost basis
            - Maximum profit occurs if the stock price is at or above the short call strike at expiration
            - The position can be rolled forward to continue generating income
            
            **Risk Management:**
            - Monitor the position if the stock price approaches the short strike
            - Consider rolling the short call up and out if the stock rallies strongly
            - Watch for earnings and other major events that could impact volatility
            """)
            
            # Display profit curve
            st.subheader("Profit/Loss Profile")
            plot_pmcc_profit_curve(pmcc_recommendations, float(leap['underlying_price']))
            
    except Exception as e:
        logger.error(f"Error displaying PMCC analysis: {str(e)}", exc_info=True)
        st.error("An error occurred while displaying the PMCC analysis")

def display_options_analysis(scored_options, budget=200, historical_data=None, indicators=None):
    """Display options analysis including technical analysis"""
    try:
        if scored_options.empty:
            st.warning("No options data available for analysis")
            return
            
        # Create filtered views
        calls = scored_options[scored_options['option_type'] == 'call'].sort_values('score', ascending=False)
        puts = scored_options[scored_options['option_type'] == 'put'].sort_values('score', ascending=False)
        
        # Display top options in tabs
        option_tabs = st.tabs([" Call Options", " Put Options"])
        
        with option_tabs[0]:
            st.write("### Top Call Options")
            if not calls.empty:
                display_options_table(calls)
            else:
                st.info("No call options available")
                
        with option_tabs[1]:
            st.write("### Top Put Options")
            if not puts.empty:
                display_options_table(puts)
            else:
                st.info("No put options available")
                
        # Create tabs for different analyses
        analysis_tab1, analysis_tab2 = st.tabs([" Profit/Loss Analysis", " Technical Analysis"])
        
        with analysis_tab1:
            # Display P/L chart for selected option
            st.write("### Profit/Loss Analysis")
            
            # Option selection
            option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
            df = calls if option_type == "Call" else puts
            
            if not df.empty:
                selected_option = st.selectbox(
                    "Select Option for P/L Analysis",
                    df.index,
                    format_func=lambda x: f"Strike: ${df.loc[x, 'strike']}, Premium: ${df.loc[x, 'ask']}, Score: {df.loc[x, 'score']:.1f}"
                )
                
                if selected_option:
                    option = df.loc[selected_option]
                    current_price = float(option['underlying_price'])
                    plot_profit_loss(option, current_price)
            else:
                st.info(f"No {option_type.lower()} options available for analysis")
                
        with analysis_tab2:
            if historical_data is not None and indicators is not None:
                st.write("### Technical Analysis")
                
                # Calculate and display trends
                trends = calculate_trends(historical_data, indicators)
                
                # Display trend information in a prominent way
                st.markdown("## Market Trends")
                
                # Create two columns for short and long term trends
                trend_col1, trend_col2 = st.columns(2)
                
                with trend_col1:
                    st.markdown("### Short-Term Trend")
                    signal = trends['short_term']['signal']
                    color = {
                        'STRONG BULLISH': 'green',
                        'BULLISH': 'lightgreen',
                        'NEUTRAL': 'gray',
                        'BEARISH': 'pink',
                        'STRONG BEARISH': 'red'
                    }.get(signal, 'gray')
                    
                    st.markdown(f"<h2 style='color: {color};'>{signal}</h2>", unsafe_allow_html=True)
                    st.markdown("#### Supporting Evidence:")
                    for reason in trends['short_term']['reasons']:
                        st.markdown(f"- {reason}")
                        
                with trend_col2:
                    st.markdown("### Long-Term Trend")
                    signal = trends['long_term']['signal']
                    color = {
                        'STRONG BULLISH': 'green',
                        'BULLISH': 'lightgreen',
                        'NEUTRAL': 'gray',
                        'BEARISH': 'pink',
                        'STRONG BEARISH': 'red'
                    }.get(signal, 'gray')
                    
                    st.markdown(f"<h2 style='color: {color};'>{signal}</h2>", unsafe_allow_html=True)
                    st.markdown("#### Supporting Evidence:")
                    for reason in trends['long_term']['reasons']:
                        st.markdown(f"- {reason}")
                
                st.markdown("---")
                
                # Create and display technical analysis plot
                try:
                    tech_fig = plot_technical_analysis(historical_data, indicators)
                    st.plotly_chart(tech_fig, use_container_width=True)
                    
                    # Display current indicator values in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'RSI' in indicators:
                            rsi = indicators['RSI']['value']
                            st.metric("RSI", f"{rsi:.1f}", 
                                    delta="Overbought" if rsi > 70 else "Oversold" if rsi < 30 else None)
                            
                    with col2:
                        if 'MACD' in indicators:
                            macd = indicators['MACD']['value']
                            signal = indicators['MACD']['signal']
                            st.metric("MACD", f"{macd:.2f}")
                            st.metric("Signal", f"{signal:.2f}")
                            
                    with col3:
                        if 'BB' in indicators:
                            bb = indicators['BB']
                            st.metric("BB Upper", f"${bb['upper']:.2f}")
                            st.metric("BB Middle", f"${bb['middle']:.2f}")
                            st.metric("BB Lower", f"${bb['lower']:.2f}")
                except Exception as e:
                    logger.error(f"Error plotting technical analysis: {str(e)}", exc_info=True)
                    st.error("Error displaying technical analysis plot")
            else:
                st.info("Technical analysis data not available")
                
    except Exception as e:
        logger.error(f"Error in display_options_analysis: {str(e)}", exc_info=True)
        st.error("Error displaying options analysis")

def create_options_analysis_table(options_df, budget=200):
    """Create an interactive table with detailed options analysis"""
    logger.debug("Creating options analysis table")
    
    try:
        if options_df.empty:
            logger.warning("Options dataframe is empty")
            st.warning("No options available for detailed analysis")
            return
            
        # Create a copy to avoid modifying original
        display_df = options_df.copy()
        
        # Extract Greeks from the 'greeks' dictionary column if it exists
        if 'greeks' in display_df.columns:
            logger.debug("Extracting Greeks from dictionary column")
            for greek in ['delta', 'gamma', 'theta', 'vega', 'mid_iv']:
                display_df[greek] = display_df['greeks'].apply(
                    lambda x: float(x.get(greek, 0)) if isinstance(x, dict) else 0
                )
        
        # Calculate premiums and total prices
        display_df['Premium'] = (display_df['bid'] + display_df['ask']) / 2
        display_df['Total Cost'] = display_df['Premium'] * 100  # Standard contract size
        
        # Calculate maximum loss for each option
        display_df['Max Loss'] = display_df.apply(lambda row: 
            row['Total Cost'] if row['option_type'].lower() == 'call' 
            else row['strike'] * 100 if row['option_type'].lower() == 'put'
            else None, axis=1
        )
        
        # Calculate percentage of budget
        display_df['Budget %'] = (display_df['Total Cost'] / budget * 100).round(1)
        
        # Select and rename columns for display
        columns_map = {
            'score': 'Score',  # Move score to the top of the map
            'strike': 'Strike',
            'option_type': 'Type',
            'expiration_date': 'Expiration',
            'Premium': 'Premium/Share',
            'Total Cost': 'Total Cost',
            'Budget %': 'Budget %',
            'Max Loss': 'Max Loss',
            'bid': 'Bid',
            'ask': 'Ask',
            'volume': 'Volume',
            'open_interest': 'OI',
            'mid_iv': 'IV',
            'delta': 'Delta',
            'theta': 'Theta'
        }
        
        # Select only existing columns
        available_columns = [col for col in columns_map.keys() if col in display_df.columns]
        display_df = display_df[available_columns].copy()
        
        # Rename columns
        display_df.rename(columns={old: new for old, new in columns_map.items() if old in available_columns}, inplace=True)
        
        # Format numeric columns
        if 'Score' in display_df.columns:
            display_df['Score'] = display_df['Score'].round(2)
        for col in ['Bid', 'Ask', 'IV', 'Delta', 'Theta']:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(4)
                
        # Format Premium and Total Cost
        if 'Premium/Share' in display_df.columns:
            display_df['Premium/Share'] = pd.to_numeric(display_df['Premium/Share'], errors='coerce').round(2)
        if 'Total Cost' in display_df.columns:
            display_df['Total Cost'] = pd.to_numeric(display_df['Total Cost'], errors='coerce').round(2)
        if 'Max Loss' in display_df.columns:
            display_df['Max Loss'] = pd.to_numeric(display_df['Max Loss'], errors='coerce').round(2)
        
        # Sort by Score in descending order
        if 'Score' in display_df.columns:
            display_df = display_df.sort_values('Score', ascending=False)
            
            # Ensure Score is the first column
            cols = ['Score'] + [col for col in display_df.columns if col != 'Score']
            display_df = display_df[cols]
        
        logger.info(f"Created analysis table with {len(display_df)} rows")
        
        # Display the table in Streamlit with enhanced formatting
        st.subheader(" Options Ranked by Score")
        
        # Display budget warning if any option exceeds 20% of budget
        budget_warnings = display_df[display_df['Budget %'] > 20]
        if not budget_warnings.empty:
            st.warning(" Some options exceed 20% of your budget! Consider reducing position size or increasing budget.")
        
        st.dataframe(
            display_df.style
            .format({
                'Score': '{:.2f}',
                'Strike': '${:.2f}',
                'Premium/Share': '${:.2f}',
                'Total Cost': '${:.2f}',
                'Max Loss': '${:.2f}',
                'Budget %': '{:.1f}%',
                'Bid': '${:.2f}',
                'Ask': '${:.2f}',
                'IV': '{:.1%}',
                'Delta': '{:.3f}',
                'Theta': '{:.3f}'
            })
            .set_properties(**{
                'font-weight': 'bold',
                'background-color': '#f0f2f6'
            }, subset=['Score', 'Total Cost', 'Budget %', 'Max Loss'])  # Make important columns stand out
            .background_gradient(cmap='RdYlGn', subset=['Score'])  # Add color gradient based on score
            .background_gradient(
                cmap='RdYlGn_r',
                subset=['Budget %'],
                vmin=0,
                vmax=10
            ).apply(
                lambda x: ['background-color: #ffcdd2' if v > 5 else '' for v in x],
                subset=['Budget %']
            )
        )
        
        # Add explanatory notes
        st.info(" **Notes:**\n"
                "- Total Cost is calculated based on standard contract size (100 shares)\n"
                "- Budget % shows what percentage of your trading budget this position would use\n"
                "- Max Loss shows the maximum potential loss for the position\n"
                "- Options highlighted in red exceed 20% of your budget")
        
    except Exception as e:
        logger.error(f"Error creating options analysis table: {str(e)}", exc_info=True)
        st.error("An error occurred while creating the options analysis table")

def plot_option_profit_curves(options_df, current_price):
    """Create interactive profit/loss visualization for top options"""
    logger.debug(f"Plotting profit curves for {len(options_df) if not options_df.empty else 0} options")
    
    try:
        if options_df.empty:
            logger.warning("No options data to plot")
            return go.Figure()
        
        fig = go.Figure()
        
        # Calculate price range for x-axis
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        
        for _, option in options_df.iterrows():
            try:
                # Validate required fields
                if not all(field in option for field in ['option_type', 'strike', 'ask', 'score']):
                    logger.warning(f"Missing required fields in option: {option}")
                    continue
                    
                # Calculate profit curve points
                if option['option_type'].lower() == 'call':
                    profits = np.maximum(price_range - float(option['strike']), 0) - float(option['ask'])
                else:  # put
                    profits = np.maximum(float(option['strike']) - price_range, 0) - float(option['ask'])
                
                # Add trace for this option
                fig.add_trace(go.Scatter(
                    x=price_range,
                    y=profits,
                    name=f"{option['option_type']} {option['strike']} (Score: {float(option['score']):.1f})",
                    hovertemplate="Price: $%{x:.2f}<br>P/L: $%{y:.2f}<extra></extra>"
                ))
            except Exception as e:
                logger.error(f"Error plotting option {option}: {str(e)}")
                continue
        
        # Update layout
        fig.update_layout(
            title="Profit/Loss Analysis",
            xaxis_title="Stock Price ($)",
            yaxis_title="Profit/Loss ($)",
            hovermode='x unified',
            showlegend=True
        )
        
        # Add vertical line for current price
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Current Price: ${current_price:.2f}"
        )
        
        logger.info("Successfully created profit/loss visualization")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error creating profit/loss plot: {str(e)}", exc_info=True)
        return go.Figure()

def plot_technical_analysis(historical_data, indicators):
    """Create interactive technical analysis visualization with Bloomberg Terminal theme"""
    logger.debug("Creating technical analysis plot")
    
    try:
        if historical_data.empty or not indicators:
            logger.warning("Missing data for technical analysis plot")
            return go.Figure()
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Price and Bollinger Bands', 'Momentum Indicators'))
        
        # Plot candlestick chart
        fig.add_trace(go.Candlestick(
            x=historical_data.index,
            open=historical_data['open'],
            high=historical_data['high'],
            low=historical_data['low'],
            close=historical_data['close'],
            name="Price",
            increasing_line_color=BLOOMBERG_COLORS['positive_green'],
            decreasing_line_color=BLOOMBERG_COLORS['negative_red']
        ), row=1, col=1)
        
        # Add Bollinger Bands if available
        if 'BB' in indicators:
            bb_data = indicators['BB']
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=bb_data['upper_data'],
                name="BB Upper",
                line=dict(dash='dash', color=BLOOMBERG_COLORS['terminal_amber']),
                opacity=0.7
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=bb_data['lower_data'],
                name="BB Lower",
                line=dict(dash='dash', color=BLOOMBERG_COLORS['terminal_amber']),
                opacity=0.7,
                fill='tonexty',  # Fill area between upper and lower bands
                fillcolor='rgba(255, 165, 0, 0.1)'  # Slightly visible orange
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=bb_data['middle_data'],
                name="BB Middle",
                line=dict(color=BLOOMBERG_COLORS['terminal_amber'], dash='dot'),
                opacity=0.7
            ), row=1, col=1)
        
        # Add RSI if available
        if 'RSI' in indicators:
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=indicators['RSI']['data'],
                name="RSI",
                line=dict(color=BLOOMBERG_COLORS['terminal_green'])
            ), row=2, col=1)
            
            # Add RSI threshold lines
            fig.add_hline(y=70, line_dash="dash", line_color=BLOOMBERG_COLORS['alert_red'], row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=BLOOMBERG_COLORS['positive_green'], row=2, col=1)
        
        # Add MACD if available
        if 'MACD' in indicators:
            macd_data = indicators['MACD']
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=macd_data['data'],
                name="MACD",
                line=dict(color=BLOOMBERG_COLORS['terminal_amber'])
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=macd_data['signal_data'],
                name="Signal",
                line=dict(color=BLOOMBERG_COLORS['terminal_green'])
            ), row=2, col=1)
            
            # Add MACD histogram with color based on value
            histogram_colors = [
                BLOOMBERG_COLORS['positive_green'] if val >= 0 else BLOOMBERG_COLORS['negative_red']
                for val in macd_data['histogram_data']
            ]
            
            fig.add_trace(go.Bar(
                x=historical_data.index,
                y=macd_data['histogram_data'],
                name="MACD Histogram",
                marker_color=histogram_colors,
                opacity=0.5
            ), row=2, col=1)
        
        # Update layout with Bloomberg theme
        fig.update_layout(
            title={
                'text': "Technical Analysis",
                'font': {'color': BLOOMBERG_COLORS['terminal_amber']}
            },
            height=800,  # Increased height for better visibility
            showlegend=True,
            legend={
                'font': {'color': BLOOMBERG_COLORS['terminal_green']},
                'bgcolor': 'rgba(0,0,0,0)',
                'bordercolor': BLOOMBERG_COLORS['grid_gray']
            },
            xaxis_rangeslider_visible=False  # Hide rangeslider for cleaner look
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=BLOOMBERG_COLORS['grid_gray'],
            title_font={'color': BLOOMBERG_COLORS['terminal_amber']},
            tickfont={'color': BLOOMBERG_COLORS['terminal_green']}
        )
        
        fig.update_yaxes(
            gridcolor=BLOOMBERG_COLORS['grid_gray'],
            title_font={'color': BLOOMBERG_COLORS['terminal_amber']},
            tickfont={'color': BLOOMBERG_COLORS['terminal_green']}
        )
        
        logger.info("Successfully created technical analysis visualization")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating technical analysis plot: {str(e)}", exc_info=True)
        return go.Figure()

def plot_pmcc_profit_curve(strategy, current_price):
    """
    Plot PMCC strategy profit/loss curve
    """
    try:
        leap, short = strategy
        
        # Calculate price range (20% below current price to 20% above short strike)
        price_range_min = current_price * 0.8
        price_range_max = max(float(short['strike']) * 1.2, current_price * 1.2)
        price_range = np.linspace(price_range_min, price_range_max, 100)
        
        # Calculate profits at each price point
        leap_profits = np.maximum(price_range - float(leap['strike']), 0) - float(leap['ask'])
        short_profits = float(short['bid']) - np.maximum(price_range - float(short['strike']), 0)
        total_profits = (leap_profits + short_profits) * 100  # Convert to dollar amounts
        
        # Create figure
        fig = go.Figure()
        
        # Add profit curves
        fig.add_trace(go.Scatter(
            x=price_range,
            y=leap_profits * 100,
            name='LEAPS Call',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=price_range,
            y=short_profits * 100,
            name='Short Call',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=price_range,
            y=total_profits,
            name='Total P/L',
            line=dict(color='blue', width=3)
        ))
        
        # Add key price points
        fig.add_vline(x=current_price, line_dash="dash", line_color="gray", annotation_text="Current Price")
        fig.add_vline(x=float(leap['strike']), line_dash="dash", line_color="green", annotation_text="LEAPS Strike")
        fig.add_vline(x=float(short['strike']), line_dash="dash", line_color="red", annotation_text="Short Strike")
        
        # Add break-even line
        net_debit = (float(leap['ask']) - float(short['bid'])) * 100
        breakeven = float(leap['strike']) + net_debit/100
        fig.add_vline(x=breakeven, line_dash="dot", line_color="gray", annotation_text="Break Even")
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="solid", line_color="gray")
        
        # Update layout
        fig.update_layout(
            title='PMCC Strategy Profit/Loss Profile',
            xaxis_title='Stock Price',
            yaxis_title='Profit/Loss ($)',
            hovermode='x unified',
            showlegend=True,
            template='plotly_dark',
            height=500,
            margin=dict(t=30)
        )
        
        # Add hover templates
        fig.update_traces(
            hovertemplate='Stock Price: $%{x:.2f}<br>P/L: $%{y:.2f}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error plotting PMCC profit curve: {str(e)}", exc_info=True)
        st.error("An error occurred while plotting the PMCC profit curve")

def get_risk_level_description(risk_level):
    """Get detailed description for each risk level"""
    descriptions = {
        "Conservative": """
        Conservative Risk Profile (1-3):
        - Focuses on capital preservation
        - Prefers options with lower premiums
        - Favors higher open interest and volume
        - Avoids high IV options
        - Strict spread requirements (<3%)
        - Prefers near-the-money options
        - Maximum position size: 10% of budget
        """,
        
        "Moderate": """
        Moderate Risk Profile (4-7):
        - Balanced risk-reward approach
        - Accepts moderate premiums
        - Standard volume/OI requirements
        - Tolerates moderate IV levels
        - Standard spread requirements (<5%)
        - Considers both ITM and OTM options
        - Maximum position size: 15% of budget
        """,
        
        "Aggressive": """
        Aggressive Risk Profile (8-10):
        - Focuses on higher returns
        - Accepts higher premiums
        - More flexible with volume/OI
        - Tolerates higher IV levels
        - Flexible spread requirements (<7%)
        - More emphasis on OTM options
        - Maximum position size: 20% of budget
        """
    }
    return descriptions.get(risk_level, "Invalid risk level")

def risk_level_to_score(risk_level):
    """Convert risk level to numeric score"""
    risk_scores = {
        "Conservative": 2,  # Middle of 1-3 range
        "Moderate": 5,     # Middle of 4-7 range
        "Aggressive": 9    # Middle of 8-10 range
    }
    return risk_scores.get(risk_level, 5)  # Default to moderate if invalid

def display_options_table(options_df):
    """Display a formatted table of options data with enhanced information"""
    try:
        if options_df.empty:
            return
            
        # Create a copy to avoid modifying original
        display_df = options_df.copy()
        
        # Format monetary values
        for col in ['strike', 'bid', 'ask']:
            display_df[col] = display_df[col].apply(lambda x: float(x))
            
        # Calculate additional metrics
        display_df['Premium'] = (display_df['bid'] + display_df['ask']) / 2
        display_df['Spread %'] = ((display_df['ask'] - display_df['bid']) / display_df['bid'] * 100).round(2)
        display_df['Total Cost'] = display_df['ask'] * 100  # Standard contract size
        
        # Extract Greeks
        if 'greeks' in display_df.columns:
            for greek in ['delta', 'gamma', 'theta', 'vega', 'mid_iv']:
                display_df[greek] = display_df['greeks'].apply(
                    lambda x: float(x.get(greek, 0)) if isinstance(x, dict) else 0
                )
        
        # Format expiration date
        display_df['Expiry'] = pd.to_datetime(display_df['expiration_date']).dt.strftime('%Y-%m-%d')
        
        # Define column order and names (priority columns first)
        priority_columns = {
            'strike': 'Strike',
            'Expiry': 'Expiry',
            'score': 'Score',
            'bid': 'Bid'
        }
        
        other_columns = {
            'ask': 'Ask',
            'Premium': 'Premium',
            'Spread %': 'Spread%',
            'Total Cost': 'Cost',
            'volume': 'Volume',
            'open_interest': 'OI',
            'delta': 'Delta',
            'gamma': 'Gamma',
            'theta': 'Theta',
            'vega': 'Vega',
            'mid_iv': 'IV'
        }
        
        # Combine all columns in desired order
        columns = {**priority_columns, **other_columns}
        
        # Select only available columns
        available_cols = [col for col in columns.keys() if col in display_df.columns]
        display_df = display_df[available_cols].copy()
        display_df.rename(columns=columns, inplace=True)
        
        # Create style function
        def style_dataframe(df):
            return df.style.format({
                'Strike': '${:,.2f}',
                'Score': '{:.2f}',
                'Bid': '${:,.2f}',
                'Ask': '${:,.2f}',
                'Premium': '${:,.2f}',
                'Spread%': '{:.1f}%',
                'Cost': '${:,.2f}',
                'Volume': '{:,.0f}',
                'OI': '{:,.0f}',
                'Delta': '{:.3f}',
                'Gamma': '{:.3f}',
                'Theta': '{:.3f}',
                'Vega': '{:.3f}',
                'IV': '{:.1%}'
            }).background_gradient(
                cmap='RdYlGn',
                subset=['Score']
            ).background_gradient(
                cmap='RdYlGn_r',
                subset=['Spread%'],
                vmin=0,
                vmax=10
            ).apply(
                lambda x: ['background-color: #ffcdd2' if v > 5 else '' for v in x],
                subset=['Spread%']
            )
        
        # Display tables with headers
        st.markdown("#### Detailed Options Analysis")
        
        # Add explanatory text
        st.markdown("""
        - **Score**: Higher is better (0-100)
        - **Spread%**: Lower is better (<5% preferred)
        - **Volume/OI**: Higher numbers indicate better liquidity
        - **Greeks**: Delta (direction), Gamma (speed), Theta (time decay), Vega (volatility)
        """)
        
        # Sort by score
        display_df = display_df.sort_values('Score', ascending=False)
        
        # Display the styled table
        st.dataframe(
            style_dataframe(display_df),
            height=400  # Set a fixed height for better visibility
        )
        
        # Add warning for high spread options
        high_spread = display_df[display_df['Spread%'] > 5]
        if not high_spread.empty:
            st.warning(" Some options have high bid-ask spreads (>5%). Consider the impact on your trading costs.")
            
    except Exception as e:
        logger.error(f"Error displaying options table: {str(e)}", exc_info=True)
        st.error("Error displaying options table")

def plot_profit_loss(option, current_price):
    """Create interactive profit/loss visualization with Bloomberg Terminal theme"""
    try:
        # Calculate price range
        strike = float(option['strike'])
        premium = float(option['ask'])  # Use ask price for conservative estimate
        option_type = option['option_type']
        
        # Create price range (20% below and above current price)
        price_range = np.linspace(current_price * 0.8, current_price * 1.2, 100)
        
        # Calculate profit/loss for each price point
        if option_type == 'call':
            profit_loss = np.maximum(price_range - strike, 0) - premium
        else:  # put
            profit_loss = np.maximum(strike - price_range, 0) - premium
            
        # Create profit/loss trace
        profit_color = BLOOMBERG_COLORS['positive_green']
        loss_color = BLOOMBERG_COLORS['negative_red']
        
        fig = go.Figure()
        
        # Add profit/loss line with color gradient
        for i in range(len(price_range) - 1):
            color = profit_color if profit_loss[i] >= 0 else loss_color
            fig.add_trace(go.Scatter(
                x=price_range[i:i+2],
                y=profit_loss[i:i+2] * 100,  # Convert to per contract (100 shares)
                mode='lines',
                line=dict(color=color),
                showlegend=False,
                hovertemplate="Price: $%{x:.2f}<br>P/L: $%{y:.2f}<extra></extra>"
            ))
            
        # Add break-even line
        fig.add_hline(y=0, line_dash="dash", line_color=BLOOMBERG_COLORS['terminal_amber'])
        
        # Add current price marker
        fig.add_vline(x=current_price, line_dash="dash", line_color=BLOOMBERG_COLORS['terminal_amber'])
        
        # Add strike price marker
        fig.add_vline(x=strike, line_dash="dot", line_color=BLOOMBERG_COLORS['terminal_green'])
        
        # Calculate max profit/loss for y-axis range
        max_profit = max(profit_loss) * 100
        max_loss = min(profit_loss) * 100
        y_range = [max_loss * 1.1, max_profit * 1.1]  # Add 10% padding
        
        # Update layout with Bloomberg theme
        fig.update_layout(
            title={
                'text': f"{option_type.upper()} Option P/L Analysis - Strike: ${strike}",
                'font': {'color': BLOOMBERG_COLORS['terminal_amber']}
            },
            xaxis_title="Stock Price",
            yaxis_title="Profit/Loss ($)",
            showlegend=False,
            hovermode='x unified',
            height=500,
            xaxis=dict(
                gridcolor=BLOOMBERG_COLORS['grid_gray'],
                title_font={'color': BLOOMBERG_COLORS['terminal_amber']},
                tickfont={'color': BLOOMBERG_COLORS['terminal_green']}
            ),
            yaxis=dict(
                gridcolor=BLOOMBERG_COLORS['grid_gray'],
                title_font={'color': BLOOMBERG_COLORS['terminal_amber']},
                tickfont={'color': BLOOMBERG_COLORS['terminal_green']},
                tickformat='$,.0f',
                range=y_range
            )
        )
        
        # Add annotations for key points
        annotation_y = y_range[1]  # Use calculated y-range for annotations
        
        fig.add_annotation(
            x=current_price,
            y=annotation_y,
            text="Current Price",
            showarrow=True,
            arrowhead=2,
            arrowcolor=BLOOMBERG_COLORS['terminal_amber'],
            font={'color': BLOOMBERG_COLORS['terminal_amber']},
            yshift=10
        )
        
        fig.add_annotation(
            x=strike,
            y=annotation_y,
            text="Strike Price",
            showarrow=True,
            arrowhead=2,
            arrowcolor=BLOOMBERG_COLORS['terminal_green'],
            font={'color': BLOOMBERG_COLORS['terminal_green']},
            yshift=-10
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Break Even",
                f"${strike + premium if option_type == 'call' else strike - premium:.2f}",
                delta=f"{((strike + premium if option_type == 'call' else strike - premium) - current_price) / current_price * 100:.1f}%",
                delta_color="normal"
            )
            
        with col2:
            max_loss_display = abs(premium * 100)  # Per contract
            st.metric(
                "Max Loss",
                f"${max_loss_display:.2f}",
                delta=f"{max_loss_display / float(option['ask']) * 100:.1f}% of investment",
                delta_color="inverse"
            )
            
        with col3:
            if option_type == 'call':
                profit_at_target = max(0, current_price * 1.2 - strike - premium) * 100
                st.metric(
                    "Profit at +20%",
                    f"${profit_at_target:.2f}",
                    delta=f"{max(0, profit_at_target / (premium * 100) * 100):.1f}% return",
                    delta_color="normal"
                )
            else:
                profit_at_target = max(0, strike - current_price * 0.8 - premium) * 100
                st.metric(
                    "Profit at -20%",
                    f"${profit_at_target:.2f}",
                    delta=f"{max(0, profit_at_target / (premium * 100) * 100):.1f}% return",
                    delta_color="normal"
                )
        
    except Exception as e:
        logger.error(f"Error plotting profit/loss: {str(e)}", exc_info=True)
        st.error("Error creating profit/loss visualization")

def calculate_trends(historical_data, indicators):
    """Calculate short and long term trends based on multiple indicators"""
    try:
        trends = {
            'short_term': {'signal': 'NEUTRAL', 'strength': 0, 'reasons': []},
            'long_term': {'signal': 'NEUTRAL', 'strength': 0, 'reasons': []}
        }
        
        if historical_data.empty or not indicators:
            return trends
            
        # Get latest price and SMA values
        current_price = historical_data['close'].iloc[-1]
        sma_20 = historical_data['close'].rolling(window=20).mean().iloc[-1]
        sma_50 = historical_data['close'].rolling(window=50).mean().iloc[-1]
        
        # Short term trend analysis (1-5 days)
        if 'RSI' in indicators:
            rsi = indicators['RSI']['value']
            if rsi > 70:
                trends['short_term']['strength'] -= 1
                trends['short_term']['reasons'].append(f"RSI overbought ({rsi:.1f})")
            elif rsi < 30:
                trends['short_term']['strength'] += 1
                trends['short_term']['reasons'].append(f"RSI oversold ({rsi:.1f})")
                
        if 'MACD' in indicators:
            macd = indicators['MACD']['value']
            signal = indicators['MACD']['signal']
            if macd > signal:
                trends['short_term']['strength'] += 1
                trends['short_term']['reasons'].append("MACD above signal line")
            else:
                trends['short_term']['strength'] -= 1
                trends['short_term']['reasons'].append("MACD below signal line")
                
        if 'BB' in indicators:
            bb = indicators['BB']
            if current_price > bb['upper']:
                trends['short_term']['strength'] -= 1
                trends['short_term']['reasons'].append("Price above upper BB")
            elif current_price < bb['lower']:
                trends['short_term']['strength'] += 1
                trends['short_term']['reasons'].append("Price below lower BB")
                
        # Price vs short-term SMA
        if current_price > sma_20:
            trends['short_term']['strength'] += 1
            trends['short_term']['reasons'].append("Price above 20-day SMA")
        else:
            trends['short_term']['strength'] -= 1
            trends['short_term']['reasons'].append("Price below 20-day SMA")
            
        # Long term trend analysis (20+ days)
        # Price vs long-term SMA
        if current_price > sma_50:
            trends['long_term']['strength'] += 1
            trends['long_term']['reasons'].append("Price above 50-day SMA")
        else:
            trends['long_term']['strength'] -= 1
            trends['long_term']['reasons'].append("Price below 50-day SMA")
            
        # Trend strength based on consecutive closes
        last_5_closes = historical_data['close'].tail(5)
        last_20_closes = historical_data['close'].tail(20)
        
        if last_5_closes.is_monotonic_increasing:
            trends['short_term']['strength'] += 1
            trends['short_term']['reasons'].append("5-day consecutive higher closes")
        elif last_5_closes.is_monotonic_decreasing:
            trends['short_term']['strength'] -= 1
            trends['short_term']['reasons'].append("5-day consecutive lower closes")
            
        if last_20_closes.is_monotonic_increasing:
            trends['long_term']['strength'] += 2
            trends['long_term']['reasons'].append("20-day upward trend")
        elif last_20_closes.is_monotonic_decreasing:
            trends['long_term']['strength'] -= 2
            trends['long_term']['reasons'].append("20-day downward trend")
            
        # Determine final signals based on strength
        for timeframe in ['short_term', 'long_term']:
            if trends[timeframe]['strength'] >= 2:
                trends[timeframe]['signal'] = 'STRONG BULLISH'
            elif trends[timeframe]['strength'] > 0:
                trends[timeframe]['signal'] = 'BULLISH'
            elif trends[timeframe]['strength'] <= -2:
                trends[timeframe]['signal'] = 'STRONG BEARISH'
            elif trends[timeframe]['strength'] < 0:
                trends[timeframe]['signal'] = 'BEARISH'
            else:
                trends[timeframe]['signal'] = 'NEUTRAL'
                
        return trends
        
    except Exception as e:
        logger.error(f"Error calculating trends: {str(e)}", exc_info=True)
        return trends

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def main():
    st.title("Ronk Trading Solutions")
    
    # Initialize session state
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now() - timedelta(minutes=5)
        
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
        
    # Add test button at the top of the app
    if st.sidebar.button("Test API Connection"):
        test_api_connection()
    
    # Sidebar inputs
    with st.sidebar:
        ticker = st.text_input("Enter Stock Ticker", value="SPY").upper()
        
        # Main navigation
        selected_tab = st.radio(
            "Select Strategy",
            ["Options Analysis", "PMCC Strategy"],
            index=0,
            help="Choose between regular options analysis or PMCC strategy"
        )
        
        # Risk appetite selection
        risk_appetite = st.selectbox(
            "Risk Appetite",
            options=["Conservative", "Moderate", "Aggressive"],
            index=1,
            help="Select your risk tolerance level"
        )
        
        # Display risk level description
        st.info(get_risk_level_description(risk_appetite))
        
        if selected_tab == "Options Analysis":
            # Option Type Selection
            option_type = st.selectbox(
                "Option Type",
                options=["Call", "Put", "Both"],
                index=0,
                help="Select the type of options to analyze"
            )
            
            # Date range selection dropdown for regular options analysis
            date_ranges = get_date_ranges()
            selected_range = st.selectbox(
                "Select Date Range",
                options=list(date_ranges.keys())[:4],  # Only show up to 6 months for regular analysis
                index=1,  # Default to 1 Month
                help="Select the expiration date range for options analysis"
            )
        else:  # PMCC Strategy
            option_type = "Call"  # PMCC only uses calls
            
            # Date range selection dropdown for PMCC
            date_ranges = get_date_ranges()
            selected_range = st.selectbox(
                "Select Date Range",
                options=list(date_ranges.keys())[3:],  # Only show 6 months and longer for PMCC
                index=1,  # Default to 1 Year
                help="Select the expiration date range for PMCC analysis (longer dates needed for LEAPS)"
            )
        
        start_date, end_date = date_ranges[selected_range]
        
        # Display selected date range
        st.info(f"Analyzing options from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        budget = st.number_input(
            "Trading Budget ($)", 
            value=200.0,
            step=50.0,
            format="%.2f",
            help="Enter your trading budget. Note: The system will warn you if positions exceed 20% of your budget."
        )
        
        if budget <= 0:
            st.warning("Please enter a positive budget amount")
            return
            
        # Convert risk level to score for calculations
        risk_appetite_score = risk_level_to_score(risk_appetite)
        
    # Main content area
    try:
        # Fetch current price and historical data
        current_price = fetch_current_price(ticker)
        if current_price is None:
            st.error(f"Could not fetch current price for {ticker}")
            return
            
        col1, col2 = st.columns([3, 1])
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
        
        # Fetch and display historical data
        historical_data = fetch_historical_data(ticker)
        if historical_data.empty:
            st.error(f"Could not fetch historical data for {ticker}")
            return
            
        # Calculate technical indicators
        indicators = calculate_technical_indicators(historical_data)
        
        # Calculate and display sentiment
        sentiment_score = calculate_sentiment(ticker)
        if sentiment_score is not None:
            st.metric("Market Sentiment", 
                     f"{sentiment_score:.1f}/100",
                     "Bullish" if sentiment_score > 50 else "Bearish")
        
        # Fetch and analyze options for the selected date range
        options_chain = fetch_options_for_range(
            ticker, 
            start_date,
            end_date,
            min_oi=100,
            min_volume=50,
            max_iv=90,
            option_type=option_type
        )
        
        if options_chain.empty:
            st.error(f"No options found for {ticker} in the selected date range")
            return
            
        # Score and display options
        scored_options = score_options(options_chain, indicators, current_price, budget, risk_appetite_score, sentiment_score, historical_data)
        display_options_analysis(scored_options, budget, historical_data, indicators)
        
        # PMCC analysis if enabled
        if selected_tab == "PMCC Strategy":
            pmcc_recommendations = analyze_pmcc_strategy(options_chain, current_price, budget, indicators)
            if pmcc_recommendations:
                display_pmcc_analysis(pmcc_recommendations)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        st.error("An error occurred while analyzing options. Please check the logs for details.")

def display_technical_analysis(indicators):
    """
    Display technical analysis indicators in an organized format
    
    Args:
        indicators (dict): Dictionary containing technical indicators
    """
    try:
        if not indicators:
            st.warning("No technical indicators available")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Moving Averages")
            sma_20 = indicators.get('SMA_20', {}).get('value')
            sma_50 = indicators.get('SMA_50', {}).get('value')
            if sma_20 and sma_50:
                st.metric("SMA 20", f"${sma_20:.2f}")
                st.metric("SMA 50", f"${sma_50:.2f}")
                trend = "Bullish" if sma_20 > sma_50 else "Bearish"
                st.write(f"Trend: {trend}")
            
        with col2:
            st.write("### Momentum Indicators")
            rsi = indicators.get('RSI', {}).get('value')
            macd = indicators.get('MACD', {}).get('value')
            
            if rsi:
                st.metric("RSI", f"{rsi:.1f}", 
                        delta="Overbought" if rsi > 70 else "Oversold" if rsi < 30 else None)
                
            if macd is not None:
                st.metric("MACD", f"{macd:.3f}")
                
        # Display Bollinger Bands
        bb_upper = indicators.get('BB_upper', {}).get('value')
        bb_middle = indicators.get('BB_middle', {}).get('value')
        bb_lower = indicators.get('BB_lower', {}).get('value')
        
        if all(x is not None for x in [bb_upper, bb_middle, bb_lower]):
            st.write("### Bollinger Bands")
            st.metric("Upper Band", f"${bb_upper:.2f}")
            st.metric("Middle Band", f"${bb_middle:.2f}")
            st.metric("Lower Band", f"${bb_lower:.2f}")
            
    except Exception as e:
        logger.error(f"Error displaying technical analysis: {str(e)}", exc_info=True)
        st.error("Could not display technical analysis")

def fetch_news(ticker):
    """
    Fetch news articles for a ticker using Alpha Vantage News API
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        list: List of news headlines and summaries
    """
    try:
        # Get API key
        api_key = ALPHA_VANTAGE_API_KEY
        if not api_key:
            logger.warning("Alpha Vantage API key not found in secrets!")
            return []
            
        # Construct URL and parameters
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": api_key,
            "limit": 10,
            "sort": "LATEST"  # Get latest news first
        }
        
        # Make request with exponential backoff
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching news for {ticker}, attempt {attempt + 1}")
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                logger.debug(f"API Response keys: {data.keys() if isinstance(data, dict) else 'Invalid response'}")
                
                # Check for API errors
                if "Error Message" in data:
                    logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                    return []
                    
                if 'Note' in data:  # API call frequency limit reached
                    logger.warning(f"Alpha Vantage API limit note: {data['Note']}")
                    time.sleep(60)  # Wait for a minute before retrying
                    continue
                    
                if 'feed' not in data:
                    logger.warning(f"No news feed found in response for {ticker}")
                    return []
                    
                news_items = []
                for item in data["feed"]:
                    title = item.get("title", "")
                    summary = item.get("summary", "")
                    sentiment = item.get("overall_sentiment_score", 0)
                    time_published = item.get("time_published", "")
                    
                    news_items.append({
                        "title": title,
                        "summary": summary,
                        "sentiment": float(sentiment),
                        "time_published": time_published
                    })
                    
                logger.info(f"Successfully fetched {len(news_items)} news items for {ticker}")
                return news_items
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch news after {max_retries} attempts: {str(e)}")
                    return []
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Retrying news fetch in {wait_time} seconds...")
                time.sleep(wait_time)
                
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {str(e)}")
        return []

def calculate_sentiment(ticker):
    """
    Calculate sentiment score for a ticker using Alpha Vantage News API
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        float: Sentiment score from 0 to 100, or 50 if error/no data
    """
    try:
        news_items = fetch_news(ticker)
        if not news_items:
            logger.warning(f"No news found for {ticker}, using neutral sentiment")
            return 50.0
            
        # Calculate weighted sentiment score with time decay
        total_weight = 0
        weighted_sentiment = 0
        
        # Get current time for time decay calculation
        current_time = datetime.now()
        
        for item in news_items:
            try:
                # Parse time published
                time_str = item.get("time_published", "")
                if time_str:
                    time_published = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
                    # Calculate time weight (more recent = higher weight)
                    time_diff = (current_time - time_published).total_seconds() / 3600  # hours
                    time_weight = 1.0 / (1.0 + time_diff/24)  # Decay over days
                else:
                    time_weight = 0.5  # Default weight if no time available
                    
                sentiment = item["sentiment"]
                weighted_sentiment += sentiment * time_weight
                total_weight += time_weight
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Error processing news item: {str(e)}")
                continue
                
        if total_weight > 0:
            avg_sentiment = weighted_sentiment / total_weight
            # Convert to 0-100 scale (Alpha Vantage sentiment is typically -1 to 1)
            normalized_sentiment = (avg_sentiment + 1) * 50
            normalized_sentiment = max(0, min(100, normalized_sentiment))  # Ensure within bounds
            
            logger.info(f"Calculated sentiment for {ticker}: {normalized_sentiment:.2f}")
            return normalized_sentiment
        else:
            logger.warning(f"No valid sentiment data for {ticker}, using neutral sentiment")
            return 50.0
            
    except Exception as e:
        logger.error(f"Error calculating sentiment: {str(e)}")
        return 50.0  # Return neutral sentiment on error

if __name__ == "__main__":
    main()
