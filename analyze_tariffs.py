import pandas as pd
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_comment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process multiple comment columns into a single series.
    
    Args:
        df: DataFrame with comment_1 through comment_10 columns
        
    Returns:
        DataFrame with melted comments and their dates
    """
    # Find all comment columns
    comment_cols = [col for col in df.columns if col.startswith('comment_')]
    
    # Melt the comment columns into a single series
    melted_df = pd.melt(
        df,
        id_vars=['date'],
        value_vars=comment_cols,
        var_name='comment_number',
        value_name='comment'
    )
    
    # Remove rows with empty comments
    melted_df = melted_df.dropna(subset=['comment'])
    melted_df['date'] = pd.to_datetime(melted_df['date'])

    return melted_df

def analyze_sentiment(comments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform sentiment analysis on Reddit comments from CSV.
    
    Args:
        comments_df: DataFrame containing Reddit comments
    
    Returns:
        DataFrame with daily aggregated sentiment scores
    """
    try:
        # Process comment columns if not already processed
        if 'comment' not in comments_df.columns:
            comments_df = process_comment_columns(comments_df)
        
        # Perform sentiment analysis
        comments_df['sentiment'] = comments_df['comment'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        
        # Aggregate by date
        daily_sentiment = comments_df.groupby('date')['sentiment'].agg([
                ('mean_sentiment', 'mean'),
                ('count', 'count'),
                ('std', 'std')
            ]).reset_index()
        
        # Filter out days with too few comments for statistical significance
        daily_sentiment = daily_sentiment[daily_sentiment['count'] >= 5]
        
        return daily_sentiment
    except Exception as e:
        logger.error(f"Error performing sentiment analysis: {e}")
        return pd.DataFrame()

def fetch_exchange_rate(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch USD/CAD exchange rate data from Yahoo Finance.
    
    Args:
        start_date: Start date for data collection
        end_date: End date for data collection
    
    Returns:
        DataFrame with columns ['date', 'exchange_rate']
        Returns empty DataFrame with same structure if error occurs
    """
    # Define the empty DataFrame structure for error cases
    empty_df = pd.DataFrame(columns=['date', 'exchange_rate'])
    
    try:
        logger.info(f"Fetching USD/CAD exchange rate from {start_date} to {end_date}")
        usdcad = yf.download('USDCAD=X', start=start_date, end=end_date)
        
        if usdcad.empty:
            logger.error("No data retrieved from Yahoo Finance")
            return empty_df
            
        # Log the original DataFrame structure
        logger.debug(f"Raw Yahoo Finance columns: {usdcad.columns.tolist()}")
        logger.debug(f"Raw data sample:\n{usdcad.head()}")
        
        # Verify 'Close' column exists
        if 'Close' not in usdcad.columns:
            logger.error("Close price column not found in Yahoo Finance data")
            return empty_df
        
        # Create DataFrame with only needed columns
        df = usdcad[['Close']].copy()
        df = df.reset_index()  # Convert Date index to column
        
        # Rename columns
        df.columns = ['date', 'exchange_rate']
        
        # Clean the data
        df = df.dropna()  # Remove any rows with NaN values
        
        # Ensure proper data types
        df['date'] = pd.to_datetime(df['date'])
        df['exchange_rate'] = df['exchange_rate'].astype(float)
        
        logger.debug(f"Processed DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"Processed data sample:\n{df.head()}")
        logger.info(f"Successfully retrieved {len(df)} days of exchange rate data")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching exchange rate data: {str(e)}")
        logger.debug("Full error details", exc_info=True)
        return empty_df

def analyze_correlation(sentiment_df: pd.DataFrame, 
                    exchange_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Analyze correlation between sentiment and exchange rate data.
    
    Args:
        sentiment_df: DataFrame with sentiment data
        exchange_df: DataFrame with exchange rate data
    
    Returns:
        Tuple of correlation coefficient and p-value
    """
    try:
        # Log initial dataframe information
        logger.debug(f"Sentiment DataFrame columns: {sentiment_df.columns.tolist()}")
        logger.debug(f"Exchange DataFrame columns: {exchange_df.columns.tolist()}")
        logger.debug(f"First few rows of sentiment_df:\n{sentiment_df.head()}")
        logger.debug(f"First few rows of exchange_df:\n{exchange_df.head()}")
        
        # Validate required columns
        required_sentiment_cols = ['date', 'mean_sentiment']
        required_exchange_cols = ['date', 'exchange_rate']
        
        missing_sentiment_cols = [col for col in required_sentiment_cols if col not in sentiment_df.columns]
        missing_exchange_cols = [col for col in required_exchange_cols if col not in exchange_df.columns]
        
        if missing_sentiment_cols:
            raise ValueError(f"Missing required columns in sentiment_df: {missing_sentiment_cols}")
        if missing_exchange_cols:
            raise ValueError(f"Missing required columns in exchange_df: {missing_exchange_cols}")
        
        # Ensure dates are in datetime64[ns] format
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        exchange_df['date'] = pd.to_datetime(exchange_df['date'])
        
        # Validate data is not empty
        if len(sentiment_df) == 0 or len(exchange_df) == 0:
            raise ValueError("One or both dataframes are empty")
        
        # Merge datasets on date
        merged_df = pd.merge(sentiment_df, exchange_df, on='date', how='inner')
        
        if len(merged_df) == 0:
            raise ValueError("No matching dates between sentiment and exchange rate data")
        
        logger.info(f"Successfully merged dataframes with {len(merged_df)} matching dates")
        
        # Calculate correlation
        correlation_coef, p_value = stats.pearsonr(
            merged_df['mean_sentiment'],
            merged_df['exchange_rate']
        )
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Sentiment vs Exchange Rate scatter plot
        plt.subplot(2, 1, 1)
        plt.scatter(merged_df['mean_sentiment'], merged_df['exchange_rate'], alpha=0.5)
        plt.xlabel('Average Daily Sentiment Score')
        plt.ylabel('USD/CAD Exchange Rate')
        plt.title(f'Sentiment vs Exchange Rate (r={correlation_coef:.3f}, p={p_value:.3f})')
        
        # Time series plot
        plt.subplot(2, 1, 2)
        plt.plot(merged_df['date'], merged_df['mean_sentiment'], label='Sentiment', color='blue')
        plt.twinx().plot(merged_df['date'], merged_df['exchange_rate'], 
                    label='Exchange Rate', color='red', alpha=0.7)
        plt.xlabel('Date')
        plt.title('Sentiment and Exchange Rate Over Time')
        
        plt.tight_layout()
        plt.savefig('sentiment_analysis_results.png')
        
        return correlation_coef, p_value
    except Exception as e:
        logger.error(f"Error analyzing correlation: {e}")
        return 0.0, 1.0

def main():
    """Main execution function."""
    try:
        # Load Reddit comments from CSV
        logger.info("Loading comments from CSV...")
        comments_df = pd.read_csv('subreddit_full_comments.csv')
        
        # Get date range from the comments
        start_date = pd.to_datetime(comments_df['date']).min()
        end_date = pd.to_datetime(comments_df['date']).max()
        
        logger.info(f"Analyzing comments from {start_date} to {end_date}")
        
        # Perform sentiment analysis
        sentiment_df = analyze_sentiment(comments_df)
        logger.info(f"Analyzed sentiment for {len(sentiment_df)} days")
        
        # Fetch exchange rate data
        exchange_df = fetch_exchange_rate(start_date, end_date)
        logger.info(f"Fetched exchange rate data for {len(exchange_df)} days")
        
        # Analyze correlation
        correlation, p_value = analyze_correlation(sentiment_df, exchange_df)
        
        logger.info(f"Analysis complete. Correlation: {correlation:.3f}, p-value: {p_value:.3f}")
        
        # Save results
        sentiment_df.to_csv('sentiment_analysis.csv', index=False)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()

