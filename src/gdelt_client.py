"""GDELT API client with rate limiting and sentiment analysis."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import aiohttp
from aiohttp import ClientTimeout
import backoff
from loguru import logger
from asyncio_throttle import Throttler

from .config import config


class GDELTClient:
    """GDELT API client with rate limiting and data processing."""
    
    def __init__(self):
        self.base_url = "https://api.gdeltproject.org/api/v2"
        self.throttler = Throttler(
            rate_limit=config.gdelt_rate_limit_requests_per_minute,
            period=60
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = ClientTimeout(total=30)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a rate-limited request to GDELT API."""
        async with self.throttler:
            if not self.session:
                raise RuntimeError("Session not initialized. Use async context manager.")
            
            logger.debug(f"Making GDELT API request: {url}")
            
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                
                # GDELT returns CSV or JSON depending on the endpoint
                content_type = response.headers.get('content-type', '').lower()
                
                if 'json' in content_type:
                    return await response.json()
                else:
                    # For CSV responses, return raw text
                    text = await response.text()
                    return {"raw_data": text}
    
    async def get_doc_search(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_records: int = 250
    ) -> Dict[str, Any]:
        """
        Search GDELT DOC 2.0 API for articles.
        
        Args:
            query: Search query (e.g., company name)
            start_date: Start date for search
            end_date: End date for search
            max_records: Maximum number of records to return
            
        Returns:
            Dictionary containing search results
        """
        url = f"{self.base_url}/doc/doc"
        
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": max_records,
            "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
            "sort": "datedesc"
        }
        
        try:
            result = await self._make_request(url, params)
            logger.info(f"Retrieved {len(result.get('articles', []))} articles for query: {query}")
            return result
        except Exception as e:
            logger.error(f"Error fetching GDELT doc search for {query}: {e}")
            return {"articles": []}
    

    
    async def get_stock_related_news(
        self,
        ticker: str,
        company_name: str,
        hours_back: int = 2
    ) -> Dict[str, Any]:
        """
        Get news data for a stock ticker using single API call.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            hours_back: How many hours back to search
            
        Returns:
            News data from GDELT doc endpoint (updated every 15 minutes)
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours_back)
        
        # Single optimized query that covers both ticker and company name
        query = f'{ticker} OR "{company_name}"'
        
        try:
            # Single API call to the doc endpoint
            result = await self.get_doc_search(query, start_date, end_date, 100)
            
            articles = result.get("articles", [])
            
            logger.info(f"Retrieved {len(articles)} articles for {ticker}")
            
            return {
                "articles": articles,
                "gkg_data": [],  # Not using GKG endpoint
                "timeline": [],   # Not using timeline endpoint
                "ticker": ticker,
                "company_name": company_name,
                "search_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return {
                "articles": [],
                "gkg_data": [],
                "timeline": [],
                "ticker": ticker,
                "company_name": company_name,
                "search_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }


# Utility functions for working with GDELT data
def extract_sentiment_from_article(article: Dict[str, Any]) -> Optional[float]:
    """Extract sentiment score from GDELT article data."""
    try:
        # GDELT sentiment is typically in the 'socialimage' field or 'tone' field
        if 'tone' in article:
            return float(article['tone'])
        elif 'socialimage' in article and 'sentiment' in article['socialimage']:
            return float(article['socialimage']['sentiment'])
        return None
    except (ValueError, TypeError, KeyError):
        return None


def extract_themes_from_gkg(gkg_entry: Dict[str, Any]) -> List[str]:
    """Extract themes from GDELT GKG entry."""
    try:
        themes = gkg_entry.get('themes', '')
        if isinstance(themes, str):
            return [theme.strip() for theme in themes.split(';') if theme.strip()]
        return []
    except (TypeError, AttributeError):
        return []


def is_relevant_to_stock(article: Dict[str, Any], ticker: str, company_name: str) -> bool:
    """
    Determine if an article is directly relevant to a stock.
    
    This helps filter out false positives where the ticker or company name
    appears in unrelated contexts.
    """
    try:
        title = article.get('title', '').lower()
        url = article.get('url', '').lower()
        
        # Direct mentions in title are highly relevant
        if ticker.lower() in title or company_name.lower() in title:
            return True
        
        # Check if it's from a financial news source
        financial_domains = [
            'bloomberg.com', 'reuters.com', 'wsj.com', 'marketwatch.com',
            'cnbc.com', 'yahoo.com/finance', 'fool.com', 'seekingalpha.com',
            'investopedia.com', 'barrons.com', 'ft.com'
        ]
        
        if any(domain in url for domain in financial_domains):
            return True
        
        # Check for financial keywords in title
        financial_keywords = [
            'stock', 'shares', 'trading', 'earnings', 'revenue', 'profit',
            'market', 'investor', 'wall street', 'nasdaq', 'nyse'
        ]
        
        if any(keyword in title for keyword in financial_keywords):
            return True
        
        return False
        
    except Exception:
        return True  # Default to including the article if we can't determine relevance
