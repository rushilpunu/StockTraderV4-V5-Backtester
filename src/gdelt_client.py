"""GDELT API client with rate limiting and sentiment analysis."""

import asyncio
import time
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import aiohttp
from aiohttp import ClientTimeout
import backoff
from loguru import logger
from asyncio_throttle import Throttler
from urllib.parse import urlparse

# Import config from the small capital trader directory
import sys
import os
small_capital_path = os.path.join(os.path.dirname(__file__), '..', 'small_capital_trader')
sys.path.insert(0, small_capital_path)
from config import SmallCapitalTradingConfig
config = SmallCapitalTradingConfig()


class GDELTClient:
    """GDELT API client with rate limiting and data processing."""
    
    def __init__(self):
        self.base_url = "https://api.gdeltproject.org/api/v2"
        self.fallback_url = "https://api.gdeltproject.org/api/v1"  # Fallback to v1 if v2 fails
        self.throttler = Throttler(
            rate_limit=30,  # Reduced from 60 to 30 requests per minute to be more conservative
            period=60
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = ClientTimeout(total=120)  # Increased timeout to 2 minutes for GDELT
    
    async def __aenter__(self):
        """Async context manager entry."""
        # Set a User-Agent to avoid being blocked by some endpoints
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            headers={
                "User-Agent": "StockTraderV2/1.0 (+https://localhost)",
                "Accept": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError, aiohttp.ServerTimeoutError),
        max_tries=5,  # Increased retries
        max_time=300  # Increased max time to 5 minutes
    )
    async def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a rate-limited request to GDELT API."""
        async with self.throttler:
            if not self.session:
                raise RuntimeError("Session not initialized. Use async context manager.")
            
            logger.debug(f"Making GDELT API request: {url} params={params}")
            
            async with self.session.get(url, params=params) as response:
                status = response.status
                if status != 200:
                    body_preview = (await response.text())[:400].replace("\n", " ")
                    logger.error(f"GDELT HTTP {status} for {url} params={params} body: {body_preview}")
                    response.raise_for_status()
                
                # GDELT returns CSV or JSON depending on the endpoint
                content_type = response.headers.get('content-type', '').lower()
                
                if 'json' in content_type:
                    try:
                        data = await response.json()
                        return data
                    except Exception:
                        text = await response.text()
                        preview = text[:500].replace("\n", " ")
                        logger.error(f"GDELT JSON parse error. Raw preview: {preview}")
                        raise
                else:
                    # Some GDELT servers return JSON with text/plain content-type
                    text = await response.text()
                    try:
                        data = json.loads(text)
                        logger.debug("Parsed JSON from non-JSON content-type response")
                        return data
                    except Exception:
                        preview = text[:500].replace("\n", " ")
                        logger.error(f"GDELT non-JSON response (content-type={content_type}). Preview: {preview}")
                        return {"raw_data": text}
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test basic GDELT API connectivity with a simple query."""
        try:
            # Try a very simple query that should always work
            test_params = {
                "query": "Apple",
                "startdatetime": "20250101000000",
                "enddatetime": "20250102000000",
                "maxrecords": 5
            }
            
            url = f"{self.base_url}/doc/doc"
            result = await self._make_request(url, test_params)
            
            if "raw_data" in result:
                logger.warning("GDELT returned raw data instead of JSON")
                return {"status": "warning", "message": "API responding but format issues"}
            
            return {"status": "healthy", "message": "GDELT API responding normally"}
            
        except Exception as e:
            logger.error(f"GDELT connection test failed: {e}")
            return {"status": "error", "message": f"Connection failed: {str(e)}"}
    
    async def get_doc_search(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_records: int = 250
    ) -> Dict[str, Any]:
        """Get document search results with fallback to v1 API if v2 fails."""
        try:
            return await self._get_doc_search_v2(query, start_date, end_date, max_records)
        except Exception as e:
            logger.warning(f"GDELT v2 API failed, trying v1 fallback: {e}")
            try:
                return await self._get_doc_search_v1(query, start_date, end_date, max_records)
            except Exception as e2:
                logger.error(f"Both GDELT APIs failed: v2={e}, v1={e2}")
                raise
    
    async def _get_doc_search_v2(
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
            num = len(result.get("articles", []))
            logger.info(f"Retrieved {num} articles for query: {query}")
            if num == 0:
                # Retry using a timespan parameter instead of explicit datetimes
                hours = max(1, math.ceil((end_date - start_date).total_seconds() / 3600))
                params_ts = {
                    "query": query,
                    "mode": "artlist",
                    "format": "json",
                    "maxrecords": max_records,
                    "timespan": f"{hours}h",
                    "sort": "datedesc"
                }
                logger.info(f"No results with start/end. Retrying with timespan={hours}h")
                result = await self._make_request(url, params_ts)
            return result
        except Exception as e:
            logger.error(f"Error fetching GDELT doc search for {query}: {e}")
            return {"articles": []}
    

    async def get_doc_search_timespan(
        self,
        query: str,
        timespan_hours: int,
        max_pages: int = 4,
        page_size: int = 250
    ) -> Dict[str, Any]:
        """Search using timespan and simple pagination to gather more records."""
        url = f"{self.base_url}/doc/doc"
        collected: List[Dict[str, Any]] = []
        start_record = 1
        for page_index in range(max_pages):
            params = {
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": page_size,
                "timespan": f"{max(1, timespan_hours)}h",
                "sort": "datedesc",
                "startrecord": start_record
            }
            try:
                result = await self._make_request(url, params)
                page_articles = result.get("articles", []) if isinstance(result, dict) else []
                logger.info(
                    f"Timespan page {page_index + 1}: {len(page_articles)} articles for query: {query} (startrecord={start_record})"
                )
                collected.extend(page_articles)
                if len(page_articles) < page_size:
                    break
                start_record += page_size
            except Exception as e:
                logger.error(f"Error in timespan pagination: {e}")
                break
        return {"articles": collected}


    
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
        # Skip very short tickers that GDELT will reject
        if len(ticker) < 3:
            logger.warning(f"Skipping {ticker} - ticker too short for GDELT API")
            return {
                "articles": [],
                "gkg_data": [],
                "timeline": [],
                "ticker": ticker,
                "company_name": company_name,
                "search_period": {
                    "start": (datetime.utcnow() - timedelta(hours=hours_back)).isoformat(),
                    "end": datetime.utcnow().isoformat()
                }
            }
            
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours_back)
        
        # Build multiple query variants to maximize matches
        def strip_suffixes(name: str) -> str:
            suffixes = [
                "Inc.", "Inc", "Corporation", "Corp.", "Corp", "Ltd.", "Ltd", "PLC", "LLC", "N.V.", "N.V"
            ]
            parts = [p for p in name.split() if p not in suffixes]
            return " ".join(parts) if parts else name
        company_core = strip_suffixes(company_name)

        # Use simpler, GDELT-compatible query formats
        query_variants = [
            f'{ticker}',  # Simple ticker
            f'"{company_name}"',  # Company name in quotes
            f'title:{ticker}',  # Ticker in title
        ]
        
        try:
            # Use fixed 6-hour window for consistent results
            window_hours = 6  # Fixed window instead of trying multiple
            articles: List[Dict[str, Any]] = []
            chosen_query: Optional[str] = None
            
            for q in query_variants:
                # Use timespan search with single window
                result = await self.get_doc_search_timespan(
                    q,
                    window_hours,
                    max_pages=1,  # Single page to avoid excessive API calls
                    page_size=50   # Smaller page size
                )
                articles = result.get("articles", [])
                logger.info(
                    f"Query variant returned {len(articles)} articles for {ticker}: q='{q}', window={window_hours}h"
                )
                if len(articles) > 0:
                    chosen_query = q
                    break
            
            return {
                "articles": articles,
                "gkg_data": [],
                "timeline": [],
                "ticker": ticker,
                "company_name": company_name,
                "search_period": {
                    "start": (end_date - timedelta(hours=window_hours)).isoformat(),
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
        title = (article.get('title') or '').lower()
        url = (article.get('url') or '').lower()
        domain_field = (article.get('domain') or '').lower()
        parsed = urlparse(url)
        host = (parsed.netloc or '').lower()
        
        # Direct mentions in title are highly relevant
        company_core = company_name.lower().replace(' inc.', '').replace(' inc', '')
        if ticker.lower() in title or company_name.lower() in title or company_core in title:
            return True
        
        # Financial domains (broader set)
        financial_domains = [
            'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'cnbc.com',
            'marketwatch.com', 'barrons.com', 'seekingalpha.com', 'investing.com',
            'morningstar.com', 'thestreet.com', 'zacks.com', 'benzinga.com',
            'finance.yahoo.com', 'markets.businessinsider.com', 'businessinsider.com', 'forbes.com'
        ]
        
        def host_matches(fin_domain: str) -> bool:
            return host.endswith(fin_domain) or fin_domain in url or domain_field.endswith(fin_domain)
        
        if any(host_matches(d) for d in financial_domains):
            return True
        
        # Financial keywords in title
        financial_keywords = [
            'stock', 'stocks', 'share', 'shares', 'trading', 'trade', 'earnings',
            'revenue', 'profit', 'loss', 'guidance', 'market', 'investor',
            'analyst', 'downgrade', 'upgrade', 'dividend', 'merger', 'acquisition',
            'ipo', 'nyse', 'nasdaq', 'sec', 'buyback', 'valuation', 'forecast'
        ]
        if any(k in title for k in financial_keywords):
            return True
        
        return False
        
    except Exception:
        # Be conservative on errors: include the article rather than miss a signal
        return True

    async def _get_doc_search_v1(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_records: int = 250
    ) -> Dict[str, Any]:
        """Fallback to GDELT v1 API with different endpoint structure."""
        try:
            # v1 API uses different parameter names
            params = {
                "query": query,
                "start": start_date.strftime("%Y%m%d%H%M%S"),
                "end": end_date.strftime("%Y%m%d%H%M%S"),
                "max": max_records
            }
            
            url = f"{self.fallback_url}/api/v1/doc/doc"
            result = await self._make_request(url, params)
            
            # v1 returns different format, convert to match v2
            if "raw_data" in result:
                logger.warning("GDELT v1 returned raw data")
                return {"articles": []}
            
            # Convert v1 format to v2 format
            articles = []
            if "results" in result:
                for item in result["results"]:
                    article = {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "seendate": item.get("seendate", ""),
                        "domain": item.get("domain", ""),
                        "language": item.get("language", "english")
                    }
                    articles.append(article)
            
            return {"articles": articles}
            
        except Exception as e:
            logger.error(f"GDELT v1 API failed: {e}")
            return {"articles": []}



