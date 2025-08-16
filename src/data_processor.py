"""Data processing and sentiment aggregation for GDELT data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
from loguru import logger

from .config import config
from .gdelt_client import extract_sentiment_from_article, extract_themes_from_gkg, is_relevant_to_stock


@dataclass
class SentimentMetrics:
    """Container for sentiment analysis metrics."""
    average_sentiment: float
    sentiment_variance: float
    positive_count: int
    negative_count: int
    neutral_count: int
    total_articles: int
    sentiment_change: float
    weighted_sentiment: float
    confidence_score: float


@dataclass
class EventData:
    """Container for processed event data."""
    ticker: str
    company_name: str
    timestamp: datetime
    articles: List[Dict[str, Any]] = field(default_factory=list)
    gkg_data: List[Dict[str, Any]] = field(default_factory=list)
    timeline_data: List[Dict[str, Any]] = field(default_factory=list)
    sentiment_metrics: Optional[SentimentMetrics] = None
    themes: List[str] = field(default_factory=list)
    event_volume: int = 0
    relevance_score: float = 0.0


class DataProcessor:
    """Process and aggregate GDELT data for sentiment analysis."""
    
    def __init__(self):
        self.historical_data: Dict[str, List[EventData]] = defaultdict(list)
        self.sentiment_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
    
    def process_gdelt_response(self, gdelt_data: Dict[str, Any]) -> EventData:
        """
        Process raw GDELT API response into structured EventData.
        
        Args:
            gdelt_data: Raw response from GDELT API
            
        Returns:
            Processed EventData object
        """
        ticker = gdelt_data.get("ticker", "")
        company_name = gdelt_data.get("company_name", "")
        
        event_data = EventData(
            ticker=ticker,
            company_name=company_name,
            timestamp=datetime.utcnow()
        )
        
        # Process articles
        articles = gdelt_data.get("articles", [])
        relevant_articles = []
        
        for article in articles:
            if is_relevant_to_stock(article, ticker, company_name):
                relevant_articles.append(article)
        
        event_data.articles = relevant_articles
        event_data.gkg_data = gdelt_data.get("gkg_data", [])
        event_data.timeline_data = gdelt_data.get("timeline", [])
        
        # Calculate sentiment metrics
        event_data.sentiment_metrics = self._calculate_sentiment_metrics(relevant_articles, ticker)
        
        # Extract themes from GKG data
        event_data.themes = self._extract_all_themes(gdelt_data.get("gkg_data", []))
        
        # Calculate event volume and relevance
        event_data.event_volume = len(relevant_articles)
        event_data.relevance_score = self._calculate_relevance_score(event_data)
        
        # Store in historical data
        self.historical_data[ticker].append(event_data)
        
        # Maintain rolling window of historical data
        self._maintain_rolling_window(ticker)
        
        # Update sentiment history
        if event_data.sentiment_metrics:
            self.sentiment_history[ticker].append(
                (event_data.timestamp, event_data.sentiment_metrics.average_sentiment)
            )
            self._maintain_sentiment_history(ticker)
        
        logger.info(
            f"Processed {event_data.event_volume} relevant articles for {ticker} "
            f"with average sentiment: {event_data.sentiment_metrics.average_sentiment:.3f}"
        )
        
        return event_data
    
    def _calculate_sentiment_metrics(self, articles: List[Dict[str, Any]], ticker: str) -> SentimentMetrics:
        """Calculate comprehensive sentiment metrics from articles."""
        if not articles:
            return SentimentMetrics(
                average_sentiment=0.0,
                sentiment_variance=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                total_articles=0,
                sentiment_change=0.0,
                weighted_sentiment=0.0,
                confidence_score=0.0
            )
        
        sentiments = []
        weights = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            sentiment = extract_sentiment_from_article(article)
            if sentiment is not None:
                sentiments.append(sentiment)
                
                # Weight by recency and source credibility
                weight = self._calculate_article_weight(article)
                weights.append(weight)
                
                # Count sentiment categories
                if sentiment > 0.1:
                    positive_count += 1
                elif sentiment < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1
        
        if not sentiments:
            return SentimentMetrics(
                average_sentiment=0.0,
                sentiment_variance=0.0,
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                total_articles=len(articles),
                sentiment_change=0.0,
                weighted_sentiment=0.0,
                confidence_score=0.0
            )
        
        # Calculate metrics
        average_sentiment = statistics.mean(sentiments)
        sentiment_variance = statistics.variance(sentiments) if len(sentiments) > 1 else 0.0
        
        # Calculate weighted sentiment
        if weights:
            weighted_sentiment = np.average(sentiments, weights=weights)
        else:
            weighted_sentiment = average_sentiment
        
        # Calculate sentiment change compared to historical average
        sentiment_change = self._calculate_sentiment_change(ticker, average_sentiment)
        
        # Calculate confidence score based on volume and agreement
        confidence_score = self._calculate_confidence_score(sentiments, len(articles))
        
        return SentimentMetrics(
            average_sentiment=average_sentiment,
            sentiment_variance=sentiment_variance,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            total_articles=len(articles),
            sentiment_change=sentiment_change,
            weighted_sentiment=weighted_sentiment,
            confidence_score=confidence_score
        )
    
    def _calculate_article_weight(self, article: Dict[str, Any]) -> float:
        """Calculate weight for an article based on source credibility and recency."""
        weight = 1.0
        
        # Weight by source credibility
        url = article.get('url', '').lower()
        high_credibility_sources = [
            'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com',
            'cnbc.com', 'marketwatch.com', 'barrons.com'
        ]
        
        if any(source in url for source in high_credibility_sources):
            weight *= 1.5
        
        # Weight by recency (articles from last hour get higher weight)
        try:
            article_date = datetime.fromisoformat(article.get('seendate', ''))
            hours_old = (datetime.utcnow() - article_date).total_seconds() / 3600
            if hours_old < 1:
                weight *= 1.3
            elif hours_old < 6:
                weight *= 1.1
        except (ValueError, TypeError):
            pass
        
        return weight
    
    def _calculate_sentiment_change(self, ticker: str, current_sentiment: float) -> float:
        """Calculate sentiment change compared to recent history."""
        if ticker not in self.sentiment_history or len(self.sentiment_history[ticker]) < 2:
            return 0.0
        
        # Get average sentiment from last few data points
        recent_history = self.sentiment_history[ticker][-5:]  # Last 5 data points
        historical_avg = statistics.mean([sent for _, sent in recent_history])
        
        return current_sentiment - historical_avg
    
    def _calculate_confidence_score(self, sentiments: List[float], total_articles: int) -> float:
        """Calculate confidence score based on sentiment agreement and volume."""
        if not sentiments:
            return 0.0
        
        # Base confidence on article volume
        volume_score = min(total_articles / 10.0, 1.0)  # Max score at 10+ articles
        
        # Confidence based on sentiment agreement (low variance = high agreement)
        if len(sentiments) > 1:
            variance = statistics.variance(sentiments)
            agreement_score = max(0.0, 1.0 - variance)
        else:
            agreement_score = 0.5
        
        return (volume_score + agreement_score) / 2.0
    
    def _extract_all_themes(self, gkg_data: List[Dict[str, Any]]) -> List[str]:
        """Extract and aggregate all themes from GKG data."""
        all_themes = []
        for entry in gkg_data:
            themes = extract_themes_from_gkg(entry)
            all_themes.extend(themes)
        
        # Count theme frequency and return most common ones
        theme_counts = defaultdict(int)
        for theme in all_themes:
            theme_counts[theme] += 1
        
        # Return themes sorted by frequency
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:10]]  # Top 10 themes
    
    def _calculate_relevance_score(self, event_data: EventData) -> float:
        """Calculate overall relevance score for the event data."""
        score = 0.0
        
        # Base score from article volume
        score += min(event_data.event_volume / 20.0, 0.4)  # Max 0.4 for volume
        
        # Score from sentiment metrics confidence
        if event_data.sentiment_metrics:
            score += event_data.sentiment_metrics.confidence_score * 0.3
        
        # Score from theme relevance
        financial_themes = [
            'STOCK_MARKET', 'EARNINGS', 'REVENUE', 'PROFIT', 'INVESTMENT',
            'FINANCIAL_CRISIS', 'MARKET_VOLATILITY', 'CORPORATE_GOVERNANCE'
        ]
        
        relevant_themes = sum(1 for theme in event_data.themes 
                            if any(fin_theme in theme.upper() for fin_theme in financial_themes))
        score += min(relevant_themes / 5.0, 0.3)  # Max 0.3 for themes
        
        return min(score, 1.0)
    
    def _maintain_rolling_window(self, ticker: str):
        """Maintain rolling window of historical data."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=config.rolling_window_minutes * 2)
        
        self.historical_data[ticker] = [
            data for data in self.historical_data[ticker]
            if data.timestamp > cutoff_time
        ]
    
    def _maintain_sentiment_history(self, ticker: str):
        """Maintain rolling window of sentiment history."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)  # Keep 24 hours of history
        
        self.sentiment_history[ticker] = [
            (timestamp, sentiment) for timestamp, sentiment in self.sentiment_history[ticker]
            if timestamp > cutoff_time
        ]
    
    def get_aggregated_metrics(self, ticker: str, minutes_back: int = None) -> Optional[Dict[str, Any]]:
        """
        Get aggregated metrics for a ticker over a specified time window.
        
        Args:
            ticker: Stock ticker symbol
            minutes_back: How many minutes back to aggregate (default: config window)
            
        Returns:
            Dictionary with aggregated metrics
        """
        if minutes_back is None:
            minutes_back = config.rolling_window_minutes
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes_back)
        
        recent_data = [
            data for data in self.historical_data.get(ticker, [])
            if data.timestamp > cutoff_time
        ]
        
        if not recent_data:
            return None
        
        # Aggregate metrics
        total_articles = sum(data.event_volume for data in recent_data)
        
        # Aggregate sentiment
        all_sentiments = []
        for data in recent_data:
            if data.sentiment_metrics and data.sentiment_metrics.total_articles > 0:
                # Weight by article count
                weight = data.sentiment_metrics.total_articles
                all_sentiments.extend([data.sentiment_metrics.average_sentiment] * weight)
        
        if not all_sentiments:
            return None
        
        aggregated_sentiment = statistics.mean(all_sentiments)
        sentiment_std = statistics.stdev(all_sentiments) if len(all_sentiments) > 1 else 0.0
        
        # Calculate sentiment trend
        if len(recent_data) >= 2:
            first_half = recent_data[:len(recent_data)//2]
            second_half = recent_data[len(recent_data)//2:]
            
            first_sentiment = statistics.mean([
                d.sentiment_metrics.average_sentiment for d in first_half
                if d.sentiment_metrics
            ]) if first_half else 0.0
            
            second_sentiment = statistics.mean([
                d.sentiment_metrics.average_sentiment for d in second_half
                if d.sentiment_metrics
            ]) if second_half else 0.0
            
            sentiment_trend = second_sentiment - first_sentiment
        else:
            sentiment_trend = 0.0
        
        # Aggregate themes
        all_themes = []
        for data in recent_data:
            all_themes.extend(data.themes)
        
        theme_counts = defaultdict(int)
        for theme in all_themes:
            theme_counts[theme] += 1
        
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "ticker": ticker,
            "time_window_minutes": minutes_back,
            "total_articles": total_articles,
            "data_points": len(recent_data),
            "aggregated_sentiment": aggregated_sentiment,
            "sentiment_std": sentiment_std,
            "sentiment_trend": sentiment_trend,
            "top_themes": top_themes,
            "latest_update": recent_data[-1].timestamp.isoformat() if recent_data else None,
            "confidence_score": statistics.mean([
                data.sentiment_metrics.confidence_score for data in recent_data
                if data.sentiment_metrics
            ]) if recent_data else 0.0
        }
    
    def detect_sentiment_anomalies(self, ticker: str) -> Dict[str, Any]:
        """
        Detect sentiment anomalies that might indicate trading opportunities.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with anomaly detection results
        """
        current_metrics = self.get_aggregated_metrics(ticker, config.analysis_interval_minutes)
        historical_metrics = self.get_aggregated_metrics(ticker, config.rolling_window_minutes)
        
        if not current_metrics or not historical_metrics:
            return {"anomalies_detected": False, "reason": "Insufficient data"}
        
        anomalies = {
            "anomalies_detected": False,
            "anomaly_types": [],
            "current_sentiment": current_metrics["aggregated_sentiment"],
            "historical_sentiment": historical_metrics["aggregated_sentiment"],
            "sentiment_change": current_metrics["aggregated_sentiment"] - historical_metrics["aggregated_sentiment"],
            "confidence": min(current_metrics["confidence_score"], historical_metrics["confidence_score"]),
            "article_volume": current_metrics["total_articles"],
            "recommendations": []
        }
        
        # Check for sentiment spike
        sentiment_change = abs(anomalies["sentiment_change"])
        if sentiment_change > config.sentiment_change_threshold:
            anomalies["anomalies_detected"] = True
            anomalies["anomaly_types"].append("sentiment_spike")
            
            if anomalies["sentiment_change"] > 0:
                anomalies["recommendations"].append("positive_sentiment_surge")
            else:
                anomalies["recommendations"].append("negative_sentiment_surge")
        
        # Check for high/low absolute sentiment
        current_sentiment = current_metrics["aggregated_sentiment"]
        if current_sentiment > config.sentiment_threshold_high:
            anomalies["anomalies_detected"] = True
            anomalies["anomaly_types"].append("high_positive_sentiment")
            anomalies["recommendations"].append("bullish_signal")
        elif current_sentiment < config.sentiment_threshold_low:
            anomalies["anomalies_detected"] = True
            anomalies["anomaly_types"].append("high_negative_sentiment")
            anomalies["recommendations"].append("bearish_signal")
        
        # Check for unusual volume
        if current_metrics["total_articles"] > config.event_count_threshold:
            anomalies["anomalies_detected"] = True
            anomalies["anomaly_types"].append("high_event_volume")
            anomalies["recommendations"].append("increased_attention")
        
        # Check for sentiment volatility
        if current_metrics["sentiment_std"] > 0.5:  # High standard deviation
            anomalies["anomalies_detected"] = True
            anomalies["anomaly_types"].append("high_sentiment_volatility")
            anomalies["recommendations"].append("market_uncertainty")
        
        return anomalies
