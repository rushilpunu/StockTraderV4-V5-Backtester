"""Volatility analysis and event detection system."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import statistics
from loguru import logger

from .config import config
from .data_processor import EventData, SentimentMetrics


@dataclass
class VolatilitySignal:
    """Container for volatility detection signals."""
    ticker: str
    signal_type: str
    strength: float  # 0.0 to 1.0
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    triggers: List[str]
    metrics: Dict[str, Any]
    recommended_action: str


@dataclass
class MarketEvent:
    """Container for detected market events."""
    event_id: str
    ticker: str
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: datetime
    duration_minutes: int
    affected_sentiment: float
    volatility_score: float
    supporting_data: Dict[str, Any]


class VolatilityAnalyzer:
    """Advanced volatility analysis and event detection."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.historical_volatility: Dict[str, List[float]] = {}
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.detected_events: List[MarketEvent] = []
        
    def analyze_volatility(self, event_data: EventData) -> VolatilitySignal:
        """
        Analyze volatility indicators from event data.
        
        Args:
            event_data: Processed event data from GDELT
            
        Returns:
            VolatilitySignal with analysis results
        """
        ticker = event_data.ticker
        
        # Initialize baseline if not exists
        if ticker not in self.baseline_metrics:
            self._initialize_baseline(ticker)
        
        # Calculate various volatility indicators
        indicators = self._calculate_volatility_indicators(event_data)
        
        # Detect anomalies using multiple methods
        anomaly_scores = self._detect_anomalies(ticker, indicators)
        
        # Generate volatility signal
        signal = self._generate_volatility_signal(ticker, indicators, anomaly_scores)
        
        # Update historical data
        self._update_historical_data(ticker, indicators)
        
        # Detect market events
        market_event = self._detect_market_event(event_data, signal)
        if market_event:
            self.detected_events.append(market_event)
            self._maintain_event_history()
        
        logger.info(
            f"Volatility analysis for {ticker}: {signal.signal_type} "
            f"({signal.strength:.3f} strength, {signal.confidence:.3f} confidence)"
        )
        
        return signal
    
    def _calculate_volatility_indicators(self, event_data: EventData) -> Dict[str, float]:
        """Calculate comprehensive volatility indicators."""
        indicators = {}
        
        if not event_data.sentiment_metrics:
            return {"error": 1.0}
        
        sentiment = event_data.sentiment_metrics
        
        # Basic sentiment indicators
        indicators['sentiment_score'] = sentiment.average_sentiment
        indicators['sentiment_variance'] = sentiment.sentiment_variance
        indicators['sentiment_change'] = sentiment.sentiment_change
        indicators['weighted_sentiment'] = sentiment.weighted_sentiment
        
        # Volume indicators
        indicators['article_volume'] = event_data.event_volume
        indicators['volume_normalized'] = min(event_data.event_volume / 20.0, 1.0)
        
        # Sentiment distribution indicators
        total_articles = sentiment.total_articles
        if total_articles > 0:
            indicators['positive_ratio'] = sentiment.positive_count / total_articles
            indicators['negative_ratio'] = sentiment.negative_count / total_articles
            indicators['neutral_ratio'] = sentiment.neutral_count / total_articles
            indicators['sentiment_polarization'] = (
                abs(indicators['positive_ratio'] - indicators['negative_ratio'])
            )
        else:
            indicators['positive_ratio'] = 0.0
            indicators['negative_ratio'] = 0.0
            indicators['neutral_ratio'] = 1.0
            indicators['sentiment_polarization'] = 0.0
        
        # Confidence and relevance indicators
        indicators['confidence_score'] = sentiment.confidence_score
        indicators['relevance_score'] = event_data.relevance_score
        
        # Theme-based indicators
        indicators['theme_diversity'] = len(event_data.themes) / 10.0  # Normalized
        
        # Calculate sentiment momentum (rate of change)
        indicators['sentiment_momentum'] = self._calculate_sentiment_momentum(
            event_data.ticker, sentiment.average_sentiment
        )
        
        # Calculate sentiment acceleration (change in momentum)
        indicators['sentiment_acceleration'] = self._calculate_sentiment_acceleration(
            event_data.ticker, indicators['sentiment_momentum']
        )
        
        # Calculate volatility index combining multiple factors
        indicators['volatility_index'] = self._calculate_volatility_index(indicators)
        
        return indicators
    
    def _calculate_sentiment_momentum(self, ticker: str, current_sentiment: float) -> float:
        """Calculate sentiment momentum (rate of change)."""
        if ticker not in self.historical_volatility:
            return 0.0
        
        history = self.historical_volatility[ticker]
        if len(history) < 2:
            return 0.0
        
        # Calculate momentum as the difference from recent average
        recent_avg = statistics.mean(history[-3:]) if len(history) >= 3 else history[-1]
        return current_sentiment - recent_avg
    
    def _calculate_sentiment_acceleration(self, ticker: str, current_momentum: float) -> float:
        """Calculate sentiment acceleration (change in momentum)."""
        if not hasattr(self, '_momentum_history'):
            self._momentum_history = {}
        
        if ticker not in self._momentum_history:
            self._momentum_history[ticker] = []
        
        momentum_history = self._momentum_history[ticker]
        
        if len(momentum_history) < 2:
            momentum_history.append(current_momentum)
            return 0.0
        
        # Calculate acceleration as change in momentum
        previous_momentum = momentum_history[-1]
        acceleration = current_momentum - previous_momentum
        
        # Update history
        momentum_history.append(current_momentum)
        if len(momentum_history) > 10:  # Keep last 10 values
            momentum_history.pop(0)
        
        return acceleration
    
    def _calculate_volatility_index(self, indicators: Dict[str, float]) -> float:
        """Calculate composite volatility index."""
        # Weighted combination of key volatility factors
        weights = {
            'sentiment_variance': 0.2,
            'sentiment_change': 0.25,
            'sentiment_polarization': 0.15,
            'volume_normalized': 0.15,
            'sentiment_momentum': 0.15,
            'sentiment_acceleration': 0.1
        }
        
        volatility_index = 0.0
        total_weight = 0.0
        
        for factor, weight in weights.items():
            if factor in indicators:
                # Normalize and scale the factor
                factor_value = abs(indicators[factor])
                if factor in ['sentiment_variance', 'sentiment_polarization']:
                    factor_value = min(factor_value, 1.0)  # Cap at 1.0
                elif factor in ['sentiment_change', 'sentiment_momentum', 'sentiment_acceleration']:
                    factor_value = min(abs(factor_value), 1.0)  # Use absolute value and cap
                
                volatility_index += factor_value * weight
                total_weight += weight
        
        return volatility_index / total_weight if total_weight > 0 else 0.0
    
    def _detect_anomalies(self, ticker: str, indicators: Dict[str, float]) -> Dict[str, float]:
        """Detect anomalies using multiple statistical methods."""
        anomaly_scores = {}
        
        # Z-score based anomaly detection
        if ticker in self.baseline_metrics:
            baseline = self.baseline_metrics[ticker]
            
            for key, value in indicators.items():
                if key in baseline and baseline[f"{key}_std"] > 0:
                    z_score = abs(value - baseline[f"{key}_mean"]) / baseline[f"{key}_std"]
                    anomaly_scores[f"{key}_zscore"] = min(z_score / 3.0, 1.0)  # Normalize to 0-1
        
        # Threshold-based anomaly detection
        thresholds = {
            'sentiment_score': config.sentiment_threshold_high,
            'sentiment_change': config.sentiment_change_threshold,
            'article_volume': config.event_count_threshold,
            'volatility_index': 0.7,
            'sentiment_polarization': 0.8
        }
        
        for key, threshold in thresholds.items():
            if key in indicators:
                if key == 'sentiment_score':
                    # Bidirectional threshold for sentiment
                    anomaly_scores[f"{key}_threshold"] = max(
                        (indicators[key] - threshold) / (1.0 - threshold),
                        (config.sentiment_threshold_low - indicators[key]) / 
                        (config.sentiment_threshold_low - (-1.0))
                    ) if indicators[key] > threshold or indicators[key] < config.sentiment_threshold_low else 0.0
                else:
                    anomaly_scores[f"{key}_threshold"] = max(0.0, 
                        (indicators[key] - threshold) / threshold
                    ) if indicators[key] > threshold else 0.0
        
        # Composite anomaly score
        if anomaly_scores:
            anomaly_scores['composite'] = statistics.mean(anomaly_scores.values())
        else:
            anomaly_scores['composite'] = 0.0
        
        return anomaly_scores
    
    def _generate_volatility_signal(
        self, 
        ticker: str, 
        indicators: Dict[str, float], 
        anomaly_scores: Dict[str, float]
    ) -> VolatilitySignal:
        """Generate volatility signal from indicators and anomaly scores."""
        
        # Determine signal strength
        strength = anomaly_scores.get('composite', 0.0)
        
        # Determine direction based on sentiment and momentum
        sentiment_score = indicators.get('sentiment_score', 0.0)
        sentiment_momentum = indicators.get('sentiment_momentum', 0.0)
        
        if sentiment_score > 0.1 and sentiment_momentum > 0:
            direction = 'bullish'
        elif sentiment_score < -0.1 and sentiment_momentum < 0:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Determine signal type
        volatility_index = indicators.get('volatility_index', 0.0)
        
        if strength > 0.7:
            signal_type = 'high_volatility'
        elif strength > 0.4:
            signal_type = 'medium_volatility'
        elif volatility_index > 0.5:
            signal_type = 'emerging_volatility'
        else:
            signal_type = 'low_volatility'
        
        # Calculate confidence
        confidence_factors = [
            indicators.get('confidence_score', 0.0),
            indicators.get('relevance_score', 0.0),
            min(indicators.get('volume_normalized', 0.0), 1.0),
            1.0 - indicators.get('sentiment_variance', 0.0)  # Lower variance = higher confidence
        ]
        
        confidence = statistics.mean([f for f in confidence_factors if f >= 0])
        
        # Identify triggers
        triggers = []
        for key, score in anomaly_scores.items():
            if score > 0.5 and key != 'composite':
                triggers.append(key.replace('_threshold', '').replace('_zscore', ''))
        
        # Determine recommended action (more aggressive for medium/emerging)
        if signal_type == 'high_volatility' and confidence > 0.55:
            if direction == 'bullish':
                recommended_action = 'buy'
            elif direction == 'bearish':
                recommended_action = 'sell'
            else:
                recommended_action = 'hold'
        elif signal_type in ['medium_volatility', 'emerging_volatility'] and confidence > 0.5:
            if direction == 'bullish':
                recommended_action = 'buy'
            elif direction == 'bearish':
                recommended_action = 'sell'
            else:
                recommended_action = 'monitor'
        else:
            recommended_action = 'hold'
        
        return VolatilitySignal(
            ticker=ticker,
            signal_type=signal_type,
            strength=strength,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            triggers=triggers,
            metrics=indicators,
            recommended_action=recommended_action
        )
    
    def _detect_market_event(self, event_data: EventData, signal: VolatilitySignal) -> Optional[MarketEvent]:
        """Detect significant market events."""
        
        # Only create events for significant signals
        if signal.strength < 0.5 or signal.confidence < 0.4:
            return None
        
        # Determine event severity
        if signal.strength > 0.8 and signal.confidence > 0.7:
            severity = 'critical'
        elif signal.strength > 0.6 and signal.confidence > 0.5:
            severity = 'high'
        elif signal.strength > 0.4:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Generate event description
        sentiment = event_data.sentiment_metrics
        description = (
            f"{signal.direction.title()} sentiment signal detected for {event_data.ticker}. "
            f"Sentiment: {sentiment.average_sentiment:.3f}, "
            f"Volume: {event_data.event_volume} articles, "
            f"Change: {sentiment.sentiment_change:.3f}"
        )
        
        # Add top themes to description
        if event_data.themes:
            top_themes = event_data.themes[:3]
            description += f". Key themes: {', '.join(top_themes)}"
        
        event_id = f"{event_data.ticker}_{int(datetime.utcnow().timestamp())}"
        
        return MarketEvent(
            event_id=event_id,
            ticker=event_data.ticker,
            event_type=signal.signal_type,
            severity=severity,
            description=description,
            timestamp=datetime.utcnow(),
            duration_minutes=0,  # Will be updated as event evolves
            affected_sentiment=sentiment.average_sentiment,
            volatility_score=signal.strength,
            supporting_data={
                'signal': signal,
                'indicators': signal.metrics,
                'themes': event_data.themes,
                'article_count': event_data.event_volume
            }
        )
    
    def _initialize_baseline(self, ticker: str):
        """Initialize baseline metrics for a ticker."""
        self.baseline_metrics[ticker] = {
            'sentiment_score_mean': 0.0,
            'sentiment_score_std': 0.3,
            'sentiment_variance_mean': 0.2,
            'sentiment_variance_std': 0.1,
            'article_volume_mean': 5.0,
            'article_volume_std': 3.0,
            'volatility_index_mean': 0.3,
            'volatility_index_std': 0.2
        }
        
        self.historical_volatility[ticker] = []
    
    def _update_historical_data(self, ticker: str, indicators: Dict[str, float]):
        """Update historical data and baseline metrics."""
        if ticker not in self.historical_volatility:
            self.historical_volatility[ticker] = []
        
        # Add current volatility index to history
        volatility_index = indicators.get('volatility_index', 0.0)
        self.historical_volatility[ticker].append(volatility_index)
        
        # Maintain rolling window
        if len(self.historical_volatility[ticker]) > 100:
            self.historical_volatility[ticker].pop(0)
        
        # Update baseline metrics if we have enough data
        if len(self.historical_volatility[ticker]) >= 10:
            self._update_baseline_metrics(ticker, indicators)
    
    def _update_baseline_metrics(self, ticker: str, indicators: Dict[str, float]):
        """Update baseline metrics using historical data."""
        history = self.historical_volatility[ticker]
        
        # Update volatility baseline
        self.baseline_metrics[ticker]['volatility_index_mean'] = statistics.mean(history)
        self.baseline_metrics[ticker]['volatility_index_std'] = statistics.stdev(history)
        
        # Update other metrics with exponential smoothing
        alpha = 0.1  # Smoothing factor
        
        for key, value in indicators.items():
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            
            if mean_key in self.baseline_metrics[ticker]:
                old_mean = self.baseline_metrics[ticker][mean_key]
                self.baseline_metrics[ticker][mean_key] = (
                    alpha * value + (1 - alpha) * old_mean
                )
                
                # Update standard deviation estimate
                old_std = self.baseline_metrics[ticker][std_key]
                diff = abs(value - old_mean)
                self.baseline_metrics[ticker][std_key] = (
                    alpha * diff + (1 - alpha) * old_std
                )
    
    def _maintain_event_history(self):
        """Maintain rolling window of detected events."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        self.detected_events = [
            event for event in self.detected_events
            if event.timestamp > cutoff_time
        ]
    
    def get_recent_events(self, ticker: str = None, hours_back: int = 6) -> List[MarketEvent]:
        """Get recent market events, optionally filtered by ticker."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        events = [
            event for event in self.detected_events
            if event.timestamp > cutoff_time
        ]
        
        if ticker:
            events = [event for event in events if event.ticker == ticker]
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)
    
    def get_volatility_summary(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive volatility summary for a ticker."""
        if ticker not in self.historical_volatility:
            return {"error": "No historical data available"}
        
        history = self.historical_volatility[ticker]
        recent_events = self.get_recent_events(ticker, hours_back=24)
        
        if not history:
            return {"error": "Insufficient historical data"}
        
        current_volatility = history[-1] if history else 0.0
        avg_volatility = statistics.mean(history)
        volatility_trend = (
            statistics.mean(history[-5:]) - statistics.mean(history[-10:-5])
            if len(history) >= 10 else 0.0
        )
        
        return {
            "ticker": ticker,
            "current_volatility": current_volatility,
            "average_volatility": avg_volatility,
            "volatility_trend": volatility_trend,
            "volatility_percentile": (
                sum(1 for v in history if v <= current_volatility) / len(history)
            ),
            "recent_events_count": len(recent_events),
            "highest_severity_event": (
                max(recent_events, key=lambda x: {
                    'low': 1, 'medium': 2, 'high': 3, 'critical': 4
                }.get(x.severity, 0)).severity
                if recent_events else "none"
            ),
            "data_points": len(history),
            "last_updated": datetime.utcnow().isoformat()
        }
