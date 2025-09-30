"""Alert generation and management system."""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from loguru import logger

# Import config from the small capital trader directory
import sys
import os
small_capital_path = os.path.join(os.path.dirname(__file__), '..', 'small_capital_trader')
sys.path.insert(0, small_capital_path)
from config import SmallCapitalTradingConfig
config = SmallCapitalTradingConfig()
from volatility_analyzer import VolatilitySignal, MarketEvent
from data_processor import EventData


class AlertLevel(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    SENTIMENT_SPIKE = "sentiment_spike"
    VOLATILITY_SURGE = "volatility_surge"
    VOLUME_ANOMALY = "volume_anomaly"
    MARKET_EVENT = "market_event"
    TRADING_OPPORTUNITY = "trading_opportunity"
    RISK_WARNING = "risk_warning"


@dataclass
class TradingAlert:
    """Container for trading alerts."""
    alert_id: str
    ticker: str
    alert_type: AlertType
    alert_level: AlertLevel
    title: str
    description: str
    timestamp: datetime
    expires_at: datetime
    
    # Core metrics
    sentiment_score: float
    sentiment_change: float
    volatility_score: float
    confidence: float
    
    # Supporting data
    article_count: int
    top_headlines: List[str]
    key_themes: List[str]
    
    # Trading recommendations
    recommended_action: str
    position_size_recommendation: float
    stop_loss_suggestion: Optional[float]
    take_profit_suggestion: Optional[float]
    
    # Metadata
    source_data: Dict[str, Any]
    processed: bool = False
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['timestamp'] = self.timestamp.isoformat()
        data['expires_at'] = self.expires_at.isoformat()
        # Convert enums to strings
        data['alert_type'] = self.alert_type.value
        data['alert_level'] = self.alert_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingAlert':
        """Create alert from dictionary."""
        # Convert string timestamps back to datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        # Convert string enums back to enums
        data['alert_type'] = AlertType(data['alert_type'])
        data['alert_level'] = AlertLevel(data['alert_level'])
        return cls(**data)


class AlertSystem:
    """Comprehensive alert generation and management system."""
    
    def __init__(self):
        self.active_alerts: Dict[str, TradingAlert] = {}
        self.alert_history: List[TradingAlert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = self._initialize_alert_rules()
        self.cooldown_periods: Dict[str, datetime] = {}
        
    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize alert generation rules."""
        return {
            "sentiment_spike": {
                "min_sentiment_change": config.sentiment_change_threshold,
                "min_confidence": 0.3,
                "min_articles": 2,
                "cooldown_minutes": 10
            },
            "volatility_surge": {
                "min_volatility_score": 0.5,
                "min_confidence": 0.4,
                "min_articles": 3,
                "cooldown_minutes": 15
            },
            "volume_anomaly": {
                "min_article_count": max(3, config.event_count_threshold),
                "volume_multiplier": 1.5,
                "cooldown_minutes": 20
            },
            "high_sentiment": {
                "sentiment_threshold": config.sentiment_threshold_high,
                "min_confidence": 0.5,
                "cooldown_minutes": 30
            },
            "low_sentiment": {
                "sentiment_threshold": config.sentiment_threshold_low,
                "min_confidence": 0.5,
                "cooldown_minutes": 30
            },
            "trading_opportunity": {
                "min_signal_strength": 0.5,
                "min_confidence": 0.5,
                "required_action": ["buy", "sell"],
                "cooldown_minutes": 20
            }
        }
    
    def process_event_data(
        self, 
        event_data: EventData, 
        volatility_signal: VolatilitySignal
    ) -> List[TradingAlert]:
        """
        Process event data and generate appropriate alerts.
        
        Args:
            event_data: Processed GDELT event data
            volatility_signal: Volatility analysis results
            
        Returns:
            List of generated alerts
        """
        generated_alerts = []
        ticker = event_data.ticker
        
        # Check if we're in cooldown period for this ticker
        if self._is_in_cooldown(ticker):
            logger.debug(f"Skipping alert generation for {ticker} - in cooldown period")
            return generated_alerts
        
        # Generate different types of alerts
        alert_generators = [
            self._check_sentiment_spike,
            self._check_volatility_surge,
            self._check_volume_anomaly,
            self._check_extreme_sentiment,
            self._check_trading_opportunity,
            self._check_risk_conditions
        ]
        
        for generator in alert_generators:
            try:
                alert = generator(event_data, volatility_signal)
                if alert:
                    generated_alerts.append(alert)
                    self._add_alert(alert)
            except Exception as e:
                logger.error(f"Error in alert generator {generator.__name__}: {e}")
        
        # Update cooldown if alerts were generated
        if generated_alerts:
            self._update_cooldown(ticker)
        
        logger.info(f"Generated {len(generated_alerts)} alerts for {ticker}")
        return generated_alerts
    
    def _check_sentiment_spike(
        self, 
        event_data: EventData, 
        volatility_signal: VolatilitySignal
    ) -> Optional[TradingAlert]:
        """Check for sentiment spike alerts."""
        if not event_data.sentiment_metrics:
            return None
        
        rules = self.alert_rules["sentiment_spike"]
        sentiment = event_data.sentiment_metrics
        
        # Check if conditions are met
        if (abs(sentiment.sentiment_change) < rules["min_sentiment_change"] or
            sentiment.confidence_score < rules["min_confidence"] or
            event_data.event_volume < rules["min_articles"]):
            return None
        
        # Determine alert level
        change_magnitude = abs(sentiment.sentiment_change)
        if change_magnitude > 0.8:
            alert_level = AlertLevel.CRITICAL
        elif change_magnitude > 0.6:
            alert_level = AlertLevel.HIGH
        elif change_magnitude > 0.4:
            alert_level = AlertLevel.MEDIUM
        else:
            alert_level = AlertLevel.LOW
        
        # Create alert
        direction = "positive" if sentiment.sentiment_change > 0 else "negative"
        title = f"{direction.title()} Sentiment Spike - {event_data.ticker}"
        
        description = (
            f"Significant {direction} sentiment change detected for {event_data.ticker}. "
            f"Sentiment changed by {sentiment.sentiment_change:.3f} "
            f"(current: {sentiment.average_sentiment:.3f}). "
            f"Based on {event_data.event_volume} articles with "
            f"{sentiment.confidence_score:.1%} confidence."
        )
        
        return self._create_alert(
            ticker=event_data.ticker,
            alert_type=AlertType.SENTIMENT_SPIKE,
            alert_level=alert_level,
            title=title,
            description=description,
            event_data=event_data,
            volatility_signal=volatility_signal,
            expires_minutes=rules["cooldown_minutes"]
        )
    
    def _check_volatility_surge(
        self, 
        event_data: EventData, 
        volatility_signal: VolatilitySignal
    ) -> Optional[TradingAlert]:
        """Check for volatility surge alerts."""
        rules = self.alert_rules["volatility_surge"]
        
        if (volatility_signal.strength < rules["min_volatility_score"] or
            volatility_signal.confidence < rules["min_confidence"] or
            event_data.event_volume < rules["min_articles"]):
            return None
        
        # Determine alert level based on volatility strength
        if volatility_signal.strength > 0.9:
            alert_level = AlertLevel.CRITICAL
        elif volatility_signal.strength > 0.75:
            alert_level = AlertLevel.HIGH
        elif volatility_signal.strength > 0.6:
            alert_level = AlertLevel.MEDIUM
        else:
            alert_level = AlertLevel.LOW
        
        title = f"Volatility Surge - {event_data.ticker}"
        description = (
            f"High volatility detected for {event_data.ticker} "
            f"({volatility_signal.signal_type}). "
            f"Strength: {volatility_signal.strength:.1%}, "
            f"Direction: {volatility_signal.direction}, "
            f"Confidence: {volatility_signal.confidence:.1%}. "
            f"Key triggers: {', '.join(volatility_signal.triggers[:3])}"
        )
        
        return self._create_alert(
            ticker=event_data.ticker,
            alert_type=AlertType.VOLATILITY_SURGE,
            alert_level=alert_level,
            title=title,
            description=description,
            event_data=event_data,
            volatility_signal=volatility_signal,
            expires_minutes=rules["cooldown_minutes"]
        )
    
    def _check_volume_anomaly(
        self, 
        event_data: EventData, 
        volatility_signal: VolatilitySignal
    ) -> Optional[TradingAlert]:
        """Check for volume anomaly alerts."""
        rules = self.alert_rules["volume_anomaly"]
        
        if event_data.event_volume < rules["min_article_count"]:
            return None
        
        # Check if volume is significantly higher than usual
        # (This would ideally compare to historical averages)
        expected_volume = 5  # Baseline assumption
        volume_ratio = event_data.event_volume / expected_volume
        
        if volume_ratio < rules["volume_multiplier"]:
            return None
        
        # Determine alert level
        if volume_ratio > 5:
            alert_level = AlertLevel.HIGH
        elif volume_ratio > 3:
            alert_level = AlertLevel.MEDIUM
        else:
            alert_level = AlertLevel.LOW
        
        title = f"Volume Anomaly - {event_data.ticker}"
        description = (
            f"Unusual news volume detected for {event_data.ticker}. "
            f"{event_data.event_volume} articles found "
            f"({volume_ratio:.1f}x normal volume). "
            f"This may indicate developing news or market interest."
        )
        
        return self._create_alert(
            ticker=event_data.ticker,
            alert_type=AlertType.VOLUME_ANOMALY,
            alert_level=alert_level,
            title=title,
            description=description,
            event_data=event_data,
            volatility_signal=volatility_signal,
            expires_minutes=rules["cooldown_minutes"]
        )
    
    def _check_extreme_sentiment(
        self, 
        event_data: EventData, 
        volatility_signal: VolatilitySignal
    ) -> Optional[TradingAlert]:
        """Check for extreme sentiment alerts."""
        if not event_data.sentiment_metrics:
            return None
        
        sentiment = event_data.sentiment_metrics.average_sentiment
        
        # Check for extremely positive sentiment
        if sentiment > self.alert_rules["high_sentiment"]["sentiment_threshold"]:
            if event_data.sentiment_metrics.confidence_score < self.alert_rules["high_sentiment"]["min_confidence"]:
                return None
            
            alert_level = AlertLevel.HIGH if sentiment > 0.8 else AlertLevel.MEDIUM
            title = f"Extremely Positive Sentiment - {event_data.ticker}"
            description = (
                f"Extremely positive sentiment detected for {event_data.ticker} "
                f"({sentiment:.3f}). This may indicate overoptimism or "
                f"significant positive developments. Consider profit-taking opportunities."
            )
            
        # Check for extremely negative sentiment
        elif sentiment < self.alert_rules["low_sentiment"]["sentiment_threshold"]:
            if event_data.sentiment_metrics.confidence_score < self.alert_rules["low_sentiment"]["min_confidence"]:
                return None
            
            alert_level = AlertLevel.HIGH if sentiment < -0.8 else AlertLevel.MEDIUM
            title = f"Extremely Negative Sentiment - {event_data.ticker}"
            description = (
                f"Extremely negative sentiment detected for {event_data.ticker} "
                f"({sentiment:.3f}). This may indicate overselling or "
                f"significant negative developments. Consider buying opportunities."
            )
        else:
            return None
        
        return self._create_alert(
            ticker=event_data.ticker,
            alert_type=AlertType.SENTIMENT_SPIKE,
            alert_level=alert_level,
            title=title,
            description=description,
            event_data=event_data,
            volatility_signal=volatility_signal,
            expires_minutes=45
        )
    
    def _check_trading_opportunity(
        self, 
        event_data: EventData, 
        volatility_signal: VolatilitySignal
    ) -> Optional[TradingAlert]:
        """Check for trading opportunity alerts."""
        rules = self.alert_rules["trading_opportunity"]
        
        if (volatility_signal.strength < rules["min_signal_strength"] or
            volatility_signal.confidence < rules["min_confidence"] or
            volatility_signal.recommended_action not in rules["required_action"]):
            return None
        
        alert_level = AlertLevel.HIGH
        action = volatility_signal.recommended_action.upper()
        
        title = f"Trading Opportunity: {action} {event_data.ticker}"
        description = (
            f"Strong {action} signal detected for {event_data.ticker}. "
            f"Signal strength: {volatility_signal.strength:.1%}, "
            f"Confidence: {volatility_signal.confidence:.1%}, "
            f"Direction: {volatility_signal.direction}. "
            f"Consider {action.lower()} position with appropriate risk management."
        )
        
        return self._create_alert(
            ticker=event_data.ticker,
            alert_type=AlertType.TRADING_OPPORTUNITY,
            alert_level=alert_level,
            title=title,
            description=description,
            event_data=event_data,
            volatility_signal=volatility_signal,
            expires_minutes=rules["cooldown_minutes"]
        )
    
    def _check_risk_conditions(
        self, 
        event_data: EventData, 
        volatility_signal: VolatilitySignal
    ) -> Optional[TradingAlert]:
        """Check for risk warning conditions."""
        risk_conditions = []
        
        # High volatility with low confidence
        if volatility_signal.strength > 0.7 and volatility_signal.confidence < 0.4:
            risk_conditions.append("High volatility with low confidence")
        
        # Conflicting signals
        if (event_data.sentiment_metrics and 
            abs(event_data.sentiment_metrics.average_sentiment - 
                event_data.sentiment_metrics.weighted_sentiment) > 0.3):
            risk_conditions.append("Conflicting sentiment signals")
        
        # High sentiment variance (disagreement in market)
        if (event_data.sentiment_metrics and 
            event_data.sentiment_metrics.sentiment_variance > 0.6):
            risk_conditions.append("High sentiment disagreement")
        
        if not risk_conditions:
            return None
        
        title = f"Risk Warning - {event_data.ticker}"
        description = (
            f"Risk conditions detected for {event_data.ticker}: "
            f"{'; '.join(risk_conditions)}. "
            f"Exercise caution with trading decisions."
        )
        
        return self._create_alert(
            ticker=event_data.ticker,
            alert_type=AlertType.RISK_WARNING,
            alert_level=AlertLevel.MEDIUM,
            title=title,
            description=description,
            event_data=event_data,
            volatility_signal=volatility_signal,
            expires_minutes=30
        )
    
    def _create_alert(
        self,
        ticker: str,
        alert_type: AlertType,
        alert_level: AlertLevel,
        title: str,
        description: str,
        event_data: EventData,
        volatility_signal: VolatilitySignal,
        expires_minutes: int = 60
    ) -> TradingAlert:
        """Create a trading alert with all required information."""
        
        # Generate unique alert ID
        alert_content = f"{ticker}_{alert_type.value}_{int(datetime.utcnow().timestamp())}"
        alert_id = hashlib.md5(alert_content.encode()).hexdigest()[:12]
        
        # Extract top headlines
        top_headlines = []
        if event_data.articles:
            for article in event_data.articles[:5]:
                headline = article.get('title', '')
                if headline:
                    top_headlines.append(headline)
        
        # Calculate position size recommendation
        position_size = self._calculate_position_size_recommendation(volatility_signal)
        
        # Calculate stop loss and take profit suggestions
        stop_loss, take_profit = self._calculate_risk_management_levels(
            volatility_signal, event_data.sentiment_metrics
        )
        
        return TradingAlert(
            alert_id=alert_id,
            ticker=ticker,
            alert_type=alert_type,
            alert_level=alert_level,
            title=title,
            description=description,
            timestamp=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=expires_minutes),
            
            # Core metrics
            sentiment_score=event_data.sentiment_metrics.average_sentiment if event_data.sentiment_metrics else 0.0,
            sentiment_change=event_data.sentiment_metrics.sentiment_change if event_data.sentiment_metrics else 0.0,
            volatility_score=volatility_signal.strength,
            confidence=volatility_signal.confidence,
            
            # Supporting data
            article_count=event_data.event_volume,
            top_headlines=top_headlines,
            key_themes=event_data.themes[:5],
            
            # Trading recommendations
            recommended_action=volatility_signal.recommended_action,
            position_size_recommendation=position_size,
            stop_loss_suggestion=stop_loss,
            take_profit_suggestion=take_profit,
            
            # Metadata
            source_data={
                'event_data': {
                    'ticker': event_data.ticker,
                    'timestamp': event_data.timestamp.isoformat(),
                    'article_count': event_data.event_volume,
                    'relevance_score': event_data.relevance_score
                },
                'volatility_signal': {
                    'signal_type': volatility_signal.signal_type,
                    'strength': volatility_signal.strength,
                    'direction': volatility_signal.direction,
                    'confidence': volatility_signal.confidence,
                    'triggers': volatility_signal.triggers
                }
            }
        )
    
    def _calculate_position_size_recommendation(self, volatility_signal: VolatilitySignal) -> float:
        """Calculate recommended position size based on signal strength and confidence."""
        base_size = config.max_position_size
        
        # Adjust based on confidence
        confidence_multiplier = volatility_signal.confidence
        
        # Adjust based on signal strength
        strength_multiplier = min(volatility_signal.strength, 1.0)
        
        # Conservative approach: reduce size for high volatility
        if volatility_signal.signal_type == 'high_volatility':
            volatility_multiplier = 0.7
        elif volatility_signal.signal_type == 'medium_volatility':
            volatility_multiplier = 0.85
        else:
            volatility_multiplier = 1.0
        
        recommended_size = base_size * confidence_multiplier * strength_multiplier * volatility_multiplier
        
        # Ensure minimum and maximum bounds
        return max(100, min(recommended_size, config.max_position_size))
    
    def _calculate_risk_management_levels(
        self, 
        volatility_signal: VolatilitySignal, 
        sentiment_metrics
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        
        # Base stop loss and take profit from config
        base_stop_loss = config.stop_loss_percentage
        base_take_profit = config.take_profit_percentage
        
        # Adjust based on volatility
        if volatility_signal.strength > 0.8:
            # High volatility: wider stops
            stop_loss = base_stop_loss * 1.5
            take_profit = base_take_profit * 1.3
        elif volatility_signal.strength > 0.6:
            # Medium volatility: slightly wider stops
            stop_loss = base_stop_loss * 1.2
            take_profit = base_take_profit * 1.1
        else:
            # Low volatility: normal stops
            stop_loss = base_stop_loss
            take_profit = base_take_profit
        
        # Adjust based on confidence
        if volatility_signal.confidence < 0.5:
            # Low confidence: tighter stops
            stop_loss *= 0.8
            take_profit *= 0.9
        
        return stop_loss, take_profit
    
    def _add_alert(self, alert: TradingAlert):
        """Add alert to active alerts and history."""
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Maintain history size
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-500:]
        
        logger.info(f"Generated alert: {alert.title} (ID: {alert.alert_id})")
    
    def _is_in_cooldown(self, ticker: str) -> bool:
        """Check if ticker is in cooldown period."""
        if ticker not in self.cooldown_periods:
            return False
        
        return datetime.utcnow() < self.cooldown_periods[ticker]
    
    def _update_cooldown(self, ticker: str):
        """Update cooldown period for ticker."""
        self.cooldown_periods[ticker] = datetime.utcnow() + timedelta(
            minutes=config.cooldown_minutes
        )
    
    def get_active_alerts(
        self, 
        ticker: str = None, 
        alert_level: AlertLevel = None
    ) -> List[TradingAlert]:
        """Get active alerts, optionally filtered by ticker and/or level."""
        alerts = list(self.active_alerts.values())
        
        # Remove expired alerts
        current_time = datetime.utcnow()
        active_alerts = [alert for alert in alerts if alert.expires_at > current_time]
        
        # Update active alerts dict
        self.active_alerts = {alert.alert_id: alert for alert in active_alerts}
        
        # Apply filters
        if ticker:
            active_alerts = [alert for alert in active_alerts if alert.ticker == ticker]
        
        if alert_level:
            active_alerts = [alert for alert in active_alerts if alert.alert_level == alert_level]
        
        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert system status."""
        active_alerts = self.get_active_alerts()
        
        # Count by level
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.value] = sum(
                1 for alert in active_alerts if alert.alert_level == level
            )
        
        # Count by type
        type_counts = {}
        for alert_type in AlertType:
            type_counts[alert_type.value] = sum(
                1 for alert in active_alerts if alert.alert_type == alert_type
            )
        
        # Recent activity
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]
        
        return {
            "active_alerts_count": len(active_alerts),
            "alerts_by_level": level_counts,
            "alerts_by_type": type_counts,
            "recent_24h_count": len(recent_alerts),
            "total_alerts_generated": len(self.alert_history),
            "tickers_with_alerts": list(set(alert.ticker for alert in active_alerts)),
            "last_alert_time": (
                max(alert.timestamp for alert in active_alerts).isoformat()
                if active_alerts else None
            )
        }
