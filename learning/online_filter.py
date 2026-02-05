"""
Online Learning Filter using River.

Implements real-time machine learning filter that:
1. Predicts probability of trade success based on features
2. Learns continuously from each trade outcome
3. Detects concept drift when market conditions change

Uses River library for memory-efficient online learning.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("paradex_trader.learning.online_filter")

# Try to import River
try:
    from river import (
        compose,
        linear_model,
        preprocessing,
        metrics,
        drift,
        ensemble,
        optim,
    )
    # Try to import Rolling - location varies by River version
    try:
        from river.utils import Rolling
        ROLLING_AVAILABLE = True
    except ImportError:
        try:
            Rolling = metrics.Rolling
            ROLLING_AVAILABLE = True
        except AttributeError:
            ROLLING_AVAILABLE = False
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    ROLLING_AVAILABLE = False
    logger.warning("River not installed. Online learning will be disabled.")


class SimpleRollingAccuracy:
    """Simple rolling accuracy tracker as fallback when River's Rolling is unavailable."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._results: List[bool] = []

    def update(self, y_true: bool, y_pred: bool) -> None:
        self._results.append(y_true == y_pred)
        if len(self._results) > self.window_size:
            self._results.pop(0)

    def get(self) -> float:
        if not self._results:
            return 0.0
        return sum(self._results) / len(self._results)


@dataclass
class PredictionResult:
    """Result of online model prediction."""
    probability: float
    should_trade: bool
    confidence: float
    model_ready: bool


class OnlineLearningFilter:
    """
    Online learning filter for signal quality prediction.

    Uses River for memory-efficient incremental learning:
    - StandardScaler: Online feature normalization
    - LogisticRegression: Binary classification
    - ADWIN: Drift detection

    The filter learns from each trade outcome and adapts to
    changing market conditions in real-time.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        l2_regularization: float = 0.01,
        min_samples: int = 20,
        trade_threshold: float = 0.45,
        drift_sensitivity: float = 0.002,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize online learning filter.

        Args:
            learning_rate: Learning rate for SGD optimizer.
            l2_regularization: L2 regularization strength.
            min_samples: Minimum samples before predictions are used.
            trade_threshold: Minimum probability to recommend trading.
            drift_sensitivity: ADWIN delta parameter for drift detection.
            feature_names: Optional list of expected feature names.
        """
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.min_samples = min_samples
        self.trade_threshold = trade_threshold
        self.drift_sensitivity = drift_sensitivity
        self.feature_names = feature_names or []

        self.enabled = RIVER_AVAILABLE
        self.samples_seen = 0
        self.drift_count = 0
        self.last_drift_at = 0

        if not self.enabled:
            logger.warning("Online learning disabled (River not available)")
            return

        # Initialize model pipeline
        self._init_model()

        # Initialize metrics
        self._init_metrics()

        # Feature importance tracking
        self._feature_importance: Dict[str, float] = {}

        logger.info(
            f"OnlineLearningFilter initialized: lr={learning_rate}, "
            f"l2={l2_regularization}, min_samples={min_samples}"
        )

    def _init_model(self) -> None:
        """Initialize the online learning model."""
        # Main model: StandardScaler + Logistic Regression
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(
                optimizer=optim.SGD(self.learning_rate),
                l2=self.l2_regularization,
            )
        )

        # Drift detector using ADWIN
        self.drift_detector = drift.ADWIN(delta=self.drift_sensitivity)

        # Secondary model for comparison (helps detect regime change)
        self.secondary_model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(
                optimizer=optim.SGD(self.learning_rate * 2),
                l2=self.l2_regularization * 0.5,
            )
        )

    def _init_metrics(self) -> None:
        """Initialize performance metrics."""
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1 = metrics.F1()
        self.auc = metrics.ROCAUC()

        # Rolling metrics (last 50 samples)
        # Use River's Rolling if available, otherwise use simple fallback
        if ROLLING_AVAILABLE:
            self.rolling_accuracy = Rolling(metrics.Accuracy(), window_size=50)
        else:
            self.rolling_accuracy = SimpleRollingAccuracy(window_size=50)

    def predict(self, features: Dict[str, float]) -> PredictionResult:
        """
        Predict probability of trade success.

        Args:
            features: Feature dictionary.

        Returns:
            PredictionResult with probability and recommendation.
        """
        if not self.enabled:
            return PredictionResult(
                probability=0.5,
                should_trade=True,
                confidence=0.0,
                model_ready=False,
            )

        # Not enough data yet
        if self.samples_seen < self.min_samples:
            return PredictionResult(
                probability=0.5,
                should_trade=True,
                confidence=0.0,
                model_ready=False,
            )

        # Get prediction probability
        try:
            prob_dict = self.model.predict_proba_one(features)
            if prob_dict:
                success_prob = prob_dict.get(True, prob_dict.get(1, 0.5))
            else:
                success_prob = 0.5
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            success_prob = 0.5

        # Calculate confidence based on variance in rolling accuracy
        confidence = self._calculate_confidence()

        # Trading recommendation
        should_trade = success_prob >= self.trade_threshold

        return PredictionResult(
            probability=success_prob,
            should_trade=should_trade,
            confidence=confidence,
            model_ready=True,
        )

    def _calculate_confidence(self) -> float:
        """Calculate confidence in model predictions."""
        if self.samples_seen < self.min_samples:
            return 0.0

        # Use rolling accuracy as confidence proxy
        rolling_acc = self.rolling_accuracy.get()

        # Confidence is higher when accuracy is consistently good
        if rolling_acc > 0.6:
            return min(1.0, (rolling_acc - 0.5) * 2)
        else:
            return max(0.0, rolling_acc - 0.4)

    def learn(self, features: Dict[str, float], outcome: bool) -> None:
        """
        Learn from a trade outcome.

        Args:
            features: Features at trade entry.
            outcome: True if profitable, False otherwise.
        """
        if not self.enabled:
            return

        self.samples_seen += 1

        # Test-then-train approach
        if self.samples_seen > self.min_samples:
            # Get prediction before learning
            prediction = self.model.predict_one(features)

            if prediction is not None:
                # Update metrics
                self._update_metrics(outcome, prediction)

                # Check for drift
                self._check_drift(outcome, prediction)

        # Learn from this sample
        try:
            self.model.learn_one(features, outcome)
            self.secondary_model.learn_one(features, outcome)
        except Exception as e:
            logger.error(f"Learning error: {e}")

        # Update feature importance
        self._update_feature_importance(features, outcome)

    def _update_metrics(self, outcome: bool, prediction: bool) -> None:
        """Update all metrics."""
        self.accuracy.update(outcome, prediction)
        self.precision.update(outcome, prediction)
        self.recall.update(outcome, prediction)
        self.f1.update(outcome, prediction)
        self.rolling_accuracy.update(outcome, prediction)

        # AUC needs probability
        try:
            prob = self.model.predict_proba_one({})
            if prob:
                self.auc.update(outcome, prob.get(True, 0.5))
        except Exception:
            pass

    def _check_drift(self, outcome: bool, prediction: bool) -> None:
        """Check for concept drift."""
        error = int(prediction != outcome)
        self.drift_detector.update(error)

        if self.drift_detector.drift_detected:
            self.drift_count += 1
            self.last_drift_at = self.samples_seen

            logger.warning(
                f"Concept drift detected! Total drifts: {self.drift_count}, "
                f"samples: {self.samples_seen}"
            )

            # Handle drift by slightly resetting learning rate
            self._handle_drift()

    def _handle_drift(self) -> None:
        """Handle detected concept drift."""
        # Option 1: Reset model (aggressive)
        # self._init_model()

        # Option 2: Increase learning rate temporarily (moderate)
        # The secondary model with higher lr will adapt faster
        # Main model will follow more slowly

        # Option 3: Just log and let natural adaptation happen (conservative)
        # This is the current approach - River's online learning will adapt
        pass

    def _update_feature_importance(
        self,
        features: Dict[str, float],
        outcome: bool
    ) -> None:
        """Track feature importance using correlation-based approach."""
        outcome_val = 1.0 if outcome else 0.0

        for name, value in features.items():
            if name not in self._feature_importance:
                self._feature_importance[name] = {
                    "sum_xy": 0.0,
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_x2": 0.0,
                    "n": 0,
                }

            stats = self._feature_importance[name]
            stats["sum_xy"] += value * outcome_val
            stats["sum_x"] += value
            stats["sum_y"] += outcome_val
            stats["sum_x2"] += value * value
            stats["n"] += 1

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores based on correlation."""
        importance = {}

        for name, stats in self._feature_importance.items():
            n = stats["n"]
            if n < 10:
                continue

            # Calculate correlation coefficient
            num = n * stats["sum_xy"] - stats["sum_x"] * stats["sum_y"]
            den_x = n * stats["sum_x2"] - stats["sum_x"] ** 2
            den_y = n * stats["sum_y"] - stats["sum_y"] ** 2

            if den_x > 0 and den_y > 0:
                corr = num / ((den_x ** 0.5) * (den_y ** 0.5))
                importance[name] = abs(corr)
            else:
                importance[name] = 0.0

        # Normalize
        if importance:
            max_imp = max(importance.values())
            if max_imp > 0:
                importance = {k: v / max_imp for k, v in importance.items()}

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "samples_seen": self.samples_seen,
            "model_ready": self.samples_seen >= self.min_samples,
            "accuracy": float(self.accuracy.get()) if self.samples_seen > 0 else 0,
            "precision": float(self.precision.get()) if self.samples_seen > 0 else 0,
            "recall": float(self.recall.get()) if self.samples_seen > 0 else 0,
            "f1": float(self.f1.get()) if self.samples_seen > 0 else 0,
            "rolling_accuracy": float(self.rolling_accuracy.get()) if self.samples_seen > 0 else 0,
            "drift_count": self.drift_count,
            "last_drift_at": self.last_drift_at,
            "confidence": self._calculate_confidence(),
        }

    def get_model_weights(self) -> Optional[Dict[str, float]]:
        """Get model weights (if available)."""
        if not self.enabled:
            return None

        try:
            # Access logistic regression weights
            lr = self.model.steps[-1]
            if hasattr(lr, "weights"):
                return dict(lr.weights)
        except Exception:
            pass

        return None

    def save_state(self, filepath: str) -> None:
        """Save model state to file."""
        if not self.enabled:
            return

        state = {
            "model": self.model,
            "secondary_model": self.secondary_model,
            "samples_seen": self.samples_seen,
            "drift_count": self.drift_count,
            "last_drift_at": self.last_drift_at,
            "feature_importance": self._feature_importance,
            "metrics": {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Online filter state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """Load model state from file."""
        if not self.enabled:
            return False

        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)

            self.model = state["model"]
            self.secondary_model = state["secondary_model"]
            self.samples_seen = state["samples_seen"]
            self.drift_count = state["drift_count"]
            self.last_drift_at = state["last_drift_at"]
            self._feature_importance = state.get("feature_importance", {})

            if "metrics" in state:
                self.accuracy = state["metrics"]["accuracy"]
                self.precision = state["metrics"]["precision"]
                self.recall = state["metrics"]["recall"]
                self.f1 = state["metrics"]["f1"]

            logger.info(f"Online filter state loaded from {filepath}")
            return True

        except FileNotFoundError:
            logger.info(f"No saved state found at {filepath}")
            return False
        except Exception as e:
            logger.error(f"Error loading online filter state: {e}")
            return False

    def reset(self) -> None:
        """Reset the filter to initial state."""
        if self.enabled:
            self._init_model()
            self._init_metrics()

        self.samples_seen = 0
        self.drift_count = 0
        self.last_drift_at = 0
        self._feature_importance.clear()

        logger.info("Online learning filter reset")
