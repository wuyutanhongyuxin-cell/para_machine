"""
Thompson Sampling for Strategy Selection.

Implements multi-armed bandit algorithm using Beta distributions
to automatically discover and exploit the best performing strategy.

Mathematical Background:
- Prior: Beta(α, β) where α=β=1 gives uniform distribution
- Posterior: Beta(α + wins, β + losses)
- Selection: Sample from each posterior, choose highest sample

Key Features:
- Automatic exploration vs exploitation balance
- Handles non-stationary environments
- Memory efficient
- Persistence support
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("paradex_trader.learning.thompson")


@dataclass
class StrategyArm:
    """
    Single strategy arm for multi-armed bandit.

    Tracks Beta distribution parameters and performance metrics.
    """
    name: str
    alpha: float = 1.0          # Beta distribution α (successes + prior)
    beta: float = 1.0           # Beta distribution β (failures + prior)
    total_trials: int = 0
    total_reward: float = 0.0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    last_selected: float = 0.0
    last_updated: float = 0.0

    @property
    def estimated_probability(self) -> float:
        """Get estimated win probability (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """Get 95% confidence interval for win probability."""
        from scipy import stats
        dist = stats.beta(self.alpha, self.beta)
        return dist.ppf(0.025), dist.ppf(0.975)

    @property
    def variance(self) -> float:
        """Get variance of the Beta distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def avg_reward(self) -> float:
        """Get average reward per trial."""
        if self.total_trials == 0:
            return 0.0
        return self.total_reward / self.total_trials

    @property
    def win_rate(self) -> float:
        """Get actual win rate."""
        if self.total_trials == 0:
            return 0.0
        return self.wins / self.total_trials

    def sample(self) -> float:
        """Sample from Beta posterior distribution."""
        return np.random.beta(self.alpha, self.beta)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "total_trials": self.total_trials,
            "total_reward": self.total_reward,
            "total_pnl": self.total_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "last_selected": self.last_selected,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyArm":
        """Create from dictionary."""
        return cls(**data)


class ThompsonSampling:
    """
    Thompson Sampling multi-armed bandit for strategy selection.

    Uses Bayesian approach to balance exploration and exploitation:
    1. Maintains Beta posterior for each strategy's win probability
    2. Samples from posteriors to select strategy
    3. Updates posteriors based on trade outcomes
    4. Automatically shifts to best performing strategy

    Advantages over epsilon-greedy or UCB:
    - Probability matching (explores uncertain arms more)
    - Works well with small sample sizes
    - Naturally adapts exploration rate
    """

    def __init__(
        self,
        strategies: List[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        min_trials_per_strategy: int = 20,
        exploitation_threshold: float = 0.15,
        decay_factor: float = 0.995,
        use_decay: bool = True,
    ):
        """
        Initialize Thompson Sampling.

        Args:
            strategies: List of strategy names.
            prior_alpha: Prior α for Beta distribution.
            prior_beta: Prior β for Beta distribution.
            min_trials_per_strategy: Minimum trials before exploitation.
            exploitation_threshold: Probability gap to enter exploitation mode.
            decay_factor: Factor to decay old observations (for non-stationarity).
            use_decay: Whether to apply decay to handle changing conditions.
        """
        self.strategies = strategies
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.min_trials = min_trials_per_strategy
        self.exploitation_threshold = exploitation_threshold
        self.decay_factor = decay_factor
        self.use_decay = use_decay

        # Initialize arms
        self.arms: Dict[str, StrategyArm] = {
            name: StrategyArm(name=name, alpha=prior_alpha, beta=prior_beta)
            for name in strategies
        }

        # State tracking
        self._exploration_mode = True
        self._total_selections = 0
        self._selection_history: List[Tuple[float, str]] = []

        logger.info(
            f"ThompsonSampling initialized with {len(strategies)} strategies, "
            f"prior=Beta({prior_alpha}, {prior_beta}), min_trials={min_trials_per_strategy}"
        )

    def select_strategy(self) -> str:
        """
        Select a strategy using Thompson Sampling.

        In exploration mode: Ensures each strategy gets minimum trials.
        In exploitation mode: Samples from posteriors and picks highest.

        Returns:
            Selected strategy name.
        """
        now = time.time()

        # Exploration phase: ensure minimum trials
        if self._exploration_mode:
            for name, arm in self.arms.items():
                if arm.total_trials < self.min_trials:
                    logger.debug(f"Exploration: selecting {name} (trials={arm.total_trials})")
                    self._record_selection(name, now)
                    return name

        # Thompson Sampling: sample from each posterior
        samples = {}
        for name, arm in self.arms.items():
            samples[name] = arm.sample()

        # Select highest sample
        selected = max(samples, key=samples.get)

        self._record_selection(selected, now)

        logger.debug(
            f"Thompson selected: {selected} (sample={samples[selected]:.3f}, "
            f"est_prob={self.arms[selected].estimated_probability:.3f})"
        )

        return selected

    def _record_selection(self, strategy: str, timestamp: float) -> None:
        """Record a selection."""
        self._total_selections += 1
        self.arms[strategy].last_selected = timestamp
        self._selection_history.append((timestamp, strategy))

        # Keep last 1000 selections
        if len(self._selection_history) > 1000:
            self._selection_history = self._selection_history[-500:]

    def update(
        self,
        strategy: str,
        pnl: float,
        pnl_pct: float,
        is_win: bool,
    ) -> None:
        """
        Update posterior based on trade outcome.

        Args:
            strategy: Strategy that was used.
            pnl: Profit/loss amount.
            pnl_pct: Profit/loss percentage.
            is_win: Whether trade was profitable.
        """
        if strategy not in self.arms:
            logger.warning(f"Unknown strategy: {strategy}")
            return

        arm = self.arms[strategy]

        # Apply decay if enabled (for non-stationarity)
        if self.use_decay and arm.total_trials > 0:
            self._apply_decay(arm)

        # Update counts
        arm.total_trials += 1
        arm.total_pnl += pnl

        if is_win:
            arm.wins += 1
        else:
            arm.losses += 1

        # Convert PnL to reward signal for Beta update
        # Use sigmoid to normalize PnL percentage to [0, 1]
        normalized_reward = self._pnl_to_reward(pnl_pct)
        arm.total_reward += normalized_reward

        # Update Beta parameters
        if is_win:
            arm.alpha += 1.0
        else:
            arm.beta += 1.0

        arm.last_updated = time.time()

        # Check exploration mode
        self._check_exploration_mode()

        logger.debug(
            f"Updated {strategy}: α={arm.alpha:.1f}, β={arm.beta:.1f}, "
            f"est_prob={arm.estimated_probability:.3f}, win_rate={arm.win_rate:.3f}"
        )

    def _pnl_to_reward(self, pnl_pct: float) -> float:
        """
        Convert PnL percentage to reward signal using sigmoid.

        Maps PnL to [0, 1] range where:
        - Large losses → ~0
        - Breakeven → 0.5
        - Large profits → ~1
        """
        # Scale: 1% PnL = significant
        return 1 / (1 + np.exp(-pnl_pct * 100))

    def _apply_decay(self, arm: StrategyArm) -> None:
        """
        Apply decay to arm parameters.

        This helps adapt to non-stationary environments by
        gradually "forgetting" old observations.
        """
        # Decay towards prior
        excess_alpha = arm.alpha - self.prior_alpha
        excess_beta = arm.beta - self.prior_beta

        arm.alpha = self.prior_alpha + excess_alpha * self.decay_factor
        arm.beta = self.prior_beta + excess_beta * self.decay_factor

    def _check_exploration_mode(self) -> None:
        """Check and update exploration/exploitation mode."""
        # All strategies need minimum trials
        all_explored = all(
            arm.total_trials >= self.min_trials
            for arm in self.arms.values()
        )

        if not all_explored:
            self._exploration_mode = True
            return

        # Check if there's a clear leader
        probs = self.get_probabilities()
        sorted_probs = sorted(probs.values(), reverse=True)

        if len(sorted_probs) >= 2:
            gap = sorted_probs[0] - sorted_probs[1]
            leader_prob = sorted_probs[0]

            # Enter exploitation if leader is clear and good
            if gap >= self.exploitation_threshold and leader_prob > 0.55:
                self._exploration_mode = False
                logger.info(
                    f"Entering exploitation mode. Leader prob: {leader_prob:.3f}, "
                    f"gap: {gap:.3f}"
                )
                return

        self._exploration_mode = True

    def get_probabilities(self) -> Dict[str, float]:
        """Get estimated probabilities for all strategies."""
        return {
            name: arm.estimated_probability
            for name, arm in self.arms.items()
        }

    def get_best_strategy(self) -> Tuple[str, float]:
        """
        Get the current best strategy.

        Returns:
            Tuple of (strategy_name, estimated_probability).
        """
        probs = self.get_probabilities()
        best = max(probs, key=probs.get)
        return best, probs[best]

    def is_exploring(self) -> bool:
        """Check if still in exploration mode."""
        return self._exploration_mode

    def get_selection_counts(self) -> Dict[str, int]:
        """Get selection count for each strategy."""
        return {name: arm.total_trials for name, arm in self.arms.items()}

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed statistics for all strategies."""
        stats = {}
        for name, arm in self.arms.items():
            ci_low, ci_high = arm.confidence_interval
            stats[name] = {
                "alpha": arm.alpha,
                "beta": arm.beta,
                "total_trials": arm.total_trials,
                "wins": arm.wins,
                "losses": arm.losses,
                "win_rate": arm.win_rate,
                "estimated_probability": arm.estimated_probability,
                "confidence_interval": (ci_low, ci_high),
                "variance": arm.variance,
                "total_pnl": arm.total_pnl,
                "avg_reward": arm.avg_reward,
            }
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        best_name, best_prob = self.get_best_strategy()
        return {
            "exploration_mode": self._exploration_mode,
            "total_selections": self._total_selections,
            "best_strategy": best_name,
            "best_probability": best_prob,
            "strategies": {
                name: {
                    "trials": arm.total_trials,
                    "win_rate": arm.win_rate,
                    "est_prob": arm.estimated_probability,
                    "pnl": arm.total_pnl,
                }
                for name, arm in self.arms.items()
            },
        }

    def save_state(self, filepath: str) -> None:
        """
        Save state to JSON file.

        Args:
            filepath: Path to save file.
        """
        state = {
            "arms": {name: arm.to_dict() for name, arm in self.arms.items()},
            "exploration_mode": self._exploration_mode,
            "total_selections": self._total_selections,
            "config": {
                "prior_alpha": self.prior_alpha,
                "prior_beta": self.prior_beta,
                "min_trials": self.min_trials,
                "exploitation_threshold": self.exploitation_threshold,
                "decay_factor": self.decay_factor,
                "use_decay": self.use_decay,
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Thompson state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """
        Load state from JSON file.

        Args:
            filepath: Path to load from.

        Returns:
            True if loaded successfully.
        """
        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            for name, data in state.get("arms", {}).items():
                if name in self.arms:
                    self.arms[name] = StrategyArm.from_dict(data)

            self._exploration_mode = state.get("exploration_mode", True)
            self._total_selections = state.get("total_selections", 0)

            logger.info(f"Thompson state loaded from {filepath}")
            return True

        except FileNotFoundError:
            logger.info(f"No saved state found at {filepath}")
            return False
        except Exception as e:
            logger.error(f"Error loading Thompson state: {e}")
            return False

    def reset(self) -> None:
        """Reset all arms to prior."""
        for arm in self.arms.values():
            arm.alpha = self.prior_alpha
            arm.beta = self.prior_beta
            arm.total_trials = 0
            arm.total_reward = 0.0
            arm.total_pnl = 0.0
            arm.wins = 0
            arm.losses = 0

        self._exploration_mode = True
        self._total_selections = 0
        self._selection_history.clear()

        logger.info("Thompson Sampling reset")

    def add_strategy(self, name: str) -> None:
        """Add a new strategy arm."""
        if name not in self.arms:
            self.arms[name] = StrategyArm(
                name=name,
                alpha=self.prior_alpha,
                beta=self.prior_beta,
            )
            self.strategies.append(name)
            self._exploration_mode = True  # Need to explore new strategy
            logger.info(f"Added new strategy: {name}")

    def remove_strategy(self, name: str) -> None:
        """Remove a strategy arm."""
        if name in self.arms:
            del self.arms[name]
            self.strategies.remove(name)
            logger.info(f"Removed strategy: {name}")
