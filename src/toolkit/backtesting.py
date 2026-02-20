"""
Anti-Overfitting Backtest Framework

Based on:
- López de Prado: Backtesting is NOT a research tool, it's a reality check
- Miller & Davidow: Understanding break-even % and edge validation

Key Principles:
1. Backtests are for VALIDATION, not discovery
2. Multiple testing bias is real and significant
3. Use deflated metrics to account for strategy search
4. Out-of-sample is SACRED - touch it only once
5. Track all variations you've tried (research audit trail)

"A backtest is not a research tool. It is a reality check."
- Marcos López de Prado
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats
import json
import hashlib


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    deflated_sharpe: float  # Adjusted for multiple testing
    max_drawdown: float
    win_rate: float
    n_trades: int
    n_trials: int  # How many variations were tested
    is_significant: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'strategy_name': self.strategy_name,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'deflated_sharpe': self.deflated_sharpe,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'n_trades': self.n_trades,
            'n_trials': self.n_trials,
            'is_significant': self.is_significant,
            'timestamp': self.timestamp
        }


@dataclass
class ResearchAudit:
    """
    Audit trail for strategy research.

    López de Prado emphasizes tracking ALL variations tested,
    not just the final "winning" strategy. This prevents
    publication bias and enables proper deflation.
    """
    strategy_variations: List[Dict] = field(default_factory=list)
    total_trials: int = 0
    best_sharpe: float = 0.0
    best_strategy: str = ""

    def log_trial(
        self,
        strategy_name: str,
        params: Dict,
        sharpe: float,
        notes: str = ""
    ):
        """Log a strategy variation trial."""
        self.total_trials += 1

        trial = {
            'trial_id': self.total_trials,
            'strategy_name': strategy_name,
            'params': params,
            'sharpe': sharpe,
            'timestamp': datetime.now().isoformat(),
            'notes': notes,
            'hash': self._hash_params(params)
        }
        self.strategy_variations.append(trial)

        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_strategy = strategy_name

    def _hash_params(self, params: Dict) -> str:
        """Create unique hash for parameter combination."""
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]

    def get_deflated_best(self) -> float:
        """Get deflated Sharpe for best strategy."""
        if self.total_trials == 0:
            return self.best_sharpe
        return compute_deflated_sharpe(self.best_sharpe, self.total_trials)

    def summary(self) -> str:
        """Print research audit summary."""
        lines = [
            f"\n{'='*60}",
            "RESEARCH AUDIT TRAIL",
            f"{'='*60}",
            f"  Total trials: {self.total_trials}",
            f"  Best strategy: {self.best_strategy}",
            f"  Best Sharpe: {self.best_sharpe:.2f}",
            f"  Deflated Sharpe: {self.get_deflated_best():.2f}",
            f"{'='*60}"
        ]
        return '\n'.join(lines)


class Backtester:
    """
    Anti-overfitting backtest framework.

    This framework enforces best practices:
    1. Tracks all strategy variations (audit trail)
    2. Computes deflated metrics automatically
    3. Warns about common pitfalls
    4. Separates validation from discovery

    Parameters
    ----------
    initial_capital : float
        Starting capital
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculation
    significance_level : float
        Deflated Sharpe threshold for "significant" strategy
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.02,
        significance_level: float = 0.05
    ):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.significance_level = significance_level
        self.audit = ResearchAudit()

    def run_backtest(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        bet_sizes: Optional[np.ndarray] = None,
        strategy_name: str = "strategy",
        params: Optional[Dict] = None,
        log_trial: bool = True
    ) -> BacktestResult:
        """
        Run backtest on predictions.

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions (probabilities or values)
        actuals : np.ndarray
            Actual outcomes
        bet_sizes : np.ndarray, optional
            Position sizes (default: flat betting)
        strategy_name : str
            Name for audit trail
        params : dict, optional
            Strategy parameters for logging
        log_trial : bool
            Whether to log to audit trail

        Returns
        -------
        BacktestResult
            Backtest results with deflated metrics
        """
        n = len(predictions)
        if bet_sizes is None:
            bet_sizes = np.ones(n)

        # Calculate returns
        returns = self._calculate_returns(predictions, actuals, bet_sizes)

        # Core metrics
        total_return = np.sum(returns)
        sharpe = self._calculate_sharpe(returns)
        max_dd = self._calculate_max_drawdown(returns)
        win_rate = np.mean(returns > 0)

        # Log trial if enabled
        if log_trial:
            self.audit.log_trial(
                strategy_name=strategy_name,
                params=params or {},
                sharpe=sharpe
            )

        # Deflated Sharpe accounting for all trials
        deflated = compute_deflated_sharpe(sharpe, self.audit.total_trials)
        is_sig = deflated > (1 - self.significance_level)

        # Build equity curve
        equity_curve = self.initial_capital * (1 + np.cumsum(returns))

        result = BacktestResult(
            strategy_name=strategy_name,
            total_return=total_return,
            sharpe_ratio=sharpe,
            deflated_sharpe=deflated,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_trades=n,
            n_trials=self.audit.total_trials,
            is_significant=is_sig,
            equity_curve=equity_curve,
            metrics={
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'skew': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'calmar_ratio': total_return / max_dd if max_dd > 0 else 0,
                'profit_factor': np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0]))
                                if np.sum(returns[returns < 0]) != 0 else np.inf
            }
        )

        return result

    def _calculate_returns(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        bet_sizes: np.ndarray
    ) -> np.ndarray:
        """
        Calculate returns from predictions.

        For DFS/betting: return = bet_size * (actual - prediction) / prediction
        Or simplified: return = bet_size * (outcome == prediction)
        """
        # Simple P/L based on prediction accuracy
        # This can be customized for specific use cases
        errors = actuals - predictions
        returns = bet_sizes * np.sign(errors) * np.abs(errors) / (np.abs(predictions) + 1e-8)

        return returns

    def _calculate_sharpe(
        self,
        returns: np.ndarray,
        periods_per_year: int = 52  # Weekly
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        # Annualize
        annual_return = mean_return * periods_per_year
        annual_std = std_return * np.sqrt(periods_per_year)

        sharpe = (annual_return - self.risk_free_rate) / annual_std
        return sharpe

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cum_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = running_max - cum_returns

        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    def walk_forward_backtest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        dates: pd.Series,
        n_splits: int = 5,
        strategy_name: str = "walk_forward",
        params: Optional[Dict] = None
    ) -> BacktestResult:
        """
        Walk-forward backtest (most realistic).

        At each point in time:
        1. Train on all available history
        2. Predict next period
        3. Record actual outcome
        4. Move forward

        This mimics real-world deployment.
        """
        from .data_structures import WalkForwardCV

        cv = WalkForwardCV(n_splits=n_splits)
        all_predictions = []
        all_actuals = []

        for train_idx, test_idx in cv.split(X, y, dates):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Clone and fit model
            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            # Predict
            preds = model_clone.predict(X_test)
            all_predictions.extend(preds)
            all_actuals.extend(y_test.values)

        return self.run_backtest(
            predictions=np.array(all_predictions),
            actuals=np.array(all_actuals),
            strategy_name=strategy_name,
            params=params
        )

    def combinatorial_backtest(
        self,
        base_strategy: Callable,
        param_grid: Dict[str, List],
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        top_n: int = 3
    ) -> List[BacktestResult]:
        """
        Test all parameter combinations (López de Prado).

        Rather than cherry-picking "best" parameters, this tests
        all combinations and properly deflates the results.

        Parameters
        ----------
        base_strategy : callable
            Function that takes (X, y, **params) and returns predictions
        param_grid : dict
            Parameter name -> list of values to try
        X, y, dates : data
            Features, target, and dates
        top_n : int
            Number of top strategies to return

        Returns
        -------
        List[BacktestResult]
            Top N results, properly deflated
        """
        from itertools import product

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(product(*param_values))

        print(f"\n{'='*60}")
        print(f"COMBINATORIAL BACKTEST")
        print(f"{'='*60}")
        print(f"  Parameters: {param_names}")
        print(f"  Total combinations: {len(all_combos)}")
        print(f"{'='*60}\n")

        results = []
        for i, combo in enumerate(all_combos):
            params = dict(zip(param_names, combo))

            # Get predictions from strategy
            predictions = base_strategy(X, y, **params)

            # Run backtest
            result = self.run_backtest(
                predictions=predictions,
                actuals=y.values,
                strategy_name=f"combo_{i}",
                params=params,
                log_trial=True
            )
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Tested {i + 1}/{len(all_combos)} combinations...")

        # Sort by deflated Sharpe (not raw Sharpe!)
        results.sort(key=lambda x: x.deflated_sharpe, reverse=True)

        print(f"\n  {'='*50}")
        print(f"  TOP {top_n} RESULTS (by deflated Sharpe)")
        print(f"  {'='*50}")
        for r in results[:top_n]:
            sig_marker = "✓" if r.is_significant else "✗"
            print(f"  {sig_marker} Sharpe: {r.sharpe_ratio:.2f} -> Deflated: {r.deflated_sharpe:.2f}")

        return results[:top_n]

    def print_report(self, result: BacktestResult):
        """Print formatted backtest report."""
        print(f"\n{'='*60}")
        print(f"BACKTEST REPORT: {result.strategy_name}")
        print(f"{'='*60}")
        print(f"  Trades: {result.n_trades}")
        print(f"  Total Return: {result.total_return:.2%}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")

        print(f"\n  {'='*50}")
        print(f"  RISK-ADJUSTED METRICS")
        print(f"  {'='*50}")
        print(f"  Raw Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Strategies Tested: {result.n_trials}")
        print(f"  Deflated Sharpe: {result.deflated_sharpe:.2f}")

        if result.is_significant:
            print(f"  ✓ SIGNIFICANT at {self.significance_level:.0%} level")
        else:
            print(f"  ✗ NOT SIGNIFICANT (may be due to multiple testing)")

        if result.n_trials > 10:
            print(f"\n  ⚠️  WARNING: {result.n_trials} strategy variations tested.")
            print(f"     Raw Sharpe may be misleading due to multiple testing bias.")

    def get_audit_summary(self) -> str:
        """Get research audit summary."""
        return self.audit.summary()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_deflated_sharpe(
    sharpe: float,
    n_trials: int,
    skew: float = 0,
    kurtosis: float = 3
) -> float:
    """
    Compute deflated Sharpe ratio.

    From López de Prado: adjusts Sharpe for multiple testing.

    If you test 100 strategies, finding one with Sharpe=2.0
    is actually expected by chance. This function deflates
    the Sharpe to account for the search process.

    Parameters
    ----------
    sharpe : float
        Observed Sharpe ratio
    n_trials : int
        Number of strategy variations tested
    skew : float
        Return distribution skewness
    kurtosis : float
        Return distribution kurtosis

    Returns
    -------
    float
        Deflated Sharpe (probability strategy is truly skillful)
    """
    if n_trials <= 1:
        return 1.0 if sharpe > 0 else 0.0

    # Expected maximum Sharpe under null hypothesis (no skill)
    # Using extreme value theory
    e_max_sharpe = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1/n_trials) + \
                   np.euler_gamma * stats.norm.ppf(1 - 1/(n_trials * np.e))

    # Standard error with skewness/kurtosis adjustment
    se = np.sqrt(
        (1 + 0.5 * sharpe**2 - skew * sharpe + (kurtosis - 3) / 4 * sharpe**2) /
        max(1, n_trials - 1)
    )

    # Probability Sharpe is truly above expected maximum
    if se == 0:
        return 1.0 if sharpe > e_max_sharpe else 0.0

    deflated = stats.norm.cdf((sharpe - e_max_sharpe) / se)

    return deflated


def calculate_break_even_pct(
    american_odds: int
) -> float:
    """
    Calculate break-even percentage from American odds.

    From Miller & Davidow: convert odds to break-even %
    to evaluate true edge.

    Parameters
    ----------
    american_odds : int
        American odds (e.g., -110, +150)

    Returns
    -------
    float
        Break-even winning percentage (0-1)
    """
    if american_odds < 0:
        # Favorite: BE% = |odds| / (|odds| + 100)
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        # Underdog: BE% = 100 / (odds + 100)
        return 100 / (american_odds + 100)


def calculate_edge(
    predicted_prob: float,
    american_odds: int
) -> float:
    """
    Calculate betting edge.

    Edge = Predicted Probability - Break-Even Probability

    A positive edge means you have an expected profit.
    """
    be_pct = calculate_break_even_pct(american_odds)
    return predicted_prob - be_pct


def kelly_criterion(
    predicted_prob: float,
    american_odds: int,
    kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly Criterion bet size.

    Full Kelly is often too aggressive; fractional Kelly
    (typically 0.25) is more robust.

    Parameters
    ----------
    predicted_prob : float
        Your estimated probability of winning
    american_odds : int
        American odds offered
    kelly_fraction : float
        Fraction of full Kelly to use (default 0.25)

    Returns
    -------
    float
        Recommended bet size as fraction of bankroll
    """
    # Convert to decimal odds
    if american_odds < 0:
        decimal_odds = 1 + 100 / abs(american_odds)
    else:
        decimal_odds = 1 + american_odds / 100

    # Kelly formula: f* = (bp - q) / b
    # where b = decimal_odds - 1, p = win prob, q = lose prob
    b = decimal_odds - 1
    p = predicted_prob
    q = 1 - p

    kelly = (b * p - q) / b

    # Apply fraction and ensure non-negative
    return max(0, kelly * kelly_fraction)
