"""
Loan Limit Increase Optimization — Credable Data Science Assessment
====================================================================
A Markov Decision Process (MDP) framework with Monte Carlo simulation,
demand forecasting, and risk-transition modeling to determine the optimal
loan limit increase policy under stochastic borrower behavior.

Author: Senior Data Scientist
Date:   2026-06-14
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 0. CONSTANTS & CONFIGURATION
# ============================================================================

PROFIT_PER_INCREASE = 40          # $ per accepted increase
ANNUAL_DISCOUNT_RATE = 0.19       # for NPV calculations
MONTHLY_DISCOUNT_RATE = (1 + ANNUAL_DISCOUNT_RATE) ** (1 / 12) - 1
MAX_INCREASES_PER_YEAR = 6
ELIGIBILITY_DAYS = 60             # must wait 60 days after last disbursement
MONTHS_PER_YEAR = 12
N_SIMULATIONS = 10               # Monte Carlo replications (reduced for quick run)

# Risk thresholds
RISK_CAPITAL_RATIO = 0.12         # regulatory capital requirement (12%)
MAX_PORTFOLIO_LOSS_RATE = 0.05    # max acceptable portfolio loss rate
EXPOSURE_LIMIT_MULTIPLIER = 1.5   # total exposure cannot exceed 1.5x initial


class RiskState(IntEnum):
    """Customer risk categories based on repayment performance."""
    DEFAULT = 0
    SUBPRIME = 1
    NEAR_PRIME = 2
    PRIME = 3


# Mapping for display
RISK_LABELS = {
    RiskState.PRIME: "Prime",
    RiskState.NEAR_PRIME: "Near-Prime",
    RiskState.SUBPRIME: "Subprime",
    RiskState.DEFAULT: "Default",
}


# ============================================================================
# 1. DATA LOADING & INITIAL RISK CLASSIFICATION
# ============================================================================

@dataclass
class Customer:
    """Represents a single customer and their loan profile."""
    customer_id: int
    initial_loan: float
    days_since_last_loan: int
    ontime_payments_pct: float
    num_increases_2023: int
    profit_contribution: float

    # Derived
    risk_state: RiskState = RiskState.PRIME

    def __post_init__(self):
        self.risk_state = self._classify_risk()

    def _classify_risk(self) -> RiskState:
        pct = self.ontime_payments_pct
        if pct >= 95:
            return RiskState.PRIME
        elif pct >= 90:
            return RiskState.NEAR_PRIME
        else:
            return RiskState.SUBPRIME


def load_customers(filepath: str) -> List[Customer]:
    """Load and parse the CSV dataset."""
    df = pd.read_csv(filepath, skip_blank_lines=True)
    customers = []
    for _, row in df.iterrows():
        c = Customer(
            customer_id=int(row["Customer ID"]),
            initial_loan=float(row["Initial Loan ($)"]),
            days_since_last_loan=int(row["Days Since Last Loan"]),
            ontime_payments_pct=float(row["On-time Payments (%)"]),
            num_increases_2023=int(row["No. of Increases in 2023"]),
            profit_contribution=float(row["Total Profit Contribution ($)"]),
        )
        customers.append(c)
    return customers


# ============================================================================
# 2. MARKOV CHAIN — RISK STATE TRANSITIONS
# ============================================================================

class RiskTransitionModel:
    """
    Markov chain model for customer risk-state transitions.

    Transition probabilities are a function of:
      - Current risk state
      - On-time payment percentage (signal of repayment quality)
      - Days since last loan (timing effect)
      - Initial loan amount (exposure effect)

    The transition matrix is computed per customer at each time step.
    """

    # Base transition probabilities:            To →  Def   Sub   NrPr  Prime
    #                                          From ↓
    BASE_TRANSITIONS = np.array([
        [1.00, 0.00, 0.00, 0.00],  # Default: absorbing
        [0.08, 0.60, 0.25, 0.07],  # Subprime
        [0.03, 0.12, 0.55, 0.30],  # Near-Prime
        [0.01, 0.04, 0.15, 0.80],  # Prime
    ])

    def __init__(self, customers: List[Customer]):
        self.customers = customers
        # Fit distribution parameters from data
        ontime = np.array([c.ontime_payments_pct for c in customers])
        self.ontime_mean = ontime.mean()     # ~90
        self.ontime_std = ontime.std()       # ~5.8
        loans = np.array([c.initial_loan for c in customers])
        self.loan_mean = loans.mean()
        self.loan_std = loans.std()
        days = np.array([c.days_since_last_loan for c in customers])
        self.days_mean = days.mean()

    def transition_matrix(self, customer: Customer) -> np.ndarray:
        """
        Compute a customer-specific 4×4 transition matrix.

        The base matrix is adjusted by:
          - On-time payment signal: better → better transitions
          - Loan amount signal: higher loan → slightly more risk of downgrade
        """
        T = self.BASE_TRANSITIONS.copy()

        # On-time payment adjustment factor (centered, std-normalized)
        payment_z = (customer.ontime_payments_pct - self.ontime_mean) / self.ontime_std

        # Higher payments → more likely to stay in good states
        # Scale the improvement factor: range roughly [-2, +2] for z-scores
        improvement = np.tanh(payment_z * 0.5)  # bounded [-0.46, 0.46]

        state = int(customer.risk_state)
        if state > 0:  # not default
            # Improve probability of staying in same or better state
            stay_same = T[state, state]
            move_up = T[state, min(state + 1, 3)]
            move_down = T[state, max(state - 1, 0)]

            # Adjust: more improvement → shift mass upward
            delta = improvement * 0.15
            T[state, state] = np.clip(stay_same + delta * 0.5, 0, 1)
            T[state, min(state + 1, 3)] = np.clip(move_up + delta * 0.5, 0, 1)
            T[state, max(state - 1, 0)] = np.clip(move_down - delta, 0, 1)

            # Loan-size effect: large loans carry marginally more downgrade risk
            loan_z = (customer.initial_loan - self.loan_mean) / self.loan_std
            loan_risk = np.tanh(loan_z * 0.3) * 0.05
            T[state, 0] = np.clip(T[state, 0] + max(loan_risk, 0), 0, 1)

        # Normalize rows
        for i in range(4):
            row_sum = T[i].sum()
            if row_sum > 0:
                T[i] /= row_sum
        return T

    def next_state(self, customer: Customer) -> RiskState:
        """Sample the next risk state given the customer's transition matrix."""
        T = self.transition_matrix(customer)
        probs = T[int(customer.risk_state)]
        return RiskState(np.random.choice(4, p=probs))


# ============================================================================
# 3. DEMAND FORECASTING MODEL
# ============================================================================

class DemandModel:
    """
    Predicts the probability that a customer accepts a loan limit increase offer.

    Factors:
      - Risk state: prime customers more likely to accept (better credit appetite)
      - Saturation: the more increases already received, the lower acceptance
      - Days since last loan: closer to eligibility threshold → higher need
      - Loan amount: higher initial loan → more likely to seek more credit
    """

    # Base acceptance probabilities by risk state
    BASE_ACCEPTANCE = {
        RiskState.PRIME: 0.65,
        RiskState.NEAR_PRIME: 0.50,
        RiskState.SUBPRIME: 0.35,
        RiskState.DEFAULT: 0.00,
    }

    # Saturation decay: acceptance drops as more increases are received
    SATURATION_DECAY = [1.0, 0.95, 0.85, 0.70, 0.50, 0.30, 0.15]  # index = n_increases

    def acceptance_probability(
        self,
        customer: Customer,
        num_increases_sofar: int,
        days_since_last: int,
        current_month: int,
    ) -> float:
        """
        Compute the probability a customer accepts a limit increase offer.

        Args:
            customer: Customer object
            num_increases_sofar: Number of increases already received this year
            days_since_last: Days since the last loan disbursement
            current_month: Current month of the simulation (1-12)
        """
        if customer.risk_state == RiskState.DEFAULT:
            return 0.0

        # Base rate from risk state
        p = self.BASE_ACCEPTANCE[customer.risk_state]

        # Saturation effect
        idx = min(num_increases_sofar, len(self.SATURATION_DECAY) - 1)
        p *= self.SATURATION_DECAY[idx]

        # Days-since-last-loan effect: sigmoid around eligibility threshold
        # Just past 60 days, acceptance is higher (recent need)
        days_factor = 0.7 + 0.3 * np.exp(-0.02 * max(days_since_last - ELIGIBILITY_DAYS, 0))
        p *= days_factor

        # Seasonality: slightly higher demand mid-year
        seasonal = 0.9 + 0.2 * np.sin(np.pi * (current_month - 3) / 6)
        p *= seasonal

        # Loan amount effect: larger loans correlate with higher credit appetite
        loan_z = (customer.initial_loan - 2000) / 1500  # approximate normalization
        loan_factor = 0.85 + 0.30 * np.clip(np.tanh(loan_z), 0, 1)
        p *= loan_factor

        return np.clip(p, 0.0, 1.0)


# ============================================================================
# 4. DEFAULT PROBABILITY MODEL
# ============================================================================

class DefaultModel:
    """
    Estimates the probability a customer defaults given their risk state
    and profile.
    """

    # Base annual default probabilities → converted to monthly
    BASE_DEFAULT_PROB = {
        RiskState.PRIME: 0.01 / 12,       # ~1% annual
        RiskState.NEAR_PRIME: 0.04 / 12,  # ~4% annual
        RiskState.SUBPRIME: 0.10 / 12,    # ~10% annual
        RiskState.DEFAULT: 1.00,
    }

    def default_probability(self, customer: Customer, num_increases: int) -> float:
        """
        Default probability increases with each additional increase
        (stretching repayment capacity) and depends on risk state.
        """
        if customer.risk_state == RiskState.DEFAULT:
            return 1.0

        base = self.BASE_DEFAULT_PROB[customer.risk_state]

        # Each additional increase adds marginal default risk
        # (leverage effect — more credit = higher default probability)
        leverage_factor = 1.0 + 0.08 * num_increases

        # Loan size effect: larger loans have slightly higher default risk
        loan_factor = 1.0 + 0.03 * (customer.initial_loan / 5000)

        prob = base * leverage_factor * loan_factor
        return np.clip(prob, 0.0, 0.50)


# ============================================================================
# 4.5 MACRO-ECONOMIC OVERLAY
# ============================================================================

@dataclass
class MacroOverlay:
    """
    Adjusts offer aggressiveness based on macro-economic conditions.

    Factors:
      - Inflation: higher inflation → reduce aggressiveness (default 3%)
      - Unemployment: higher unemployment → reduce aggressiveness (default 4%)
      - Interest rate: higher rates → better margins → increase aggressiveness (default 5%)
    """

    inflation: float = 0.03
    unemployment: float = 0.04
    interest_rate: float = 0.05

    # Historical norms for normalization
    INFLATION_NORM: float = 0.03   # 3% baseline
    UNEMPLOYMENT_NORM: float = 0.04  # 4% baseline
    INTEREST_NORM: float = 0.05     # 5% baseline

    def aggressiveness_multiplier(self) -> float:
        """Compute a scalar multiplier in [0.3, 1.5] for offer aggressiveness."""
        score = 1.0
        # Inflation penalizes when above normal (erodes real returns)
        score -= 0.5 * max(0, (self.inflation - self.INFLATION_NORM) / 0.05)
        # Unemployment penalizes when above normal (higher default risk)
        score -= 0.3 * max(0, (self.unemployment - self.UNEMPLOYMENT_NORM) / 0.05)
        # Interest rates help when above normal (wider NIM)
        score += 0.2 * max(0, (self.interest_rate - self.INTEREST_NORM) / 0.05)
        return np.clip(score, 0.3, 1.5)


# ============================================================================
# 4.6 CONTEXTUAL BANDIT RL AGENT
# ============================================================================

class ContextualBanditAgent:
    """
    Epsilon-greedy contextual bandit for online learning of optimal offer policy.

    Context features: (risk_state, num_increases, days_since_last, month)
    Arms: 0 = don't offer, 1 = offer
    Reward: profit from the customer minus expected default cost
    """

    def __init__(self, epsilon: float = 0.10, alpha: float = 0.05):
        self.epsilon = epsilon         # exploration rate
        self.alpha = alpha             # learning rate for EMA
        # State-value lookup: context_key → (n_pulls, value_offer, value_no_offer)
        self.q_table: Dict[Tuple, Tuple[int, float, float]] = {}

    def _context_key(self, risk_state: int, num_increases: int,
                     days_since_last: int, month: int) -> Tuple:
        days_bucket = min(days_since_last // 30, 12)  # bucket days into months
        return (risk_state, num_increases, days_bucket, month)

    def select_action(self, risk_state: int, num_increases: int,
                      days_since_last: int, month: int) -> int:
        """Epsilon-greedy action selection. Returns 0 (no offer) or 1 (offer)."""
        key = self._context_key(risk_state, num_increases, days_since_last, month)
        if key not in self.q_table:
            self.q_table[key] = (0, 0.0, 0.0)

        n, v_offer, v_no = self.q_table[key]

        # Cold-start: offer if we haven't tried it yet
        if n == 0:
            return 1  # explore offer

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)

        return 1 if v_offer > v_no else 0

    def update(self, risk_state: int, num_increases: int,
               days_since_last: int, month: int, action: int, reward: float):
        """Update Q-values with exponential moving average."""
        key = self._context_key(risk_state, num_increases, days_since_last, month)
        if key not in self.q_table:
            self.q_table[key] = (0, 0.0, 0.0)

        n, v_offer, v_no = self.q_table[key]
        n += 1

        if action == 1:
            v_offer += self.alpha * (reward - v_offer)
        else:
            v_no += self.alpha * (reward - v_no)

        self.q_table[key] = (n, v_offer, v_no)

    def get_policy(self) -> Dict[Tuple, int]:
        """Export learned policy as a state → action mapping."""
        policy = {}
        for key, (n, v_offer, v_no) in self.q_table.items():
            policy[key] = 1 if v_offer > v_no else 0
        return policy


# ============================================================================
# 5. LOAN LIFECYCLE SIMULATION (MONTE CARLO)
# ============================================================================

@dataclass
class SimulationResult:
    """Tracks simulation outcomes for a single customer over 12 months."""
    customer_id: int
    initial_risk: RiskState
    final_risk: RiskState
    increases_accepted: int
    increases_offered: int
    profit: float
    defaults: int  # 0 or 1
    npv_profit: float
    total_exposure: float
    monthly_path: List[Dict] = field(default_factory=list)


class LoanLifecycleSimulator:
    """
    Monte Carlo simulation of the loan lifecycle for the full customer base.

    Simulates month-by-month decisions, customer responses, and risk transitions
    to evaluate any given offer policy.
    """

    def __init__(
        self,
        customers: List[Customer],
        transition_model: RiskTransitionModel,
        demand_model: DemandModel,
        default_model: DefaultModel,
        macro_overlay: MacroOverlay = None,
        bandit_agent: ContextualBanditAgent = None,
    ):
        self.customers = customers
        self.transition_model = transition_model
        self.demand_model = demand_model
        self.default_model = default_model
        self.macro_overlay = macro_overlay or MacroOverlay()
        self.bandit_agent = bandit_agent

    def run_simulation(
        self,
        policy: str = "greedy",           # "greedy", "conservative", "mdp_optimal"
        threshold_params: Dict = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Run the full simulation for all customers over 12 months.

        Args:
            policy: The offer policy to evaluate
            threshold_params: Optional policy-specific parameters
            seed: Random seed for reproducibility

        Returns:
            DataFrame with per-customer simulation results
        """
        rng = np.random.RandomState(seed)
        results = []

        for customer in self.customers:
            res = self._simulate_customer(customer, policy, threshold_params, rng)
            results.append(res)

        df = pd.DataFrame([{
            "customer_id": r.customer_id,
            "initial_risk": RISK_LABELS[r.initial_risk],
            "final_risk": RISK_LABELS[r.final_risk],
            "increases_accepted": r.increases_accepted,
            "increases_offered": r.increases_offered,
            "profit": r.profit,
            "npv_profit": r.npv_profit,
            "defaults": r.defaults,
            "total_exposure": r.total_exposure,
        } for r in results])

        return df

    def _simulate_customer(
        self,
        customer: Customer,
        policy: str,
        params: Dict,
        rng: np.random.RandomState,
    ) -> SimulationResult:
        """Simulate one customer's journey over 12 months."""

        # Working copy of customer state (mutates during simulation)
        current_risk = customer.risk_state
        days_since_last = customer.days_since_last_loan
        num_increases = 0
        num_offered = 0
        total_profit = 0.0
        total_npv = 0.0
        has_defaulted = 0
        monthly_path = []

        # Initial exposure = initial loan amount
        exposure = customer.initial_loan

        # Create a mutable proxy with current risk
        class CustomerProxy:
            def __init__(self, base):
                self.customer_id = base.customer_id
                self.initial_loan = base.initial_loan
                self.ontime_payments_pct = base.ontime_payments_pct
                self.risk_state = current_risk
                self.initial_risk_state = current_risk  # for drift tracking

        proxy = CustomerProxy(customer)
        risk_drift_count = 0  # count of downgrades

        for month in range(1, MONTHS_PER_YEAR + 1):
            # Progress time
            days_since_last += 30  # approximate month length

            # Check eligibility
            eligible = days_since_last >= ELIGIBILITY_DAYS

            # Check max increases cap
            can_offer = num_increases < MAX_INCREASES_PER_YEAR

            # Policy decision: should we offer?
            # Inject context params for advanced policies
            decision_params = params or {}
            decision_params["days_since_last"] = days_since_last
            decision_params["month"] = month
            decision_params["risk_drift_count"] = risk_drift_count
            decision_params["initial_risk_state"] = proxy.initial_risk_state
            decision_params["macro_aggressiveness"] = self.macro_overlay.aggressiveness_multiplier()

            should_offer = self._policy_decision(
                proxy, num_increases, days_since_last, month, policy, decision_params
            )

            # Bandit feedback when policy chose "don't offer" (reward = 0)
            if policy == "bandit" and self.bandit_agent is not None and not should_offer:
                if eligible and can_offer:
                    self.bandit_agent.update(
                        int(proxy.risk_state), num_increases,
                        days_since_last, month, action=0, reward=0.0
                    )

            if eligible and can_offer and should_offer:
                num_offered += 1

                # Customer accepts?
                accept_prob = self.demand_model.acceptance_probability(
                    proxy, num_increases, days_since_last, month
                )
                accepted = rng.rand() < accept_prob

                # Bandit reward feedback (Rec #6)
                if policy == "bandit" and self.bandit_agent is not None:
                    # Reward: profit if accepted, 0 otherwise
                    inc_profit = PROFIT_PER_INCREASE if (num_increases + 1) > 2 else 0.0
                    reward = inc_profit if accepted else 0.0
                    self.bandit_agent.update(
                        int(proxy.risk_state), num_increases,
                        days_since_last, month, action=1, reward=reward
                    )
                # Bandit feedback when action was "don't offer" (handled in _policy_decision)

                if accepted:
                    num_increases += 1

                    # Reset days since last loan (loan was disbursed)
                    extension_days = rng.randint(60, 121)
                    days_since_last = 0  # just received new loan

                    # Profit: $40 per increase (problem statement)
                    incremental_profit = PROFIT_PER_INCREASE if num_increases > 2 else 0.0
                    total_profit += incremental_profit

                    # NPV discount
                    discount_factor = 1.0 / ((1 + MONTHLY_DISCOUNT_RATE) ** month)
                    total_npv += incremental_profit * discount_factor

                    # Update exposure (adding a marginal loan amount)
                    marginal_loan = customer.initial_loan * 0.15  # ~15% of initial
                    exposure += marginal_loan

            # Risk state transition (can happen regardless of offer)
            prev_risk = proxy.risk_state
            proxy.risk_state = self.transition_model.next_state(
                CustomerProxy(customer) if current_risk != proxy.risk_state else proxy
            )

            # Track risk drift (downgrade detected)
            if int(proxy.risk_state) < int(prev_risk) and proxy.risk_state != RiskState.DEFAULT:
                risk_drift_count += 1

            # Update for next iteration
            current_risk = proxy.risk_state

            # Check default
            default_prob = self.default_model.default_probability(proxy, num_increases)
            if rng.rand() < default_prob:
                has_defaulted = 1
                default_loss = exposure * 0.30  # 30% loss given default
                total_profit -= default_loss
                discount_factor = 1.0 / ((1 + MONTHLY_DISCOUNT_RATE) ** month)
                total_npv -= default_loss * discount_factor
                proxy.risk_state = RiskState.DEFAULT
                current_risk = RiskState.DEFAULT
                break  # customer exits after default

            monthly_path.append({
                "month": month,
                "risk_state": RISK_LABELS[current_risk],
                "offered": should_offer and eligible and can_offer,
                "accepted": False,  # simplified
                "days_since_last": days_since_last,
                "num_increases": num_increases,
            })

        return SimulationResult(
            customer_id=customer.customer_id,
            initial_risk=customer.risk_state,
            final_risk=current_risk,
            increases_accepted=num_increases,
            increases_offered=num_offered,
            profit=total_profit,
            npv_profit=total_npv,
            defaults=has_defaulted,
            total_exposure=exposure,
            monthly_path=monthly_path,
        )

    def _policy_decision(
        self,
        customer,
        num_increases: int,
        days_since_last: int,
        month: int,
        policy: str,
        params: Dict,
    ) -> bool:
        """
        Decide whether to offer a limit increase based on the policy.

        Policies:
          - "greedy":             Always offer if eligible
          - "conservative":       Offer only to Prime customers with low increases
          - "risk_aware":         Offer based on risk-adjusted expected value
          - "mdp_optimal":        Use pre-computed MDP policy table
          - "segmented":          Risk-based caps per segment (Rec #2)
          - "timing_optimized":   60-90 day optimal offer window (Rec #3)
          - "risk_drift_aware":   Reduce offers when risk downgrades detected (Rec #4)
          - "macro_aware":        Risk-aware with macro-economic overlay (Rec #5)
          - "bandit":             Contextual bandit RL agent (Rec #6)
        """
        if customer.risk_state == RiskState.DEFAULT:
            return False

        # ── Recommendation #2: SEGMENTATION ──
        if policy == "segmented":
            caps = {
                RiskState.PRIME: 5,
                RiskState.NEAR_PRIME: 3,
                RiskState.SUBPRIME: 2,
            }
            max_for_risk = caps.get(customer.risk_state, 0)
            return num_increases < max_for_risk

        # ── Recommendation #3: TIMING OPTIMIZATION ──
        if policy == "timing_optimized":
            # Only offer within the 60-90 day optimal window
            if days_since_last < 60 or days_since_last > 90:
                return False
            # Fall through to risk-aware for within-window decisions
            return self._risk_aware_offer(customer, num_increases, days_since_last, month)

        # ── Recommendation #4: RISK DRIFT MONITORING ──
        if policy == "risk_drift_aware":
            drift_count = params.get("risk_drift_count", 0)
            initial_risk = params.get("initial_risk_state", customer.risk_state)

            # Base caps from segmentation
            caps = {
                RiskState.PRIME: 5,
                RiskState.NEAR_PRIME: 3,
                RiskState.SUBPRIME: 2,
            }
            # Reduce cap by 1 for each risk downgrade detected
            base_cap = caps.get(initial_risk, 2)
            adjusted_cap = max(0, base_cap - drift_count)
            if num_increases >= adjusted_cap:
                return False
            # Also require passing risk-aware check
            return self._risk_aware_offer(customer, num_increases, days_since_last, month)

        # ── Recommendation #5: MACRO-AWARE ──
        if policy == "macro_aware":
            if not self._risk_aware_offer(customer, num_increases, days_since_last, month):
                return False
            # Further gate: macro aggressiveness threshold
            macro_agg = params.get("macro_aggressiveness", 1.0)
            if macro_agg < 0.6:
                return False  # recession regime — stop offering
            if macro_agg < 0.85:
                # Tight regime — only offer to Prime
                if customer.risk_state != RiskState.PRIME:
                    return False
                return num_increases <= 3
            return True  # normal/expansion regime

        # ── Recommendation #6: CONTEXTUAL BANDIT ──
        if policy == "bandit":
            if self.bandit_agent is None:
                return False  # no agent, default to no-offer
            action = self.bandit_agent.select_action(
                int(customer.risk_state), num_increases, days_since_last, month
            )
            return action == 1

        # ── EXISTING POLICIES ──
        if policy == "greedy":
            return True

        if policy == "conservative":
            if customer.risk_state != RiskState.PRIME:
                return False
            return num_increases <= 3

        if policy == "risk_aware":
            return self._risk_aware_offer(customer, num_increases, days_since_last, month)

        if policy == "mdp_optimal":
            if params and "policy_table" in params:
                state_key = (int(customer.risk_state), num_increases, min(month, 12))
                return params["policy_table"].get(state_key, False)
            return False

        return False

    def _risk_aware_offer(self, customer, num_increases: int,
                          days_since_last: int, month: int) -> bool:
        """Shared risk-aware logic: offer only if expected profit > expected loss."""
        accept_prob = self.demand_model.acceptance_probability(
            customer, num_increases, days_since_last, month
        )
        default_prob = self.default_model.default_probability(
            customer, num_increases + 1
        )
        expected_profit = accept_prob * PROFIT_PER_INCREASE
        expected_loss = default_prob * customer.initial_loan * 0.30
        return expected_profit > expected_loss


# ============================================================================
# 6. MDP OPTIMIZATION — VALUE ITERATION
# ============================================================================

class MDPOptimizer:
    """
    Markov Decision Process optimizer using Value Iteration.

    State space: (risk_state, num_increases, month)
    Action space: {offer, don't_offer}
    Reward: Expected profit - expected default loss
    """

    def __init__(
        self,
        transition_model: RiskTransitionModel,
        demand_model: DemandModel,
        default_model: DefaultModel,
        gamma: float = 1.0 / (1 + MONTHLY_DISCOUNT_RATE),
        theta: float = 1e-6,
    ):
        self.transition_model = transition_model
        self.demand_model = demand_model
        self.default_model = default_model
        self.gamma = gamma
        self.theta = theta

        # Value function and policy
        self.V: Dict[Tuple, float] = {}
        self.policy: Dict[Tuple, int] = {}  # 0 = don't offer, 1 = offer

    def _state_to_risk_state(self, state_idx: int) -> RiskState:
        return RiskState(min(state_idx, 3))

    def _get_all_states(self) -> List[Tuple[int, int, int]]:
        """Enumerate all possible states."""
        states = []
        for risk in range(1, 4):  # Prime, Near-Prime, Subprime (skip Default)
            for n_inc in range(0, MAX_INCREASES_PER_YEAR + 1):
                for month in range(1, MONTHS_PER_YEAR + 1):
                    states.append((risk, n_inc, month))
        # Terminal states: default at any stage
        for n_inc in range(0, MAX_INCREASES_PER_YEAR + 1):
            for month in range(1, MONTHS_PER_YEAR + 1):
                states.append((0, n_inc, month))  # Default state
        return states

    def _get_possible_actions(self, state: Tuple[int, int, int]) -> List[int]:
        """Actions: 0 = don't offer, 1 = offer."""
        risk, n_inc, month = state
        if risk == 0 or n_inc >= MAX_INCREASES_PER_YEAR or month > 12:
            return [0]  # can only "not offer" if defaulted or maxed out
        return [0, 1]

    def _expected_reward(
        self,
        state: Tuple[int, int, int],
        action: int,
        avg_loan: float = 2500,
    ) -> float:
        """
        Compute expected immediate reward for taking action in state.

        reward = P(accept) × profit_gain - P(default) × loss_given_default
        """
        risk, n_inc, month = state
        if risk == 0:
            return 0.0  # default state, no reward

        if action == 0:
            return 0.0  # no offer, no reward

        # Action = offer
        # Use average customer characteristics for demand and default
        rs = self._state_to_risk_state(risk)

        # Demand: probability of acceptance (simplified — using state-based estimates)
        accept_prob = self.demand_model.BASE_ACCEPTANCE[rs]
        idx = min(n_inc, len(self.demand_model.SATURATION_DECAY) - 1)
        accept_prob *= self.demand_model.SATURATION_DECAY[idx]

        # Default probability
        default_prob = self.default_model.BASE_DEFAULT_PROB[rs]
        default_prob *= (1.0 + 0.08 * (n_inc + 1))

        # Profit: $40 if this is the 3rd or later increase
        incremental_profit = PROFIT_PER_INCREASE if (n_inc + 1) > 2 else 0.0

        # Expected loss from default
        expected_loss = default_prob * avg_loan * 0.30

        # Expected reward
        reward = accept_prob * incremental_profit - expected_loss
        # Allow negative rewards so MDP learns when to avoid offering

        return reward

    def _transition_probs(
        self,
        state: Tuple[int, int, int],
        action: int,
    ) -> List[Tuple[Tuple[int, int, int], float]]:
        """
        Compute transition probabilities to next states.

        Returns list of (next_state, probability) pairs.
        """
        risk, n_inc, month = state

        if month >= 12 or risk == 0:
            return [((risk, n_inc, min(month + 1, 12)), 1.0)]

        next_month = month + 1

        # Risk transition probabilities (from base matrix)
        T = self.transition_model.BASE_TRANSITIONS
        trans_probs = T[risk]  # probability distribution over next risk states

        results = []

        if action == 0:
            # Don't offer: stays in same n_inc, transitions on risk
            for next_risk, prob in enumerate(trans_probs):
                if prob > 0:
                    results.append(((next_risk, n_inc, next_month), prob))

        else:
            # Offer: with some probability customer accepts
            rs = self._state_to_risk_state(risk)
            accept_prob = self.demand_model.BASE_ACCEPTANCE[rs]
            idx = min(n_inc, len(self.demand_model.SATURATION_DECAY) - 1)
            accept_prob *= self.demand_model.SATURATION_DECAY[idx]

            # If accepted: n_inc increases
            for next_risk, prob in enumerate(trans_probs):
                if prob > 0:
                    # Accepted path
                    results.append((
                        (next_risk, min(n_inc + 1, MAX_INCREASES_PER_YEAR), next_month),
                        prob * accept_prob,
                    ))
                    # Not accepted path
                    results.append((
                        (next_risk, n_inc, next_month),
                        prob * (1 - accept_prob),
                    ))

        return results

    def solve(self, max_iterations: int = 10000) -> Dict[Tuple, int]:
        """
        Run value iteration to find the optimal policy.

        Returns policy dictionary: state → action (0 or 1)
        """
        states = self._get_all_states()

        # Initialize value function
        for s in states:
            self.V[s] = 0.0

        for i in range(max_iterations):
            delta = 0.0

            for s in states:
                risk, n_inc, month = s

                # Terminal states (month 12 or default)
                if month >= 12 or risk == 0:
                    continue

                old_v = self.V[s]
                best_v = float("-inf")
                best_a = 0

                for a in self._get_possible_actions(s):
                    reward = self._expected_reward(s, a)
                    future_v = 0.0

                    for next_s, prob in self._transition_probs(s, a):
                        if next_s in self.V:
                            future_v += prob * self.V[next_s]

                    v = reward + self.gamma * future_v
                    if v > best_v:
                        best_v = v
                        best_a = a

                self.V[s] = best_v
                self.policy[s] = best_a
                delta = max(delta, abs(old_v - self.V[s]))

            if delta < self.theta:
                break

        print(f"  Value iteration converged after {i + 1} iterations (delta={delta:.2e})")
        return self.policy


# ============================================================================
# 7. MAIN ANALYSIS PIPELINE
# ============================================================================

def run_analysis():
    print("=" * 70)
    print("LOAN LIMIT INCREASE OPTIMIZATION — FULL ANALYSIS")
    print("=" * 70)

    # ---- Load Data ----
    print("\n[1] Loading dataset...")
    filepath = "/Users/satish.sahu/work/credable/loan_limit_increases.csv"
    customers = load_customers(filepath)
    print(f"  Loaded {len(customers)} customers")

    # ---- Data Summary ----
    print("\n[2] Data Summary & Risk Distribution...")
    risk_counts = defaultdict(int)
    for c in customers:
        risk_counts[RISK_LABELS[c.risk_state]] += 1
    for label, count in sorted(risk_counts.items()):
        print(f"  {label}: {count} ({100*count/len(customers):.1f}%)")

    ontime = np.array([c.ontime_payments_pct for c in customers])
    loans = np.array([c.initial_loan for c in customers])
    print(f"\n  On-time Payments: mean={ontime.mean():.2f}%, std={ontime.std():.2f}%")
    print(f"  Initial Loans: mean=${loans.mean():.0f}, std=${loans.std():.0f}")
    print(f"  Profit per increase observed: $40 (from 3rd increase onward)")

    # ---- Initialize Models ----
    print("\n[3] Initializing models...")
    transition_model = RiskTransitionModel(customers)
    demand_model = DemandModel()
    default_model = DefaultModel()
    macro_overlay = MacroOverlay(inflation=0.035, unemployment=0.042, interest_rate=0.055)
    bandit_agent = ContextualBanditAgent(epsilon=0.10, alpha=0.05)
    simulator = LoanLifecycleSimulator(
        customers, transition_model, demand_model, default_model,
        macro_overlay=macro_overlay, bandit_agent=bandit_agent,
    )
    print("  Markov chain, demand, default models, macro overlay, and bandit agent initialized.")

    # ---- Markov Chain Analysis ----
    print("\n[4] Markov Chain Transition Matrix (Base)...")
    labels = ["Default", "Subprime", "Near-Prime", "Prime"]
    T = transition_model.BASE_TRANSITIONS
    print(f"  {'From/To':<12} {'Default':>8} {'Subprime':>10} {'Near-Prime':>12} {'Prime':>8}")
    for i, label in enumerate(labels):
        row = "  " + f"{label:<12}"
        for j in range(4):
            row += f" {T[i,j]:>8.2f}"
        print(row)

    # Steady-state distribution
    print("\n  Computing steady-state distribution...")
    eigvals, eigvecs = np.linalg.eig(T.T)
    steady = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
    steady = steady / steady.sum()
    print("  Steady-state distribution:")
    for i, label in enumerate(labels):
        print(f"    {label}: {steady[i]:.3f} ({100*steady[i]:.1f}%)")

    # ---- MDP Optimization ----
    print("\n[5] Solving MDP via Value Iteration...")
    mdp = MDPOptimizer(transition_model, demand_model, default_model)
    policy = mdp.solve()

    # Summarize optimal policy
    print("\n  Optimal Policy Summary:")
    print(f"  {'Risk State':<12} {'Increases':>10} {'Month':>8} {'Action':>8}")
    print(f"  {'-'*42}")
    offer_count = 0
    total_states = 0
    for (risk, n_inc, month), action in sorted(policy.items()):
        if risk == 0:
            continue
        total_states += 1
        if action == 1:
            offer_count += 1
            if total_states <= 20:  # print first few
                print(f"  {RISK_LABELS[RiskState(risk)]:<12} {n_inc:>10} {month:>8} {'OFFER':>8}")
    print(f"\n  Policy recommends offering in {offer_count}/{total_states} "
          f"({100*offer_count/total_states:.1f}%) of non-default states")

    # ---- Monte Carlo Simulation ----
    N = N_SIMULATIONS
    print(f"\n[6] Running Monte Carlo simulations (N={N} replications)...")

    policies = [
        "greedy", "conservative", "risk_aware", "mdp_optimal",
        "segmented", "timing_optimized", "risk_drift_aware", "macro_aware", "bandit",
    ]
    summary_results = []

    for pol in policies:
        print(f"\n  Policy: {pol.upper()}")
        all_runs = []
        sim_params = None
        if pol == "mdp_optimal":
            sim_params = {"policy_table": policy}
        for run in range(N):
            seed = 42 + run * 137
            result_df = simulator.run_simulation(policy=pol, threshold_params=sim_params, seed=seed)
            agg = {
                "total_profit": result_df["profit"].sum(),
                "total_npv": result_df["npv_profit"].sum(),
                "avg_increases": result_df["increases_accepted"].mean(),
                "default_rate": result_df["defaults"].mean(),
                "total_offers": result_df["increases_offered"].sum(),
                "total_exposure": result_df["total_exposure"].sum(),
                "profitable_customers": (result_df["profit"] > 0).sum(),
            }
            all_runs.append(agg)

        # Aggregate over runs
        keys = ["total_profit", "total_npv", "avg_increases", "default_rate",
                "total_offers", "total_exposure", "profitable_customers"]
        agg_stats = {}
        for key in keys:
            vals = [r[key] for r in all_runs]
            agg_stats[f"{key}_mean"] = np.mean(vals)
            agg_stats[f"{key}_std"] = np.std(vals)
            agg_stats[f"policy"] = pol

        summary_results.append(agg_stats)

        print(f"    Mean Total Profit:     ${agg_stats['total_profit_mean']:,.0f} "
              f"(±${agg_stats['total_profit_std']:,.0f})")
        print(f"    Mean Total NPV:        ${agg_stats['total_npv_mean']:,.0f} "
              f"(±${agg_stats['total_npv_std']:,.0f})")
        print(f"    Mean Increases/Cust:   {agg_stats['avg_increases_mean']:.2f}")
        print(f"    Mean Default Rate:     {agg_stats['default_rate_mean']:.4f} "
              f"({100*agg_stats['default_rate_mean']:.2f}%)")
        print(f"    Mean Total Offers:     {agg_stats['total_offers_mean']:,.0f}")
        print(f"    Mean Profitable Cust:  {agg_stats['profitable_customers_mean']:,.0f}")

    # ---- Comparison ----
    print("\n[7] Policy Comparison...")
    print(f"\n  {'Policy':<15} {'Total Profit':>16} {'NPV':>16} {'Default Rate':>14} {'Incr/Cust':>12}")
    print(f"  {'-'*75}")
    best_npv = -float("inf")
    best_pol = ""
    for s in summary_results:
        print(f"  {s['policy']:<15} ${s['total_profit_mean']:>14,.0f} "
              f"${s['total_npv_mean']:>14,.0f} "
              f"{100*s['default_rate_mean']:>12.2f}% "
              f"{s['avg_increases_mean']:>10.2f}")
        if s['total_npv_mean'] > best_npv:
            best_npv = s['total_npv_mean']
            best_pol = s['policy']

    print(f"\n  >>> Best policy by NPV: {best_pol.upper()} (${best_npv:,.0f})")

    # ---- ASCII Bar Chart ----
    print("\n  NPV Comparison (relative to best):")
    print(f"  {'Policy':<17} {'NPV':>12} {'Δ from best':>14} {'Bar':>30}")
    print(f"  {'-'*75}")
    worst_npv = min(s['total_npv_mean'] for s in summary_results)
    best_npv_val = best_npv
    npv_range = max(best_npv_val - worst_npv, 1)
    for s in sorted(summary_results, key=lambda x: x['total_npv_mean'], reverse=True):
        npv = s['total_npv_mean']
        delta = npv - best_npv_val
        bar_len = int(40 * (npv - worst_npv) / npv_range)
        bar = '█' * bar_len
        marker = ' ◀ BEST' if s['policy'] == best_pol else ''
        print(f"  {s['policy']:<17} ${npv:>11,.0f} ${delta:>13,.0f}  {bar}{marker}")

    # ---- Risk-Adjusted Returns ----
    print("\n[8] Risk-Adjusted Return Analysis...")
    for s in summary_results:
        sharpe_like = s['total_npv_mean'] / max(s['total_profit_std'], 1)
        loss_rate = s['default_rate_mean']
        cust_count = 30000
        print(f"  {s['policy']:<15} Risk-Adj Return: {sharpe_like:,.0f}  "
              f"Loss Rate: {100*loss_rate:.3f}%  "
              f"Profit/Customer: ${s['total_profit_mean']/cust_count:,.0f}")

    # ---- Regulatory Capital Check ----
    print("\n[9] Regulatory Capital Constraint Check...")
    for s in summary_results:
        capital_required = s['total_exposure_mean'] * RISK_CAPITAL_RATIO
        portfolio_loss_rate = s['default_rate_mean']
        compliant = portfolio_loss_rate <= MAX_PORTFOLIO_LOSS_RATE
        status = "PASS" if compliant else "FAIL"
        print(f"  {s['policy']:<15} Capital Req: ${capital_required:>12,.0f}  "
              f"Loss Rate: {100*portfolio_loss_rate:.2f}%  "
              f"Threshold: {100*MAX_PORTFOLIO_LOSS_RATE:.1f}%  [{status}]")

    # ---- Data-Driven Recommendations ----
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    sorted_results = sorted(summary_results, key=lambda s: s['total_npv_mean'], reverse=True)
    top_pol = sorted_results[0]

    print(f"""
  1. POLICY: The best-performing policy by NPV is '{top_pol['policy'].upper()}'
     (${top_pol['total_npv_mean']:,.0f} NPV, {top_pol['avg_increases_mean']:.2f} increases/customer,
     {100*top_pol['default_rate_mean']:.2f}% default rate).

  2. SEGMENTATION: Risk-based offering limits are implemented via the 'segmented' policy:
     - Prime (>=95% on-time):   up to 5 increases/yr
     - Near-Prime (90-95%):    up to 3 increases/yr
     - Subprime (<90%):        up to 2 increases/yr
""")

    seg_result = next((s for s in summary_results if s['policy'] == 'segmented'), None)
    if seg_result:
        print(f"     Segmented policy result: ${seg_result['total_npv_mean']:,.0f} NPV, "
              f"{seg_result['avg_increases_mean']:.2f} incr/cust, "
              f"{100*seg_result['default_rate_mean']:.2f}% default rate")

    print("""
  3. TIMING: Offer in the 60-90 day window after last disbursement when demand
     is strongest. The 'timing_optimized' policy enforces this window and
     reduces default exposure by limiting offer frequency.
""")

    drift_result = next((s for s in summary_results if s['policy'] == 'risk_drift_aware'), None)
    if drift_result:
        print(f"""  4. MONITORING: Risk drift tracking downgrades caps when customers deteriorate.
     Drift-aware policy result: ${drift_result['total_npv_mean']:,.0f} NPV,
     {drift_result['avg_increases_mean']:.2f} incr/cust.
     Each risk downgrade reduces the allowed increase cap by 1.
""")

    macro_result = next((s for s in summary_results if s['policy'] == 'macro_aware'), None)
    if macro_result:
        print(f"""  5. ECONOMIC FACTORS: Macro-aware policy adjusts aggressiveness based on
     inflation (3.5%), unemployment (4.2%), interest rate (5.5%).
     Macro-aware result: ${macro_result['total_npv_mean']:,.0f} NPV,
     {macro_result['avg_increases_mean']:.2f} incr/cust.
""")

    bandit_result = next((s for s in summary_results if s['policy'] == 'bandit'), None)
    if bandit_result:
        n_learned = bandit_agent.q_table.__len__() if hasattr(bandit_agent, 'q_table') else 'N/A'
        print(f"""  6. REINFORCEMENT LEARNING: Contextual bandit (epsilon=0.10) learns
     optimal offer strategy online from customer responses.
     Bandit result: ${bandit_result['total_npv_mean']:,.0f} NPV,
     {bandit_result['avg_increases_mean']:.2f} incr/cust.
     States learned: {n_learned}
""")

    # ---- Operational Deployment Guide ----
    print("=" * 70)
    print("OPERATIONAL DEPLOYMENT GUIDE")
    print("=" * 70)
    print("""
  IMMEDIATE (0-3 months):
    - Deploy timing_optimized policy: add 60-90 day window filter
    - Configure risk-aware gate on top of window filter
    - Run in shadow mode for 2 weeks, compare against current policy
    - A/B test: treatment (timing-optimized) vs control (current)

  MEDIUM-TERM (3-6 months):
    - Deploy bandit agent as microservice (REST API: state -> action)
    - Log all (state, action, reward) tuples for offline analysis
    - Retrain weekly; decay epsilon from 0.10 to 0.02

  STRATEGIC (6-12 months):
    - Recalibrate Markov transition matrix on 24-36 month historical data
    - Add vintage/product/geographic segmentation
    - Implement risk-drift early-warning triggers
    - Build macro-overlay API (CPI, unemployment claims, Fed rate)

  RISK CONTROLS:
    Loss Rate > 5%     -> Pause all offers, investigate
    Subprime > 55%     -> Cap at 1 increase
    Drift count >= 2   -> Freeze customer, manual review
    Macro agg < 0.60   -> Recession protocol: Prime-only, cap 2
""")

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return summary_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_analysis()
