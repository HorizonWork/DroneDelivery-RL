import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

class GAECalculator:

    def __init__(self, gae_lambda: float = 0.95, discount_factor: float = 0.99):
        self.gae_lambda = gae_lambda
        self.discount_factor = discount_factor
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"GAE Calculator initialized: λ={gae_lambda}, γ={discount_factor}")

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                   dones: np.ndarray, next_value: float,
                   next_done: bool) - Tuple[np.ndarray, np.ndarray]:

        sequence_length = len(rewards)
        advantages = np.zeros(sequence_length, dtype=np.float32)
        returns = np.zeros(sequence_length, dtype=np.float32)

        next_value = next_value if not next_done else 0.0
        next_advantage = 0.0

        for t in reversed(range(sequence_length)):
            if t == sequence_length - 1:
                next_non_terminal = 1.0 - next_done
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val = values[t + 1]

            td_error = rewards[t] + self.discount_factor  next_val  next_non_terminal - values[t]

            advantages[t] = td_error + self.discount_factor  self.gae_lambda  next_advantage  next_non_terminal
            next_advantage = advantages[t]

            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def compute_returns_only(self, rewards: np.ndarray, dones: np.ndarray,
                           next_value: float, next_done: bool) - np.ndarray:

        sequence_length = len(rewards)
        returns = np.zeros(sequence_length, dtype=np.float32)

        next_return = next_value if not next_done else 0.0

        for t in reversed(range(sequence_length)):
            if t == sequence_length - 1:
                next_non_terminal = 1.0 - next_done
            else:
                next_non_terminal = 1.0 - dones[t + 1]

            returns[t] = rewards[t] + self.discount_factor  next_return  next_non_terminal
            next_return = returns[t]

        return returns

    def validate_gae_computation(self, rewards: np.ndarray, values: np.ndarray,
                               advantages: np.ndarray, returns: np.ndarray) - Dict[str, bool]:

        validation = {}

        validation['dimensions_match'] = (len(rewards) == len(values) ==
                                        len(advantages) == len(returns))

        returns_check = np.allclose(returns, advantages + values, rtol=1e-5)
        validation['returns_formula_correct'] = returns_check

        validation['no_nan_advantages'] = not np.any(np.isnan(advantages))
        validation['no_inf_advantages'] = not np.any(np.isinf(advantages))
        validation['no_nan_returns'] = not np.any(np.isnan(returns))
        validation['no_inf_returns'] = not np.any(np.isinf(returns))

        advantage_std = np.std(advantages)
        validation['advantage_variance_reasonable'] = 0.1 = advantage_std = 100.0

        validation['overall_valid'] = all(validation.values())

        if not validation['overall_valid']:
            self.logger.warning("GAE computation validation failed")

        return validation

    def get_gae_statistics(self, advantages: np.ndarray, returns: np.ndarray) - Dict[str, float]:

        return {
            'gae_lambda': self.gae_lambda,
            'discount_factor': self.discount_factor,
            'advantage_mean': float(np.mean(advantages)),
            'advantage_std': float(np.std(advantages)),
            'advantage_min': float(np.min(advantages)),
            'advantage_max': float(np.max(advantages)),
            'returns_mean': float(np.mean(returns)),
            'returns_std': float(np.std(returns)),
            'returns_min': float(np.min(returns)),
            'returns_max': float(np.max(returns))
        }
