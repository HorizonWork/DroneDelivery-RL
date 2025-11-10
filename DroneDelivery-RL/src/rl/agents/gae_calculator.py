"""
GAE Calculator
Generalized Advantage Estimation with λ = 0.95 (Table 2).
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

class GAECalculator:
    """
    Generalized Advantage Estimation calculator.
    Implements GAE(λ) for stable advantage computation.
    """
    
    def __init__(self, gae_lambda: float = 0.95, discount_factor: float = 0.99):
        self.gae_lambda = gae_lambda        # λ parameter (Table 2: 0.95)
        self.discount_factor = discount_factor  # γ discount factor (Table 2: 0.99)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"GAE Calculator initialized: λ={gae_lambda}, γ={discount_factor}")
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, 
                   dones: np.ndarray, next_value: float, 
                   next_done: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        # Support both 'lam' and 'lambda_' parameter names
        if lam is not None:
            lambda_ = lam
        
        # Support both 'lam' and 'lambda_' parameter names
        if 'lam' in locals() and lam is not None:
            lambda_ = lam

        Compute GAE advantages and returns.
        
        Args:
            rewards: Reward sequence [T]
            values: Value estimates [T]
            dones: Done flags [T]
            next_value: Bootstrap value for final state
            next_done: Bootstrap done flag
            
        Returns:
            (advantages, returns) tuple
        """
        sequence_length = len(rewards)
        advantages = np.zeros(sequence_length, dtype=np.float32)
        returns = np.zeros(sequence_length, dtype=np.float32)
        
        # Initialize for backward computation
        next_value = next_value if not next_done else 0.0
        next_advantage = 0.0
        
        # Compute advantages and returns backward through sequence
        for t in reversed(range(sequence_length)):
            # TD error: δt = rt + γV(st+1) - V(st)
            if t == sequence_length - 1:
                next_non_terminal = 1.0 - next_done
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val = values[t + 1]
            
            td_error = rewards[t] + self.discount_factor * next_val * next_non_terminal - values[t]
            
            # GAE: At = δt + γλAt+1
            advantages[t] = td_error + self.discount_factor * self.gae_lambda * next_advantage * next_non_terminal
            next_advantage = advantages[t]
            
            # Returns: Rt = At + V(st)
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def compute_returns_only(self, rewards: np.ndarray, dones: np.ndarray, 
                           next_value: float, next_done: bool) -> np.ndarray:
        """
        Compute returns without advantage estimation.
        
        Args:
            rewards: Reward sequence
            dones: Done flags
            next_value: Bootstrap value
            next_done: Bootstrap done flag
            
        Returns:
            Returns array
        """
        sequence_length = len(rewards)
        returns = np.zeros(sequence_length, dtype=np.float32)
        
        # Initialize
        next_return = next_value if not next_done else 0.0
        
        # Compute returns backward
        for t in reversed(range(sequence_length)):
            if t == sequence_length - 1:
                next_non_terminal = 1.0 - next_done
            else:
                next_non_terminal = 1.0 - dones[t + 1]
            
            returns[t] = rewards[t] + self.discount_factor * next_return * next_non_terminal
            next_return = returns[t]
        
        return returns
    
    def validate_gae_computation(self, rewards: np.ndarray, values: np.ndarray,
                               advantages: np.ndarray, returns: np.ndarray) -> Dict[str, bool]:
        """
        Validate GAE computation correctness.
        
        Args:
            rewards: Original rewards
            values: Value estimates
            advantages: Computed advantages
            returns: Computed returns
            
        Returns:
            Validation results
        """
        validation = {}
        
        # Check dimensions
        validation['dimensions_match'] = (len(rewards) == len(values) == 
                                        len(advantages) == len(returns))
        
        # Check that returns = advantages + values (approximately)
        returns_check = np.allclose(returns, advantages + values, rtol=1e-5)
        validation['returns_formula_correct'] = returns_check
        
        # Check for NaN or infinite values
        validation['no_nan_advantages'] = not np.any(np.isnan(advantages))
        validation['no_inf_advantages'] = not np.any(np.isinf(advantages))
        validation['no_nan_returns'] = not np.any(np.isnan(returns))
        validation['no_inf_returns'] = not np.any(np.isinf(returns))
        
        # Check advantage statistics
        advantage_std = np.std(advantages)
        validation['advantage_variance_reasonable'] = 0.1 <= advantage_std <= 100.0
        
        # Overall validation
        validation['overall_valid'] = all(validation.values())
        
        if not validation['overall_valid']:
            self.logger.warning("GAE computation validation failed")
            
        return validation
    
    def get_gae_statistics(self, advantages: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """
        Get GAE computation statistics.
        
        Args:
            advantages: Computed advantages
            returns: Computed returns
            
        Returns:
            Statistics dictionary
        """
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
