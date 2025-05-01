import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.envs.registration import register
from arguments import get_args
from collections import deque
class RandomizedPendulumEnv(PendulumEnv):
    def __init__(self, gravity_range=(5.0, 15.0), length_range=(2.0, 2.0),
                 max_speed_range=(9.0, 9.0), max_torque_range=(2.0, 2.0), render_mode=None):
        super().__init__(render_mode=render_mode)
        self.gravity_range = gravity_range
        self.length_range = length_range
        self.max_speed_range = max_speed_range
        self.max_torque_range = max_torque_range
        self.parameters = [self.g, self.l, self.max_speed, self.max_torque]
        self.T_reset =0
        self.store_parameter_indicator_pairs = deque(maxlen=1000)
        # Divide the gravity range into five sub-ranges
        self.gravity_intervals = {
            (5.0, 7.0): 0,
            (7.0, 9.0): 0,
            (9.0, 11.0): 0,
            (11.0, 13.0): 0,
            (13.0, 15.0): 0
        }

    def sample_parameters_by_indicator(self, noise_std=0.05):
        
        if not hasattr(self, 'store_parameter_indicator_pairs') or self.store_parameter_indicator_pairs is None:
            raise ValueError("Parameter buffer not set. Use set_store(...) before sampling.")

        # Filter out valid entries and remember their original indices
        indexed_valid_pairs = [(i, params, indicator) for i, (params, indicator) in enumerate(self.store_parameter_indicator_pairs) if
                               indicator is not None]
        if not indexed_valid_pairs:
            raise ValueError("No valid (parameters, indicator) pairs with non-None indicator to sample from.")

        # Compute probabilities based on ranks
        indicators = np.array([indicator for _, _, indicator in indexed_valid_pairs], dtype=np.float32)
        # Sort indicators in descending order (more negative indicators get higher priority)
        ranks = np.argsort(np.argsort(indicators)) 
        
        # Convert ranks to probabilities (similar to prioritized experience replay)
        alpha = 0.6  # Priority exponent, controls how much prioritization is used
        print(alpha)
        probs = 1.0 / (ranks + 1) ** alpha  # Add 1 to avoid division by zero
        probs /= probs.sum()  # Normalize to make it a valid probability distribution
        
        # Sample
        idx_in_list = np.random.choice(len(indexed_valid_pairs), p=probs)
        original_index, sampled_params, _ = indexed_valid_pairs[idx_in_list]

        # Remove the selected entry from the original deque
        del self.store_parameter_indicator_pairs[original_index]

        # Add noise
        sampled_params = np.array(sampled_params, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=sampled_params.shape)
        perturbed_params = sampled_params + noise

        return perturbed_params.tolist()

    def update_gravity_sampling_count(self, gravity_value):
        """
        Update the sampling count for the gravity interval that includes the given gravity_value.
        """
        for interval in self.gravity_intervals:
            if interval[0] <= gravity_value < interval[1]:
                self.gravity_intervals[interval] += 1
                break

    def randomize(self):
        g = np.random.uniform(*self.gravity_range)
        l = np.random.uniform(*self.length_range)
        max_speed = np.random.uniform(*self.max_speed_range)
        max_torque = np.random.uniform(*self.max_torque_range)
        return [g, l, max_speed, max_torque]

    def sample_mixed_parameters(self):
        k = 0.0005605
        if np.random.rand() < max(0.1, np.exp(-k * (self.T_reset))):
            # Sample fresh parameters via randomize()
            self.g, self.l, self.max_speed, self.max_torque = self.randomize()
        else:
            # Sample from past indicator weighted parameters
            self.g, self.l, self.max_speed, self.max_torque = self.sample_parameters_by_indicator()
        self.parameters = [self.g, self.l, self.max_speed, self.max_torque]


    def reset(self, *, seed=None, options=None):
        if self.T_reset>10*10:
            if len(self.store_parameter_indicator_pairs)<100:
                self.g, self.l, self.max_speed, self.max_torque = self.randomize()
                self.parameters = [self.g, self.l, self.max_speed, self.max_torque]
            else:
                self.sample_mixed_parameters()

        else:
            self.g, self.l, self.max_speed, self.max_torque = self.randomize()
            self.parameters = [self.g, self.l, self.max_speed, self.max_torque]
        
        self.update_gravity_sampling_count(self.g)
        return super().reset(seed=seed, options=options)

register(
    id='RandomizedPendulumEnv-v0',
    entry_point=RandomizedPendulumEnv,
)






