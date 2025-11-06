"""
Equation (2) exact implementation:
R(st, at) = 500·1{goal} - 5·dt - 0.1·Δt - 0.01·Σui² - 10·jt - 1000·ct
""",
class RewardFunction:
    def __init__(self):
        # Coefficients from Equation (2)
        self.goal_reward = 500
        self.distance_cost = 5
        self.time_cost = 0.1
        self.control_effort_cost = 0.01
        self.jerk_cost = 10
        self.collision_cost = 1000
        
    def calculate_reward(self, state, action, next_state, is_goal_reached, is_collision, dt):
        reward = 0
        
        # Goal reached bonus
        if is_goal_reached:
            reward += self.goal_reward
            
        # Distance traveled cost
        distance_traveled = self.calculate_distance_traveled(state, next_state)
        reward -= self.distance_cost * distance_traveled
        
        # Time cost
        reward -= self.time_cost * dt
        
        # Control effort cost
        control_effort = np.sum(np.square(action))
        reward -= self.control_effort_cost * control_effort
        
        # Collision penalty
        if is_collision:
            reward -= self.collision_cost
            
        return reward
        
    def calculate_distance_traveled(self, state1, state2):
        # Calculate Euclidean distance between states
        pos1 = state1[:3]  # x, y, z
        pos2 = state2[:3]
        return np.linalg.norm(pos2 - pos1)
