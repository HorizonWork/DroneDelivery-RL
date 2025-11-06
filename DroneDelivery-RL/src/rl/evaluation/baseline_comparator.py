"""
A*, RRT*, Random baseline comparison
""",
class BaselineComparator:
    def __init__(self, ppo_agent):
        self.ppo_agent = ppo_agent
        self.baselines = ["astar", "rrt", "random"]
        
    def compare_all(self):
        results = {}
        for baseline in self.baselines:
            results[baseline] = self.evaluate_baseline(baseline)
        return results
        
    def evaluate_baseline(self, baseline_type):
        # Evaluate specific baseline
        pass
