"""
1→2→5 floor curriculum implementation
""",
class CurriculumManager:
    def __init__(self):
        self.current_phase = 1 # Start with single floor
        self.phases = {
            1: {"floors": 1, "timesteps": 1000000},  # Phase 1: 1 floor, 1M timesteps
            2: {"floors": 2, "timesteps": 2000000},  # Phase 2: 2 floors, 2M timesteps
            3: {"floors": 5, "timesteps": 2000000}   # Phase 3: 5 floors, 2M timesteps
        }
        
    def advance_curriculum(self):
        if self.current_phase < 3:
            self.current_phase += 1
        return self.current_phase
        
    def get_current_config(self):
        return self.phases[self.current_phase]
