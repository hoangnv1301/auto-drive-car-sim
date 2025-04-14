from simulation.vehicle import Vehicle
from simulation.road import Road
from simulation.autonomous_logic import get_autonomous_logic, PurePursuitController

class Simulation:
    """Main simulation class that manages all simulation objects and logic."""

    def __init__(self):
        """Initialize the simulation with default values."""
        self.time = 0.0
        self.delta_time = 0.05  # 50ms time step, 20 FPS
        self.vehicles = []
        self.road = Road()
        
        # Create main vehicle with Pure Pursuit controller
        main_vehicle = Vehicle(
            position=[0, 0, 0],
            rotation=[0, 0, 0],
            autonomous_logic=PurePursuitController()
        )
        self.vehicles.append(main_vehicle) 