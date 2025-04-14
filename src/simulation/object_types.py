from enum import Enum, auto

class ObjectType(Enum):
    CAR = auto()
    TRUCK = auto()
    TRICAR = auto()  # Three-wheeled vehicle
    CYCLIST = auto()
    PEDESTRIAN = auto()
    
    def get_color(self):
        """Return the color associated with this object type."""
        colors = {
            ObjectType.CAR: [0, 255, 0],        # Green
            ObjectType.TRUCK: [0, 200, 200],    # Teal/Cyan
            ObjectType.TRICAR: [255, 165, 0],   # Orange
            ObjectType.CYCLIST: [0, 0, 255],    # Blue
            ObjectType.PEDESTRIAN: [255, 0, 255] # Purple/Magenta
        }
        return colors.get(self, [200, 200, 200])  # Default gray 