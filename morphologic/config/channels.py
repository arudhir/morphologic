from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass(frozen=True)
class Channel:
    """Represents a fluorescence microscopy channel with its properties."""
    index: int  # 0-based index for array access
    name: str   # Technical name (e.g., "w1")
    stain: str  # Staining agent used
    target: str # Biological target
    rgb: Tuple[int, int, int] = (255, 255, 255)  # Default visualization color
    intensity_range: Tuple[int, int] = (0, 255)   # Expected intensity range

    @property
    def channel_id(self) -> str:
        """Returns a unique identifier for the channel."""
        return f"{self.name}_{self.target.lower()}"

class ChannelMap(Enum):
    """Standard channel configuration for the HUVEC cell imaging pipeline."""
    
    NUCLEUS = Channel(
        index=0,
        name="w1",
        stain="Hoechst 33342",
        target="Nucleus",
        rgb=(19, 0, 249),
        intensity_range=(0, 51)
    )
    
    MEMBRANE_GLYCOPROTEINS = Channel(
        index=1,
        name="w2",
        stain="Concanavalin A",
        target="Membrane glycoproteins",
        rgb=(42, 255, 31),
        intensity_range=(0, 107)
    )
    
    ACTIN = Channel(
        index=2,
        name="w3",
        stain="Phalloidin",
        target="Actin",
        rgb=(255, 0, 25),
        intensity_range=(0, 64)
    )
    
    RNA = Channel(
        index=3,
        name="w4",
        stain="Syto14",
        target="RNA",
        rgb=(45, 255, 252),
        intensity_range=(0, 191)
    )
    
    GOLGI = Channel(
        index=4,
        name="w5",
        stain="Wheat germ agglutinin",
        target="Golgi",
        rgb=(250, 0, 253),
        intensity_range=(0, 89)
    )
    
    MITOCHONDRIA = Channel(
        index=5,
        name="w6",
        stain="Mitotracker",
        target="Mitochondria",
        rgb=(254, 255, 40),
        intensity_range=(0, 191)
    )

    @classmethod
    def get_by_index(cls, index: int) -> Optional['ChannelMap']:
        """Get channel by its index."""
        for channel in cls:
            if channel.value.index == index:
                return channel
        return None

    @classmethod
    def get_by_target(cls, target: str) -> Optional['ChannelMap']:
        """Get channel by its biological target."""
        target_lower = target.lower()
        for channel in cls:
            if channel.value.target.lower() == target_lower:
                return channel
        return None

    @classmethod
    def get_by_name(cls, name: str) -> Optional['ChannelMap']:
        """Get channel by its technical name."""
        for channel in cls:
            if channel.value.name == name:
                return channel
        return None

def get_channel_indices() -> Dict[str, int]:
    """Returns a dictionary mapping target names to channel indices."""
    return {channel.value.target.lower(): channel.value.index 
            for channel in ChannelMap}

def get_rgb_mappings() -> Dict[int, Dict[str, Tuple[int, int, int] | Tuple[int, int]]]:
    """Returns the RGB mapping configuration for visualization."""
    return {
        channel.value.index + 1: {  # +1 for 1-based indexing in visualization
            'rgb': channel.value.rgb,
            'range': channel.value.intensity_range
        }
        for channel in ChannelMap
    }