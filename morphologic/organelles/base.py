from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from skimage import measure, morphology
from ..config.channels import Channel, ChannelMap

@dataclass
class OrganelleFeatures:
    """Base class for organelle features."""
    area: float = 0.0
    perimeter: float = 0.0
    eccentricity: float = 0.0
    mean_intensity: float = 0.0
    orientation: float = 0.0
    major_axis_length: float = 0.0
    minor_axis_length: float = 0.0
    solidity: float = 0.0
    extent: float = 0.0

class Organelle(ABC):
    """Base class for all organelle analyzers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.channel: Optional[Channel] = None
        self._set_channel()

    @abstractmethod
    def _set_channel(self) -> None:
        """Set the channel for this organelle."""
        pass

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment the organelle from the image.
        
        Args:
            image: Input image array
            
        Returns:
            Binary mask of the segmented organelle
        """
        pass

    @abstractmethod
    def analyze(self, mask: np.ndarray, intensity_image: Optional[np.ndarray] = None) -> OrganelleFeatures:
        """
        Analyze the organelle features.
        
        Args:
            mask: Binary mask of the segmented organelle
            intensity_image: Optional intensity image for additional features
            
        Returns:
            OrganelleFeatures object containing analysis results
        """
        pass

    def extract_basic_features(self, 
                             mask: np.ndarray, 
                             intensity_image: Optional[np.ndarray] = None) -> OrganelleFeatures:
        """Extract basic morphological features."""
        features = OrganelleFeatures()
        
        if not np.any(mask):
            return features

        # Ensure mask is binary
        mask = mask > 0
        
        # Get region properties
        regions = measure.regionprops(mask.astype(int), intensity_image=intensity_image)
        if not regions:
            return features

        # Get the largest region if multiple exist
        region = max(regions, key=lambda r: r.area)
        
        # Fill basic features
        features.area = region.area
        features.perimeter = region.perimeter
        features.eccentricity = region.eccentricity
        features.orientation = region.orientation
        features.major_axis_length = region.major_axis_length
        features.minor_axis_length = region.minor_axis_length
        features.solidity = region.solidity
        features.extent = region.extent
        
        if intensity_image is not None:
            features.mean_intensity = region.mean_intensity

        return features

    def clean_mask(self, mask: np.ndarray, min_size: Optional[int] = None) -> np.ndarray:
        """
        Clean a binary mask by removing small objects and filling holes.
        
        Args:
            mask: Binary mask to clean
            min_size: Minimum object size to keep (defaults to config value)
            
        Returns:
            Cleaned binary mask
        """
        if min_size is None:
            min_size = self.config.get('min_size', 50)

        # Ensure mask is binary
        mask = mask > 0
        
        # Remove small objects
        cleaned = morphology.remove_small_objects(mask, min_size=min_size)
        
        # Fill holes
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size)
        
        return cleaned

    def compute_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute the overlap coefficient between two masks.
        
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            
        Returns:
            Overlap coefficient between 0 and 1
        """
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        
        if not np.any(union):
            return 0.0
            
        return np.sum(intersection) / np.sum(union)

    def get_boundary_points(self, mask: np.ndarray) -> np.ndarray:
        """
        Get the boundary points of a mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Array of boundary coordinates
        """
        boundaries = measure.find_contours(mask.astype(float), 0.5)
        if not boundaries:
            return np.array([])
        return boundaries[0]

    def get_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """
        Get the centroid of a mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Tuple of (y, x) coordinates of the centroid
        """
        regions = measure.regionprops(mask.astype(int))
        if not regions:
            return (0.0, 0.0)
        return regions[0].centroid