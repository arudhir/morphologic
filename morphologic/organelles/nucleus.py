from dataclasses import dataclass
from typing import Optional
import numpy as np
from skimage import feature
from cellpose import models

from .base import Organelle, OrganelleFeatures
from ..config.channels import ChannelMap

@dataclass
class NuclearFeatures(OrganelleFeatures):
    """Nuclear-specific features."""
    chromatin_homogeneity: float = 0.0
    chromatin_contrast: float = 0.0
    is_mitotic: bool = False
    nuclear_irregularity: float = 0.0
    nuclear_membrane_intensity: float = 0.0
    chromatin_distribution: float = 0.0
    number_of_nucleoli: int = 0
    condensation_state: float = 0.0

class Nucleus(Organelle):
    """Analyzer for nuclear features."""

    def _set_channel(self) -> None:
        """Set the nuclear channel from ChannelMap."""
        self.channel = ChannelMap.NUCLEUS.value

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment nuclei using Cellpose.
        
        Args:
            image: DAPI/Hoechst channel image
            
        Returns:
            Binary mask of segmented nuclei
        """
        # Initialize Cellpose model
        model = models.Cellpose(
            gpu=self.config.get('use_gpu', False),
            model_type='nuclei'
        )
        
        # Run segmentation
        masks, _, _, _ = model.eval(image, diameter=None, channels=[0, 0])
        
        # Clean up masks
        masks = self.clean_mask(masks, self.config.get('min_nucleus_size', 50))
        
        return masks

    def analyze(self, 
                mask: np.ndarray, 
                intensity_image: Optional[np.ndarray] = None) -> NuclearFeatures:
        """
        Analyze nuclear morphology and chromatin organization.
        
        Args:
            mask: Binary mask of segmented nucleus
            intensity_image: DAPI/Hoechst intensity image
            
        Returns:
            NuclearFeatures object containing analysis results
        """
        # Get basic features first
        basic_features = self.extract_basic_features(mask, intensity_image)
        features = NuclearFeatures(**basic_features.__dict__)
        
        if not np.any(mask) or intensity_image is None:
            return features

        # Normalize intensity image for texture analysis
        normalized_image = self._normalize_intensity(intensity_image)
        
        # Calculate texture features using GLCM
        glcm = feature.graycomatrix(
            normalized_image,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        
        features.chromatin_homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        features.chromatin_contrast = feature.graycoprops(glcm, 'contrast').mean()
        
        # Detect mitotic state
        features.is_mitotic = self._detect_mitotic_nucleus(normalized_image)
        
        # Calculate nuclear irregularity
        if features.area > 0:
            features.nuclear_irregularity = (features.perimeter ** 2) / (4 * np.pi * features.area)
        
        # Analyze chromatin distribution
        features.chromatin_distribution = self._analyze_chromatin_distribution(
            intensity_image, mask
        )
        
        # Detect nucleoli
        features.number_of_nucleoli = self._count_nucleoli(intensity_image, mask)
        
        # Analyze chromatin condensation state
        features.condensation_state = self._analyze_condensation_state(
            intensity_image, mask
        )
        
        return features

    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize intensity image to 8-bit range."""
        normalized = ((image - image.min()) * 255 / 
                     (image.max() - image.min())).astype(np.uint8)
        return normalized

    def _detect_mitotic_nucleus(self, normalized_image: np.ndarray) -> bool:
        """
        Detect mitotic nuclei based on chromatin patterns.
        
        Args:
            normalized_image: Normalized intensity image
            
        Returns:
            Boolean indicating if nucleus is likely in mitosis
        """
        # Detect condensed chromatin regions
        blobs = feature.blob_log(
            normalized_image,
            min_sigma=1,
            max_sigma=30,
            num_sigma=10,
            threshold=.2
        )
        
        # Check for characteristic mitotic patterns
        return len(blobs) > 2

    def _analyze_chromatin_distribution(self, 
                                     intensity_image: np.ndarray, 
                                     mask: np.ndarray) -> float:
        """
        Analyze the radial distribution of chromatin.
        
        Returns:
            Float between 0 (uniform) and 1 (peripheral)
        """
        if not np.any(mask):
            return 0.0
            
        # Get distance transform from nuclear boundary
        distance = feature.distance_transform_edt(mask)
        max_distance = distance.max()
        if max_distance == 0:
            return 0.0
            
        # Normalize distances
        distance_norm = distance / max_distance
        
        # Calculate weighted intensity by distance
        weighted_intensity = np.sum(intensity_image * distance_norm * mask)
        total_intensity = np.sum(intensity_image * mask)
        
        if total_intensity == 0:
            return 0.0
            
        return weighted_intensity / total_intensity

    def _count_nucleoli(self, intensity_image: np.ndarray, mask: np.ndarray) -> int:
        """Count number of nucleoli based on intensity peaks."""
        if not np.any(mask):
            return 0
            
        # Apply mask to intensity image
        masked_intensity = intensity_image * mask
        
        # Find local maxima
        coordinates = feature.peak_local_max(
            masked_intensity,
            min_distance=5,
            threshold_rel=0.7,
            exclude_border=False
        )
        
        return len(coordinates)

    def _analyze_condensation_state(self, 
                                  intensity_image: np.ndarray, 
                                  mask: np.ndarray) -> float:
        """
        Analyze chromatin condensation state.
        
        Returns:
            Float between 0 (decondensed) and 1 (highly condensed)
        """
        if not np.any(mask):
            return 0.0
            
        # Get intensity statistics within the nucleus
        masked_intensity = intensity_image * mask
        intensity_values = masked_intensity[mask > 0]
        
        if len(intensity_values) == 0:
            return 0.0
            
        # Calculate coefficient of variation
        cv = np.std(intensity_values) / np.mean(intensity_values)
        
        # Normalize to [0, 1] range (assuming CV > 1 indicates high condensation)
        return np.clip(cv / 2, 0, 1)