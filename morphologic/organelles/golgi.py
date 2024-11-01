from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import ndimage, spatial
from skimage import filters, measure, morphology
from .base import Organelle, OrganelleFeatures
from ..config.channels import ChannelMap

@dataclass
class GolgiFeatures(OrganelleFeatures):
    """Golgi-specific features."""
    # Fragmentation metrics
    fragment_count: int = 0
    mean_fragment_size: float = 0.0
    size_variance: float = 0.0
    
    # Structure metrics
    compactness: float = 0.0
    polarization_index: float = 0.0
    dispersion: float = 0.0
    ribbon_continuity: float = 0.0
    
    # Position metrics
    centroid_distance_to_nucleus: float = 0.0
    orientation_relative_to_nucleus: float = 0.0
    
    # Morphological metrics
    cisternal_length: float = 0.0
    stack_thickness: float = 0.0
    stack_organization: float = 0.0

class Golgi(Organelle):
    """Analyzer for Golgi apparatus features."""

    def _set_channel(self) -> None:
        """Set the Golgi channel from ChannelMap."""
        self.channel = ChannelMap.GOLGI.value

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment Golgi apparatus using adaptive thresholding.
        
        Args:
            image: Golgi channel image
            
        Returns:
            Binary mask of segmented Golgi
        """
        # Apply Gaussian blur to reduce noise
        smoothed = filters.gaussian(image, sigma=1.0)
        
        # Adaptive thresholding
        thresh = filters.threshold_local(smoothed, block_size=35, method='gaussian')
        binary = smoothed > thresh
        
        # Clean up mask
        cleaned = self.clean_mask(
            binary, 
            self.config.get('min_golgi_size', 10)
        )
        
        return cleaned

    def analyze(self, 
                mask: np.ndarray,
                intensity_image: Optional[np.ndarray] = None,
                nucleus_mask: Optional[np.ndarray] = None) -> GolgiFeatures:
        """
        Analyze Golgi apparatus morphology and distribution.
        
        Args:
            mask: Binary mask of segmented Golgi
            intensity_image: Golgi channel intensity image
            nucleus_mask: Optional nucleus mask for relative positioning
            
        Returns:
            GolgiFeatures containing analysis results
        """
        # Get basic features
        basic_features = self.extract_basic_features(mask, intensity_image)
        features = GolgiFeatures(**basic_features.__dict__)
        
        if not np.any(mask):
            return features

        # Analyze fragmentation
        frag_features = self._analyze_fragmentation(mask)
        features.fragment_count = frag_features['fragment_count']
        features.mean_fragment_size = frag_features['mean_size']
        features.size_variance = frag_features['size_variance']
        
        # Analyze structure
        struct_features = self._analyze_structure(mask)
        features.compactness = struct_features['compactness']
        features.polarization_index = struct_features['polarization']
        features.dispersion = struct_features['dispersion']
        features.ribbon_continuity = struct_features['ribbon_continuity']
        
        # Analyze position relative to nucleus if nucleus mask is provided
        if nucleus_mask is not None:
            pos_features = self._analyze_nuclear_position(mask, nucleus_mask)
            features.centroid_distance_to_nucleus = pos_features['distance']
            features.orientation_relative_to_nucleus = pos_features['orientation']
        
        # Analyze morphological features
        morph_features = self._analyze_morphology(mask, intensity_image)
        features.cisternal_length = morph_features['cisternal_length']
        features.stack_thickness = morph_features['stack_thickness']
        features.stack_organization = morph_features['stack_organization']
        
        return features

    def _analyze_fragmentation(self, mask: np.ndarray) -> Dict[str, float]:
        """Analyze Golgi fragmentation patterns."""
        # Label individual fragments
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        
        if not regions:
            return {
                'fragment_count': 0,
                'mean_size': 0,
                'size_variance': 0
            }
        
        # Calculate fragment sizes
        sizes = [r.area for r in regions]
        
        return {
            'fragment_count': len(regions),
            'mean_size': np.mean(sizes),
            'size_variance': np.var(sizes) if len(sizes) > 1 else 0
        }

    def _analyze_structure(self, mask: np.ndarray) -> Dict[str, float]:
        """Analyze Golgi structural organization."""
        if not np.any(mask):
            return {
                'compactness': 0,
                'polarization': 0,
                'dispersion': 0,
                'ribbon_continuity': 0
            }
        
        # Get regions for analysis
        regions = measure.regionprops(mask.astype(int))
        if not regions:
            return {
                'compactness': 0,
                'polarization': 0,
                'dispersion': 0,
                'ribbon_continuity': 0
            }
        
        # Calculate compactness using convex hull
        hull_area = regions[0].convex_area
        actual_area = regions[0].area
        compactness = actual_area / hull_area if hull_area > 0 else 0
        
        # Calculate polarization using angular distribution
        centroids = [r.centroid for r in regions]
        if len(centroids) > 1:
            center = np.mean(centroids, axis=0)
            angles = np.arctan2(
                [c[0] - center[0] for c in centroids],
                [c[1] - center[1] for c in centroids]
            )
            # Measure angular variance as polarization indicator
            angular_variance = np.var(angles)
            polarization = 1 - (angular_variance / (np.pi ** 2))
        else:
            polarization = 0
        
        # Calculate dispersion using mean distance between fragments
        if len(centroids) > 1:
            distances = spatial.distance.pdist(centroids)
            dispersion = np.mean(distances)
        else:
            dispersion = 0
        
        # Analyze ribbon continuity using skeleton
        skeleton = morphology.skeletonize(mask)
        # Count number of endpoints in skeleton
        endpoints = self._count_endpoints(skeleton)
        ribbon_continuity = 1 - (endpoints / (2 * len(regions))) if len(regions) > 0 else 0
        
        return {
            'compactness': compactness,
            'polarization': polarization,
            'dispersion': dispersion,
            'ribbon_continuity': ribbon_continuity
        }

    def _analyze_nuclear_position(self, 
                                mask: np.ndarray, 
                                nucleus_mask: np.ndarray) -> Dict[str, float]:
        """Analyze Golgi position relative to nucleus."""
        if not np.any(mask) or not np.any(nucleus_mask):
            return {'distance': 0, 'orientation': 0}
        
        # Get centroids
        golgi_centroid = self.get_centroid(mask)
        nucleus_centroid = self.get_centroid(nucleus_mask)
        
        # Calculate distance
        distance = np.sqrt(
            (golgi_centroid[0] - nucleus_centroid[0])**2 +
            (golgi_centroid[1] - nucleus_centroid[1])**2
        )
        
        # Calculate orientation angle
        orientation = np.arctan2(
            golgi_centroid[0] - nucleus_centroid[0],
            golgi_centroid[1] - nucleus_centroid[1]
        )
        
        return {
            'distance': distance,
            'orientation': orientation
        }

    def _analyze_morphology(self, 
                          mask: np.ndarray,
                          intensity_image: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Analyze detailed Golgi morphological features."""
        if not np.any(mask):
            return {
                'cisternal_length': 0,
                'stack_thickness': 0,
                'stack_organization': 0
            }
        
        # Get skeleton for cisternal length
        skeleton = morphology.skeletonize(mask)
        cisternal_length = np.sum(skeleton)
        
        # Estimate stack thickness using distance transform
        distance = ndimage.distance_transform_edt(mask)
        stack_thickness = np.mean(distance[skeleton > 0]) * 2 if np.any(skeleton) else 0
        
        # Analyze stack organization using intensity distribution if available
        if intensity_image is not None:
            # Get intensity profile along the stack
            masked_intensity = intensity_image * mask
            if np.any(masked_intensity):
                # Calculate coefficient of variation as measure of stack organization
                intensity_cv = np.std(masked_intensity[mask > 0]) / np.mean(masked_intensity[mask > 0])
                stack_organization = 1 - np.clip(intensity_cv, 0, 1)
            else:
                stack_organization = 0
        else:
            stack_organization = 0
        
        return {
            'cisternal_length': cisternal_length,
            'stack_thickness': stack_thickness,
            'stack_organization': stack_organization
        }

    def _count_endpoints(self, skeleton: np.ndarray) -> int:
        """Count the number of endpoints in a skeleton."""
        # Generate structure for finding neighbors
        struct = ndimage.generate_binary_structure(2, 2)
        
        # Count neighbors for each point
        neighbors = ndimage.convolve(skeleton.astype(int), struct) * skeleton
        
        # Endpoints have exactly one neighbor
        return np.sum(neighbors == 2)  # 2 because point counts itself