import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import (
    filters, morphology, measure, segmentation, feature, 
    exposure, restoration, util, draw
)
from skimage.filters import frangi
from skimage.segmentation import watershed, clear_border
from skimage.feature import peak_local_max, corner_harris, corner_peaks
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, NamedTuple, Any
from dataclasses import dataclass
from enum import Enum
import time
from joblib import Parallel, delayed
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import json
from skimage import io
import warnings
warnings.filterwarnings('ignore')

# Configuration classes
class ChannelConfig(NamedTuple):
    index: int
    name: str
    expected_min_intensity: float
    expected_max_intensity: float
    min_snr: float

@dataclass
class HUVECConfig:
    # Channel configurations
    channels: Dict[str, ChannelConfig] = None
    
    # Cell morphology parameters
    min_cell_size: int = 1000
    max_cell_size: int = 10000
    typical_cell_diameter: int = 100
    
    # Nuclear parameters
    min_nucleus_size: int = 100
    max_nucleus_size: int = 1000
    
    # Quality control
    min_snr: float = 5.0
    max_cell_overlap: float = 0.2
    min_cell_intensity: float = 0.1
    
    # Processing parameters
    use_gpu: bool = False
    n_jobs: int = -1
    
    @classmethod
    def default(cls):
        channels = {
            'nucleus': ChannelConfig(0, 'DAPI', 0.0, 1.0, 3.0),  # Relaxed thresholds
            'actin': ChannelConfig(2, 'Phalloidin', 0.0, 1.0, 3.0),
            'golgi': ChannelConfig(4, 'Golgi', 0.0, 1.0, 3.0),
            'mitochondria': ChannelConfig(5, 'MitoTracker', 0.0, 1.0, 3.0)
        }
        return cls(channels=channels)

class QualityMetrics:
    """Quality control metrics for image analysis."""
    
    @staticmethod
    def compute_snr(image: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio."""
        background = np.percentile(image, 10)
        signal = np.mean(image[image > background])
        noise = np.std(image[image <= background])
        return (signal - background) / noise if noise > 0 else 0
    
    @staticmethod
    def compute_focus_metric(image: np.ndarray) -> float:
        """Compute focus quality using variance of Laplacian."""
        return np.var(ndimage.laplace(image))
    
    @staticmethod
    def check_cell_shape(props: Any, config: HUVECConfig) -> bool:
        """Validate cell shape metrics."""
        area = props.area
        perimeter = props.perimeter
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        return (config.min_cell_size <= area <= config.max_cell_size and
                0.1 <= circularity <= 0.9)

class HUVECFeatures:
    """Enhanced feature extraction for HUVEC cells."""
    
    @staticmethod
    def compute_elongation(region: Any) -> float:
        """Compute cell elongation factor."""
        if region.major_axis_length == 0:
            return 0
        return region.minor_axis_length / region.major_axis_length
    
    @staticmethod
    def analyze_stress_fibers(actin_image: np.ndarray) -> Dict:
        """Analyze stress fiber properties in the actin channel."""
        # Enhance edges
        edges = filters.scharr(actin_image)
        
        # Threshold to get fiber mask
        thresh = filters.threshold_otsu(edges)
        fiber_mask = edges > thresh
        
        # Clean up
        fiber_mask = morphology.remove_small_objects(fiber_mask)
        
        # Get properties
        props = measure.regionprops(measure.label(fiber_mask))
        
        return {
            'fiber_density': np.sum(fiber_mask) / fiber_mask.size,
            'fiber_count': len(props),
            'mean_fiber_length': np.mean([p.major_axis_length for p in props]) if props else 0
        }
    
    @staticmethod
    def analyze_cell_junctions(cell_mask: np.ndarray) -> Dict:
        """Analyze cell-cell junctions."""
        # Find cell boundaries
        boundaries = segmentation.find_boundaries(cell_mask)
        
        # Analyze junction points (where 3 or more cells meet)
        junction_points = corner_peaks(
            corner_harris(boundaries), min_distance=3
        )
        
        return {
            'junction_count': len(junction_points),
            'boundary_length': np.sum(boundaries),
            'junction_density': len(junction_points) / np.sum(boundaries) if np.sum(boundaries) > 0 else 0
        }

class EnhancedCellSegmentation:
    """Improved cell segmentation specifically for HUVECs."""
    
    def __init__(self, config: HUVECConfig):
        self.config = config
    
    def segment_cells(self, actin_image: np.ndarray, nuclei_mask: np.ndarray) -> np.ndarray:
        """
        Segment cells using watershed with nuclei as seeds and enhanced membrane detection.
        """
        # Denoise the image
        denoised = restoration.denoise_bilateral(actin_image)
        
        # Enhance edges
        edges = filters.scharr(denoised)
        
        # Create elevation map for watershed
        elevation = filters.gaussian(edges, sigma=2)
        
        # Use nuclei as seeds
        seeds = measure.label(nuclei_mask)
        
        # Perform watershed segmentation
        cell_masks = watershed(elevation, seeds, mask=actin_image > filters.threshold_otsu(actin_image))
        
        # Post-process: remove small objects and fill holes
        cell_masks = morphology.remove_small_objects(cell_masks, min_size=self.config.min_cell_size)
        cell_masks = morphology.remove_small_holes(cell_masks)
        
        return cell_masks
    
    def refine_cell_boundaries(self, cell_masks: np.ndarray, actin_image: np.ndarray) -> np.ndarray:
        """Refine cell boundaries using active contours."""
        refined_masks = np.zeros_like(cell_masks)
        
        for label in np.unique(cell_masks)[1:]:  # Skip background
            cell_mask = cell_masks == label
            
            # Initialize contour from current mask
            contour = measure.find_contours(cell_mask, 0.5)[0]
            
            # Refine using active contour
            refined_contour = segmentation.active_contour(
                actin_image,
                contour,
                alpha=0.015,  # Length weight
                beta=10,      # Smoothness weight
                gamma=0.001   # External force weight
            )
            
            # Convert contour back to mask
            refined_mask = np.zeros_like(cell_mask, dtype=bool)
            rr, cc = draw.polygon(refined_contour[:, 0], refined_contour[:, 1])
            refined_mask[rr, cc] = True
            refined_masks[refined_mask] = label
            
        return refined_masks

class HUVECAnalyzer:
    """Main class for HUVEC cell analysis with enhanced capabilities."""
    
    def __init__(self, config: HUVECConfig = None):
        self.config = config or HUVECConfig.default()
        self.segmenter = EnhancedCellSegmentation(self.config)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger('HUVECAnalyzer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def process_image_set(self, image_paths: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Process a complete set of images."""
        try:
            # Load images
            self.logger.info("Loading images...")
            images = {}
            for channel_name, config in self.config.channels.items():
                img = io.imread(image_paths[config.index])
                images[channel_name] = self.normalize_channel(img)
            
            # Segment
            self.logger.info("Starting segmentation...")
            masks = {}
            masks['nuclei'] = self._segment_nuclei(images['nucleus'])
            masks['cells'] = self.segment_cells(images['actin'], masks['nuclei'])
            masks['golgi'] = self._segment_organelle(images['golgi'], 'golgi')
            masks['mitochondria'] = self._segment_organelle(images['mitochondria'], 'mitochondria')
            
            self.logger.info("Starting feature extraction...")
            features = self.extract_features(images, masks)
            
            self.logger.info("Computing quality metrics...")
            quality = self.compute_quality_metrics(images, masks)
            
            self.logger.info("Analysis complete!")
            return features, quality
                
        except Exception as e:
            self.logger.error(f"Error processing image set: {str(e)}")
            return None, None
    
    def _load_and_validate_images(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """Load and validate all image channels."""
        images = {}
        for channel_name, config in self.config.channels.items():
            try:
                img = io.imread(image_paths[config.index])
                img = exposure.rescale_intensity(img, out_range=(0, 1))
                
                # Validate intensity range
                if not (config.expected_min_intensity <= np.mean(img) <= config.expected_max_intensity):
                    self.logger.warning(f"Channel {channel_name} intensity out of expected range")
                
                # Check SNR
                snr = QualityMetrics.compute_snr(img)
                if snr < config.min_snr:
                    self.logger.warning(f"Low SNR in {channel_name} channel: {snr:.2f}")
                
                images[channel_name] = img
                
            except Exception as e:
                self.logger.error(f"Error loading {channel_name} channel: {str(e)}")
                return None
        
        return images

    def normalize_channel(self, image: np.ndarray) -> np.ndarray:
        """Normalize image channel to [0,1] range."""
        if image.max() == image.min():
            return np.zeros_like(image)
        return (image - image.min()) / (image.max() - image.min())

    def segment_cells(self, actin_image: np.ndarray, nuclei_mask: np.ndarray) -> np.ndarray:
        """Segment cells using the enhanced segmentation instance."""
        return self.segmenter.segment_cells(actin_image, nuclei_mask)

    def _segment_organelle(self, channel: np.ndarray, organelle_name: str) -> np.ndarray:
        """Segment organelles using faster global thresholding."""
        self.logger.info(f"Starting {organelle_name} segmentation...")
        
        try:
            # Normalize and denoise
            self.logger.info(f"{organelle_name}: Normalizing...")
            channel = self.normalize_channel(channel)
            
            self.logger.info(f"{organelle_name}: Smoothing...")
            channel_smooth = filters.gaussian(channel, sigma=1)
            
            # Use faster global Otsu thresholding
            self.logger.info(f"{organelle_name}: Thresholding...")
            thresh = filters.threshold_otsu(channel_smooth)
            binary = channel_smooth > thresh
            
            # Basic cleanup
            self.logger.info(f"{organelle_name}: Cleaning up small objects...")
            binary = morphology.remove_small_objects(binary, min_size=25)
            
            self.logger.info(f"{organelle_name}: Labeling...")
            labeled = measure.label(binary)
            
            self.logger.info(f"{organelle_name} segmentation complete")
            return labeled
            
        except Exception as e:
            self.logger.error(f"Error in {organelle_name} segmentation: {str(e)}")
            return np.zeros_like(channel, dtype=int)

    def _segment_nuclei(self, nuclei_channel: np.ndarray) -> np.ndarray:
        """Segment nuclei using thresholding and watershed."""
        self.logger.info("Segmenting nuclei...")
        
        # Denoise and normalize
        nuclei_channel = self.normalize_channel(nuclei_channel)
        nuclei_smooth = filters.gaussian(nuclei_channel, sigma=1)
        
        # Threshold
        threshold = filters.threshold_otsu(nuclei_smooth)
        binary = nuclei_smooth > threshold
        
        # Clean up
        binary = morphology.remove_small_objects(binary, min_size=self.config.min_nucleus_size)
        binary = morphology.remove_small_holes(binary)
        
        # Separate touching nuclei using watershed
        distance = ndimage.distance_transform_edt(binary)
        local_max = feature.peak_local_max(distance, min_distance=10, labels=binary)
        markers = np.zeros(binary.shape, dtype=int)
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
        markers = morphology.dilation(markers, morphology.disk(3))
        
        nuclei_labels = watershed(-distance, markers, mask=binary)
        
        return nuclei_labels

    def _perform_segmentation(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Perform segmentation of cells and organelles."""
        try:
            # Segment nuclei using dedicated method
            nuclei_mask = self._segment_nuclei(images['nucleus'])
            
            # Segment cells using enhanced method
            cell_masks = self.segmenter.segment_cells(images['actin'], nuclei_mask)
            cell_masks = self.segmenter.refine_cell_boundaries(cell_masks, images['actin'])
            
            # Segment organelles
            organelle_masks = {
                'golgi': self._segment_organelle(images['golgi'], 'golgi'),
                'mitochondria': self._segment_organelle(images['mitochondria'], 'mitochondria')
            }
            
            return {
                'nuclei': nuclei_mask,
                'cells': cell_masks,
                **organelle_masks
            }
            
        except Exception as e:
            self.logger.error(f"Segmentation error: {str(e)}")
            return None
        
    def extract_features(self, images: Dict[str, np.ndarray], 
                            masks: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Extract comprehensive feature set for all cells."""
        self.logger.info("Extracting features for all cells...")
        
        # Process each cell in parallel with limited jobs
        cell_labels = np.unique(masks['cells'])[1:]  # Skip background
        
        # Use context manager to ensure proper cleanup
        with Parallel(n_jobs=2, max_nbytes=None) as parallel:
            features_list = parallel(
                delayed(self._extract_single_cell_features)(
                    cell_label, images, masks
                )
                for cell_label in cell_labels
            )
        
        # Combine all features into a DataFrame
        df = pd.DataFrame(features_list)
        
        # Add population-level statistics
        df = self._add_population_metrics(df)
        
        return df
    
    def _extract_single_cell_features(self, cell_label: int, 
                                    images: Dict[str, np.ndarray],
                                    masks: Dict[str, np.ndarray]) -> Dict:
        """Extract features for a single cell."""
        # Get cell mask
        cell_mask = masks['cells'] == cell_label
        
        # Basic morphological features
        cell_props = measure.regionprops(cell_mask.astype(int), 
                                       intensity_image=images['actin'])[0]
        
        features = {
            'cell_label': cell_label,
            'area': cell_props.area,
            'perimeter': cell_props.perimeter,
            'eccentricity': cell_props.eccentricity,
            'elongation': HUVECFeatures.compute_elongation(cell_props),
            'mean_intensity': cell_props.mean_intensity,
        }
        
        # Add stress fiber analysis
        actin_features = HUVECFeatures.analyze_stress_fibers(
            images['actin'] * cell_mask
        )
        features.update({f'actin_{k}': v for k, v in actin_features.items()})
        
        # Add junction analysis
        junction_features = HUVECFeatures.analyze_cell_junctions(cell_mask)
        features.update({f'junction_{k}': v for k, v in junction_features.items()})
        
        # Add organelle features
        for organelle in ['golgi', 'mitochondria']:
            organelle_mask = masks[organelle] * cell_mask
            features.update(self._compute_organelle_features(
                organelle_mask, cell_mask, organelle
            ))
        
        return features
    
    def _add_population_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add population-level metrics to the feature set."""
        # Calculate z-scores for key metrics
        for col in ['area', 'elongation', 'actin_fiber_density']:
            df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
        
        # Add percentile ranks
        for col in ['area', 'perimeter', 'mean_intensity']:
            df[f'{col}_percentile'] = df[col].rank(pct=True)
        
        return df
    
    def _compute_quality_metrics(self, images: Dict[str, np.ndarray],
                               masks: Dict[str, np.ndarray]) -> Dict:
        """Compute quality metrics for the analysis."""
        metrics = {}
        
        # Image quality metrics
        for channel_name, img in images.items():
            metrics[f'{channel_name}_snr'] = QualityMetrics.compute_snr(img)
            metrics[f'{channel_name}_focus'] = QualityMetrics.compute_focus_metric(img)
        
        # Segmentation quality metrics
        metrics.update({
            'cell_count': len(np.unique(masks['cells'])) - 1,  # Subtract background
            'nucleus_count': len(np.unique(masks['nuclei'])) - 1,
            'cells_with_nucleus': self._count_cells_with_nucleus(masks['cells'], masks['nuclei']),
            'mean_cell_size': np.mean([r.area for r in measure.regionprops(masks['cells'])]),
            'cell_size_cv': np.std([r.area for r in measure.regionprops(masks['cells'])]) / 
                          np.mean([r.area for r in measure.regionprops(masks['cells'])]),
            'merged_cell_percentage': self._estimate_merged_cells(masks['cells'], masks['nuclei']),
            'fragmented_cell_percentage': self._estimate_fragmented_cells(masks['cells'])
        })
        
        return metrics
    
    def _count_cells_with_nucleus(self, cell_mask: np.ndarray, 
                                nuclei_mask: np.ndarray) -> int:
        """Count cells that contain exactly one nucleus."""
        count = 0
        for cell_label in np.unique(cell_mask)[1:]:  # Skip background
            cell_region = cell_mask == cell_label
            nuclei_in_cell = len(np.unique(nuclei_mask[cell_region])[1:])  # Skip background
            if nuclei_in_cell == 1:
                count += 1
        return count
    
    def _estimate_merged_cells(self, cell_mask: np.ndarray, 
                             nuclei_mask: np.ndarray) -> float:
        """Estimate percentage of potentially merged cells."""
        merged_count = 0
        total_cells = len(np.unique(cell_mask)) - 1
        
        for cell_label in np.unique(cell_mask)[1:]:
            cell_region = cell_mask == cell_label
            nuclei_count = len(np.unique(nuclei_mask[cell_region])[1:])
            if nuclei_count > 1:
                merged_count += 1
                
        return (merged_count / total_cells * 100) if total_cells > 0 else 0
    
    def _estimate_fragmented_cells(self, cell_mask: np.ndarray) -> float:
        """Estimate percentage of potentially fragmented cells."""
        cell_sizes = [r.area for r in measure.regionprops(cell_mask)]
        if not cell_sizes:
            return 0
        
        median_size = np.median(cell_sizes)
        small_cells = sum(1 for size in cell_sizes if size < median_size * 0.3)
        return (small_cells / len(cell_sizes) * 100)

    def _compute_organelle_features(self, organelle_mask: np.ndarray, 
                                cell_mask: np.ndarray, 
                                organelle_name: str) -> Dict:
        """Compute features for an organelle within a cell."""
        features = {}
        
        # Basic measurements
        organelle_props = measure.regionprops(organelle_mask.astype(int))
        
        if not organelle_props:
            # Return zeros if no organelles found
            features.update({
                f'{organelle_name}_count': 0,
                f'{organelle_name}_total_area': 0,
                f'{organelle_name}_mean_size': 0,
                f'{organelle_name}_coverage': 0,
            })
            return features
        
        # Count and size metrics
        features.update({
            f'{organelle_name}_count': len(organelle_props),
            f'{organelle_name}_total_area': sum(prop.area for prop in organelle_props),
            f'{organelle_name}_mean_size': np.mean([prop.area for prop in organelle_props]),
            f'{organelle_name}_coverage': np.sum(organelle_mask) / np.sum(cell_mask),
        })
        
        # Shape metrics (using the largest organelle)
        largest_prop = max(organelle_props, key=lambda x: x.area)
        features.update({
            f'{organelle_name}_max_size': largest_prop.area,
            f'{organelle_name}_eccentricity': largest_prop.eccentricity,
            f'{organelle_name}_perimeter': largest_prop.perimeter,
        })
        
        # Distribution metrics
        if len(organelle_props) > 1:
            centroids = np.array([prop.centroid for prop in organelle_props])
            distances = cdist(centroids, centroids)
            np.fill_diagonal(distances, np.inf)  # Ignore self-distances
            features.update({
                f'{organelle_name}_min_spacing': np.min(distances),
                f'{organelle_name}_mean_spacing': np.mean(distances[distances != np.inf]),
            })
        else:
            features.update({
                f'{organelle_name}_min_spacing': 0,
                f'{organelle_name}_mean_spacing': 0,
            })
        
        return features

    def _visualize_stress_fibers(self, actin_image: np.ndarray, cell_mask: np.ndarray) -> Figure:
        """Visualize stress fiber analysis results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original actin image
        axes[0].imshow(actin_image, cmap='gray')
        axes[0].set_title('Actin Channel')
        
        # Stress fiber detection
        edges = filters.scharr(actin_image)
        thresh = filters.threshold_otsu(edges)
        fiber_mask = edges > thresh
        fiber_mask = morphology.remove_small_objects(fiber_mask)
        
        axes[1].imshow(fiber_mask, cmap='gray')
        axes[1].set_title('Detected Stress Fibers')
        
        plt.tight_layout()
        return fig

    def _visualize_organelle_distribution(self, images: Dict[str, np.ndarray], 
                                        masks: Dict[str, np.ndarray]) -> Figure:
        """Visualize organelle distribution within cells."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        # Cell boundaries with nuclei
        boundaries = segmentation.find_boundaries(masks['cells'])
        composite = np.zeros((*masks['cells'].shape, 3))
        composite[..., 0] = boundaries  # Red for cell boundaries
        composite[..., 2] = masks['nuclei'] > 0  # Blue for nuclei
        axes[0].imshow(composite)
        axes[0].set_title('Cells and Nuclei')
        
        # Mitochondria distribution
        axes[1].imshow(images['mitochondria'], cmap='gray')
        axes[1].imshow(masks['mitochondria'] > 0, alpha=0.3, cmap='spring')
        axes[1].set_title('Mitochondria Distribution')
        
        # Golgi distribution
        axes[2].imshow(images['golgi'], cmap='gray')
        axes[2].imshow(masks['golgi'] > 0, alpha=0.3, cmap='autumn')
        axes[2].set_title('Golgi Distribution')
        
        # Combined organelles
        combined = np.zeros((*masks['cells'].shape, 3))
        combined[..., 0] = masks['golgi'] > 0  # Red for Golgi
        combined[..., 1] = masks['mitochondria'] > 0  # Green for mitochondria
        combined[..., 2] = masks['nuclei'] > 0  # Blue for nuclei
        axes[3].imshow(combined)
        axes[3].set_title('Combined Organelle Distribution')
        
        plt.tight_layout()
        return fig

    def visualize_results(self, images: Dict[str, np.ndarray],
                         masks: Dict[str, np.ndarray],
                         features_df: pd.DataFrame) -> Dict[str, Figure]:
        """Generate comprehensive visualization of analysis results."""
        figures = {}
        
        # Segmentation overlay
        figures['segmentation'] = self._create_segmentation_overlay(images, masks)
        
        # Feature distributions
        figures['feature_distributions'] = self._plot_feature_distributions(features_df)
        
        # Stress fiber analysis
        figures['stress_fibers'] = self._visualize_stress_fibers(
            images['actin'], masks['cells']
        )
        
        # Cell shape analysis
        figures['cell_shapes'] = self._visualize_cell_shapes(masks['cells'])
        
        # Organelle distribution
        figures['organelle_distribution'] = self._visualize_organelle_distribution(
            images, masks
        )
        
        return figures
    
    def _create_segmentation_overlay(self, images: Dict[str, np.ndarray],
                                   masks: Dict[str, np.ndarray]) -> Figure:
        """Create overlay of segmentation results on original images."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        # Original actin with cell boundaries
        boundaries = segmentation.find_boundaries(masks['cells'])
        actin_overlay = exposure.adjust_gamma(images['actin'], 0.5)
        actin_overlay[boundaries] = 1
        axes[0].imshow(actin_overlay, cmap='gray')
        axes[0].set_title('Cell Boundaries')
        
        # Nuclei and organelles
        overlay = np.zeros((*images['nucleus'].shape, 3))
        overlay[..., 0] = masks['nuclei'] > 0  # Blue for nuclei
        overlay[..., 1] = masks['mitochondria'] > 0  # Green for mitochondria
        overlay[..., 2] = masks['golgi'] > 0  # Red for golgi
        axes[1].imshow(overlay)
        axes[1].set_title('Organelle Segmentation')
        
        # Cell classification
        cell_types = self._classify_cell_shapes(masks['cells'])
        cell_type_img = self._create_cell_type_image(masks['cells'], cell_types)
        axes[2].imshow(cell_type_img)
        axes[2].set_title('Cell Shape Classification')
        
        # Quality metrics
        self._plot_quality_metrics(axes[3], masks)
        
        plt.tight_layout()
        return fig
    
    def _classify_cell_shapes(self, cell_mask: np.ndarray) -> Dict[int, str]:
        """Classify cells based on their morphological properties."""
        classifications = {}
        for region in measure.regionprops(cell_mask):
            # Calculate shape features
            elongation = region.minor_axis_length / region.major_axis_length
            circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
            
            # Classify based on shape features
            if elongation < 0.3:
                cell_type = 'elongated'
            elif circularity > 0.8:
                cell_type = 'circular'
            elif 0.3 <= elongation <= 0.7:
                cell_type = 'normal'
            else:
                cell_type = 'irregular'
                
            classifications[region.label] = cell_type
        return classifications
    
    def _create_cell_type_image(self, cell_mask: np.ndarray, 
                               classifications: Dict[int, str]) -> np.ndarray:
        """Create colored image based on cell classifications."""
        colors = {
            'elongated': [1, 0, 0],    # Red
            'circular': [0, 1, 0],     # Green
            'normal': [0, 0, 1],       # Blue
            'irregular': [1, 1, 0]     # Yellow
        }
        
        result = np.zeros((*cell_mask.shape, 3))
        for label, cell_type in classifications.items():
            result[cell_mask == label] = colors[cell_type]
        return result
    
    def _plot_quality_metrics(self, ax: Axes, masks: Dict[str, np.ndarray]) -> None:
        """Plot quality metrics summary."""
        metrics = {
            'Total Cells': len(np.unique(masks['cells'])) - 1,
            'Cells with Nucleus': self._count_cells_with_nucleus(masks['cells'], masks['nuclei']),
            'Merged Cells (%)': self._estimate_merged_cells(masks['cells'], masks['nuclei']),
            'Fragmented Cells (%)': self._estimate_fragmented_cells(masks['cells'])
        }
        
        y_pos = np.arange(len(metrics))
        ax.barh(y_pos, list(metrics.values()))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(metrics.keys()))
        ax.set_title('Quality Metrics')

    def _plot_feature_distributions(self, features_df: pd.DataFrame) -> Figure:
        """Plot distributions of key features."""
        key_features = ['area', 'elongation', 'actin_fiber_density', 
                    'mitochondria_coverage', 'golgi_coverage']
        
        fig, axes = plt.subplots(len(key_features), 1, figsize=(10, 4*len(key_features)))
        
        for ax, feature in zip(axes, key_features):
            if feature in features_df.columns:
                ax.hist(features_df[feature], bins=20)
                ax.set_title(f'{feature.replace("_", " ").title()} Distribution')
                ax.set_xlabel(feature)
                ax.set_ylabel('Count')
        
        plt.tight_layout()
        return fig

    def save_results(self, output_dir: Path, features_df: pd.DataFrame,
                    quality_metrics: Dict, figures: Dict[str, Figure]) -> None:
        """Save analysis results to disk."""
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features_df.to_csv(output_dir / 'cell_features.csv', index=False)
        
        # Save quality metrics
        with open(output_dir / 'quality_metrics.json', 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        
        # Save figures
        for name, fig in figures.items():
            fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Save analysis log
        self.logger.info(f"Results saved to {output_dir}")

# Usage example
if __name__ == "__main__":
    config = HUVECConfig.default()
    analyzer = HUVECAnalyzer(config)
    
    # Example image paths
    image_paths = [
        "path/to/nucleus_channel.tif",
        "path/to/membrane_channel.tif",
        "path/to/actin_channel.tif",
        "path/to/other_channel.tif",
        "path/to/golgi_channel.tif",
        "path/to/mitochondria_channel.tif"
    ]
    
    # Process images
    features_df, quality_metrics = analyzer.process_image_set(image_paths)
    
    # Generate visualizations
    figures = analyzer.visualize_results(images, masks, features_df)
    
    # Save results
    analyzer.save_results(Path("output_directory"), features_df, quality_metrics, figures)
