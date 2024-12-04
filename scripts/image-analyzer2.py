#!/usr/bin/env python

from pathlib import Path
import os
import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Dict, Optional
import networkx as nx
import scipy
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
from skimage import io, filters, measure, morphology, segmentation, feature
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.segmentation import clear_border
from skimage.exposure import rescale_intensity
from cellpose import models
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from scipy.ndimage import distance_transform_edt, gaussian_filter
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import argparse

# Recursion's RGB mapping for the six channels
RGB_MAP = {
    1: {
        'rgb': np.array([19, 0, 249]),
        'range': [0, 51]
    },
    2: {
        'rgb': np.array([42, 255, 31]),
        'range': [0, 107]
    },
    3: {
        'rgb': np.array([255, 0, 25]),
        'range': [0, 64]
    },
    4: {
        'rgb': np.array([45, 255, 252]),
        'range': [0, 191]
    },
    5: {
        'rgb': np.array([250, 0, 253]),
        'range': [0, 89]
    },
    6: {
        'rgb': np.array([254, 255, 40]),
        'range': [0, 191]
    }
}

class CellAnalyzer:
    def __init__(self, config: Dict):
        self.config = config


    def _create_network_graph(self, skeleton: np.ndarray) -> nx.Graph:
        """Convert skeleton image to networkx graph for topology analysis."""
        import networkx as nx
        from scipy import ndimage
        
        # Initialize empty graph
        G = nx.Graph()
        
        # Find all endpoints and junction points
        endpoints = []
        junctions = []
        
        # Generate structure for finding neighbors
        struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
        
        # Iterate through all points in skeleton
        for point in np.argwhere(skeleton):
            # Count neighbors
            neighbors = struct * skeleton[
                max(0, point[0]-1):point[0]+2,
                max(0, point[1]-1):point[1]+2
            ]
            num_neighbors = np.sum(neighbors) - 1  # Subtract center point
            
            if num_neighbors == 1:
                endpoints.append(tuple(point))
            elif num_neighbors > 2:
                junctions.append(tuple(point))
        
        # Add all points as nodes
        points = endpoints + junctions
        G.add_nodes_from(points)
        
        # Function to trace path between points
        def trace_path(start, visited):
            current = start
            path = [current]
            visited.add(current)
            
            while True:
                y, x = current
                neighbors = []
                
                # Check 8-connected neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        
                        ny, nx = y + dy, x + dx
                        if (ny, nx) not in visited and skeleton[ny, nx]:
                            neighbors.append((ny, nx))
                
                if not neighbors:
                    break
                    
                current = neighbors[0]
                path.append(current)
                visited.add(current)
                
                # Stop if we hit another endpoint or junction
                if current in points:
                    break
            
            return path
        
        # Connect points with edges
        visited = set()
        for start in points:
            if start not in visited:
                path = trace_path(start, visited)
                if len(path) > 1 and path[-1] in points:
                    G.add_edge(path[0], path[-1], weight=len(path)-1)
        
        return G



    def _compute_golgi_polarization(self, regions: List) -> float:
        """
        Compute Golgi polarization index based on spatial distribution of Golgi fragments.
        
        Args:
            regions: List of region properties from skimage.measure.regionprops
            
        Returns:
            float: Polarization index between 0 (dispersed) and 1 (highly polarized)
        """
        if not regions:
            return 0.0
        
        try:
            # Get centroids of all Golgi fragments
            centroids = np.array([region.centroid for region in regions])
            
            if len(centroids) < 2:
                return 0.0
            
            # Compute center of mass
            center = np.mean(centroids, axis=0)
            
            # Compute distances from center
            distances = np.linalg.norm(centroids - center, axis=1)
            
            # Compute angular positions relative to center
            angles = np.arctan2(centroids[:, 0] - center[0], 
                            centroids[:, 1] - center[1])
            
            # Compute circular variance (measure of angular dispersion)
            mean_angle = np.mean(angles)
            angular_distances = np.abs(angles - mean_angle)
            circular_variance = np.mean(np.min([angular_distances, 
                                            2*np.pi - angular_distances], axis=0))
            
            # Compute spatial concentration
            distance_variance = np.var(distances)
            max_distance = np.max(distances)
            spatial_concentration = 1 - (distance_variance / (max_distance**2))
            
            # Combine measures into polarization index
            polarization_index = (1 - circular_variance/(np.pi/2)) * spatial_concentration
            
            return float(np.clip(polarization_index, 0, 1))
            
        except Exception as e:
            print(f"Error computing Golgi polarization: {str(e)}")
            return 0.0

    def analyze_golgi_morphology(self, golgi_mask: np.ndarray) -> Dict:
        """
        Analyze Golgi apparatus morphology and fragmentation.
        
        Args:
            golgi_mask: Binary mask of Golgi apparatus
            
        Returns:
            Dictionary containing Golgi morphology features
        """
        features = {}
        
        try:
            # Initialize with default values
            default_features = {
                'fragment_count': 0,
                'mean_fragment_size': 0,
                'size_variance': 0,
                'compactness': 0,
                'polarization_index': 0,
                'dispersion': 0,
                'ribbon_continuity': 0
            }
            
            # Return defaults if mask is empty
            if not np.any(golgi_mask):
                return default_features
                
            # Label Golgi fragments
            labels = measure.label(golgi_mask)
            regions = measure.regionprops(labels)
            
            if not regions:
                return default_features
                
            # Basic fragment analysis
            areas = [r.area for r in regions]
            features['fragment_count'] = len(regions)
            features['mean_fragment_size'] = np.mean(areas)
            features['size_variance'] = np.var(areas) if len(areas) > 1 else 0
            
            # Compute compactness
            hull_areas = [r.convex_area for r in regions]
            features['compactness'] = np.mean([a/h for a, h in zip(areas, hull_areas)]) if hull_areas else 0
            
            # Compute polarization
            features['polarization_index'] = self._compute_golgi_polarization(regions)
            
            # Compute dispersion (mean distance between fragments)
            centroids = np.array([r.centroid for r in regions])
            if len(centroids) > 1:
                distances = scipy.spatial.distance.pdist(centroids)
                features['dispersion'] = np.mean(distances)
            else:
                features['dispersion'] = 0
                
            # Analyze Golgi ribbon continuity
            skeleton = morphology.skeletonize(golgi_mask)
            if np.any(skeleton):
                # Count number of endpoints in skeleton
                endpoints = self._count_endpoints(skeleton)
                features['ribbon_continuity'] = 1 - (endpoints / (2 * features['fragment_count']))
            else:
                features['ribbon_continuity'] = 0
                
        except Exception as e:
            print(f"Error in Golgi morphology analysis: {str(e)}")
            features = {
                'fragment_count': 0,
                'mean_fragment_size': 0,
                'size_variance': 0,
                'compactness': 0,
                'polarization_index': 0,
                'dispersion': 0,
                'ribbon_continuity': 0
            }
        
        return features

    def _count_endpoints(self, skeleton: np.ndarray) -> int:
        """
        Count the number of endpoints in a skeleton image.
        
        Args:
            skeleton: Binary skeleton image
            
        Returns:
            int: Number of endpoints
        """
        # Generate structure for finding neighbors
        struct = ndimage.generate_binary_structure(2, 2)
        
        # Count neighbors for each point
        neighbors = ndimage.convolve(skeleton.astype(int), struct) * skeleton
        
        # Endpoints have exactly one neighbor
        endpoints = (neighbors == 2).sum()  # 2 because the point itself is counted
        
        return endpoints    

    def analyze_cell_junctions(self, membrane_mask: np.ndarray, actin_channel: np.ndarray) -> Dict:
        """
        Analyze cell-cell junctions and endothelial barrier properties.
        
        Args:
            membrane_mask: Binary mask of cell membrane
            actin_channel: Actin channel intensity image
            
        Returns:
            Dictionary containing junction features
        """
        features = {}
        
        try:
            # Initialize with default values
            default_features = {
                'junction_linearity': 0.0,
                'junction_continuity': 0.0,
                'junction_actin_intensity': 0.0,
                'junction_length': 0.0,
                'junction_width': 0.0
            }
            
            # Return defaults if inputs are empty
            if not np.any(membrane_mask) or not np.any(actin_channel):
                return default_features
                
            # Ensure membrane mask is binary
            membrane_mask = membrane_mask.astype(bool)
            
            # Detect edges using Sobel
            edges = filters.sobel(membrane_mask.astype(float))
            
            # Normalize edges to 0-1 range
            edges = (edges - edges.min()) / (edges.max() - edges.min())
            
            # Threshold edges
            try:
                edge_threshold = filters.threshold_otsu(edges)
            except ValueError:
                edge_threshold = 0.5
                
            # Create binary edge mask
            edge_mask = (edges > edge_threshold)
            
            # Create potential junction mask
            potential_junctions = edge_mask & membrane_mask
            
            if np.any(potential_junctions):
                # Compute junction linearity
                skeleton = morphology.skeletonize(potential_junctions)
                features['junction_linearity'] = self._compute_junction_linearity(skeleton)
                
                # Compute junction continuity
                features['junction_continuity'] = self._compute_junction_continuity(skeleton)
                
                # Analyze actin at junctions
                junction_actin = actin_channel * potential_junctions
                if np.any(junction_actin):
                    features['junction_actin_intensity'] = np.mean(junction_actin[junction_actin > 0])
                else:
                    features['junction_actin_intensity'] = 0.0
                    
                # Measure junction properties
                features['junction_length'] = np.sum(skeleton)
                features['junction_width'] = np.sum(potential_junctions) / (np.sum(skeleton) + 1e-6)
                
            else:
                return default_features
                
        except Exception as e:
            print(f"Error in junction analysis: {str(e)}")
            return default_features
            
        return features

    def _compute_junction_linearity(self, skeleton: np.ndarray) -> float:
        """
        Compute how linear the junctions are using skeleton analysis.
        
        Args:
            skeleton: Binary skeleton image of junctions
            
        Returns:
            float: Linearity score between 0 (curved) and 1 (linear)
        """
        try:
            if not np.any(skeleton):
                return 0.0
                
            # Label the skeleton
            labels = measure.label(skeleton)
            regions = measure.regionprops(labels)
            
            if not regions:
                return 0.0
                
            # Compute linearity for each junction segment
            linearities = []
            for region in regions:
                if region.area < 2:  # Skip single-pixel regions
                    continue
                    
                # Compare actual length to end-point distance
                coords = region.coords
                if len(coords) >= 2:
                    end_points = coords[[0, -1]]
                    actual_length = region.area
                    direct_length = np.linalg.norm(end_points[0] - end_points[1])
                    
                    # Compute linearity as ratio of direct to actual length
                    linearity = direct_length / actual_length
                    linearities.append(linearity)
            
            # Return average linearity
            return np.mean(linearities) if linearities else 0.0
            
        except Exception as e:
            print(f"Error computing junction linearity: {str(e)}")
            return 0.0

    def _compute_junction_continuity(self, skeleton: np.ndarray) -> float:
        """
        Compute how continuous the junctions are using endpoint analysis.
        
        Args:
            skeleton: Binary skeleton image of junctions
            
        Returns:
            float: Continuity score between 0 (fragmented) and 1 (continuous)
        """
        try:
            if not np.any(skeleton):
                return 0.0
                
            # Find endpoints
            endpoints = self._find_endpoints(skeleton)
            
            # Count total skeleton pixels
            total_pixels = np.sum(skeleton)
            
            if total_pixels == 0:
                return 0.0
                
            # Compute continuity score
            # Fewer endpoints relative to length indicates more continuous junctions
            continuity = 1.0 - (len(endpoints) / (2 * total_pixels + 1e-6))
            
            return np.clip(continuity, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error computing junction continuity: {str(e)}")
            return 0.0

    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find endpoints in a skeleton image.
        
        Args:
            skeleton: Binary skeleton image
            
        Returns:
            List of endpoint coordinates
        """
        # Generate structure for finding neighbors
        struct = ndimage.generate_binary_structure(2, 2)
        
        # Count neighbors
        neighbors = ndimage.convolve(skeleton.astype(int), struct) * skeleton
        
        # Find points with exactly one neighbor (endpoints)
        endpoints = np.where((neighbors == 2) & skeleton)  # 2 because point counts itself
        
        return list(zip(endpoints[0], endpoints[1]))

    def analyze_cell_morphology(self, cell_mask: np.ndarray, actin_channel: np.ndarray) -> Dict:
        """
        Analyze endothelial cell shape and polarity markers.
        
        Args:
            cell_mask: Binary mask of the cell
            actin_channel: Actin channel intensity image
        
        Returns:
            Dictionary containing morphological features
        """
        features = {}
        
        try:
            # Initialize default values
            default_features = {
                'elongation': 1.0,
                'orientation': 0.0,
                'stress_fiber_density': 0.0,
                'protrusion_count': 0,
                'protrusion_length': 0.0,
                'roundness': 0.0,
                'area': 0.0,
                'perimeter': 0.0
            }
            
            # Return defaults if mask is empty
            if not np.any(cell_mask):
                return default_features
                
            # Convert mask to proper format for regionprops
            labeled_mask = measure.label(cell_mask > 0, connectivity=2)
            
            # Get region properties
            regions = measure.regionprops(labeled_mask)
            if not regions:
                return default_features
                
            # Use the largest region if multiple exist
            prop = max(regions, key=lambda x: x.area)
            
            # Basic shape features
            features['elongation'] = prop.major_axis_length / (prop.minor_axis_length + 1e-6)
            features['orientation'] = prop.orientation
            features['area'] = prop.area
            features['perimeter'] = prop.perimeter
            features['roundness'] = (4 * np.pi * prop.area) / (prop.perimeter ** 2 + 1e-6)
            
            # Analyze stress fibers
            stress_fibers = self._detect_stress_fibers(actin_channel, labeled_mask)
            features['stress_fiber_density'] = np.sum(stress_fibers) / (prop.area + 1e-6)
            
            # Analyze protrusions
            protrusions = self._detect_protrusions(labeled_mask)
            features['protrusion_count'] = len(protrusions)
            features['protrusion_length'] = np.mean([p['length'] for p in protrusions]) if protrusions else 0.0
            
        except Exception as e:
            print(f"Error in cell morphology analysis: {str(e)}")
            return default_features
            
        return features

    def _detect_stress_fibers(self, actin_channel: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
        """
        Detect stress fibers using gradient-based analysis.
        
        Args:
            actin_channel: Actin intensity image
            cell_mask: Labeled cell mask
            
        Returns:
            Binary mask of detected stress fibers
        """
        try:
            # Ensure proper types
            actin_channel = actin_channel.astype(np.float32)
            cell_mask = cell_mask > 0
            
            # Apply cell mask to actin channel
            masked_actin = actin_channel * cell_mask
            
            # Enhance linear structures using Sobel filters
            sobel_h = filters.sobel_h(masked_actin)
            sobel_v = filters.sobel_v(masked_actin)
            gradient_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
            
            # Apply Gaussian smoothing to reduce noise
            smoothed = filters.gaussian(gradient_magnitude, sigma=1.0)
            
            # Threshold to detect high gradient regions (potential fibers)
            try:
                thresh = filters.threshold_otsu(smoothed)
            except ValueError:
                thresh = np.mean(smoothed) + np.std(smoothed)
            
            stress_fibers = smoothed > thresh
            
            # Clean up the result
            stress_fibers = stress_fibers & cell_mask
            stress_fibers = morphology.remove_small_objects(stress_fibers, min_size=20)
            stress_fibers = morphology.binary_closing(stress_fibers, morphology.disk(1))
            
            # Skeletonize to get thin fibers
            stress_fibers = morphology.skeletonize(stress_fibers)
            
            return stress_fibers
            
        except Exception as e:
            print(f"Error detecting stress fibers: {str(e)}")
            return np.zeros_like(actin_channel, dtype=bool)

    def _detect_protrusions(self, cell_mask: np.ndarray) -> List[Dict]:
        """
        Detect and measure cell protrusions.
        
        Args:
            cell_mask: Labeled cell mask
            
        Returns:
            List of dictionaries containing protrusion measurements
        """
        try:
            # Ensure proper mask type
            binary_mask = cell_mask > 0
            
            # Get cell boundary
            boundary = segmentation.find_boundaries(binary_mask, mode='outer')
            
            # Get cell centroid
            props = measure.regionprops(cell_mask.astype(int))[0]
            centroid = props.centroid
            
            # Find concave regions
            distance = ndimage.distance_transform_edt(~binary_mask)
            boundary_distance = distance * boundary
            
            # Detect local maxima in boundary distance as potential protrusions
            coordinates = feature.peak_local_max(
                boundary_distance,
                min_distance=10,
                threshold_abs=2,
                exclude_border=False
            )
            
            protrusions = []
            for coord in coordinates:
                # Measure distance from centroid
                length = np.linalg.norm(coord - centroid)
                
                # Only include if significantly far from centroid
                if length > 0.1 * np.sqrt(props.area):
                    protrusions.append({
                        'position': coord,
                        'length': length,
                        'angle': np.arctan2(coord[0] - centroid[0], 
                                        coord[1] - centroid[1])
                    })
            
            return protrusions
            
        except Exception as e:
            print(f"Error detecting protrusions: {str(e)}")
            return []
            
    def segment_cells(self, cell_channel: np.ndarray) -> np.ndarray:
        """Segment cells using Otsu's thresholding and morphological operations."""
        print("Starting cell segmentation...")
        start_time = time.time()
        try:
            thresholded = filters.threshold_otsu(cell_channel)
        except ValueError:
            print("Warning: Unable to compute Otsu threshold. Using a default threshold.")
            thresholded = self.config['default_cell_threshold']
        binary_img = cell_channel > thresholded
        binary_img = morphology.remove_small_objects(binary_img, min_size=self.config['min_cell_size'])
        binary_img = clear_border(binary_img)
        binary_img = morphology.binary_closing(binary_img, morphology.disk(3))
        labeled_cells = measure.label(binary_img)
        print(f"Cell segmentation completed in {time.time() - start_time:.2f} seconds.")
        return labeled_cells

    def segment_organelle(self, channel: np.ndarray, organelle_name: str) -> np.ndarray:
        """Generic function to segment organelles with improved thresholding."""
        print(f"Starting {organelle_name} segmentation...")
        start_time = time.time()
        
        try:
            # Apply gaussian blur to reduce noise
            smoothed = filters.gaussian(channel, sigma=1)
            
            # Use more aggressive thresholding for cleaner segmentation
            thresholded = filters.threshold_otsu(smoothed)
            binary_img = smoothed > thresholded
            
            # Remove small objects and fill holes
            binary_img = morphology.remove_small_objects(
                binary_img, 
                min_size=self.config[f'min_{organelle_name}_size']
            )
            binary_img = morphology.remove_small_holes(binary_img, area_threshold=50)
            
            # Clean up edges
            binary_img = clear_border(binary_img)
            
            labeled_organelle = measure.label(binary_img)
            
        except ValueError as e:
            print(f"Warning: Unable to compute Otsu threshold for {organelle_name}. Using a default threshold.")
            print(f"Error: {str(e)}")
            thresholded = self.config[f'default_{organelle_name}_threshold']
            binary_img = channel > thresholded
            labeled_organelle = measure.label(binary_img)
        
        print(f"{organelle_name.capitalize()} segmentation completed in {time.time() - start_time:.2f} seconds.")
        return labeled_organelle

    def extract_features(self, mask: np.ndarray, intensity_image: np.ndarray, feature_type: str) -> List[Dict]:
        """Extract features for a given mask and intensity image."""
        features = []
        regions = measure.regionprops(mask, intensity_image=intensity_image)
        for region in regions:
            feature = {
                f'{feature_type}_label': int(region.label),
                'cell_label': int(region.label),  # Always include cell_label as integer
                f'{feature_type}_area': region.area,
                f'{feature_type}_perimeter': region.perimeter,
                f'{feature_type}_eccentricity': region.eccentricity,
                f'{feature_type}_mean_intensity': region.mean_intensity,
            }
            if feature_type in ['nucleus', 'cell']:
                feature[f'{feature_type}_texture_entropy'] = self.compute_entropy(region.intensity_image)
            features.append(feature)
        return features

    @staticmethod
    def compute_entropy(intensity_image: np.ndarray) -> float:
        """Compute Shannon entropy of the intensity distribution."""
        if intensity_image.size == 0:
            return 0
        histogram, _ = np.histogram(intensity_image, bins=256, range=(0, 1), density=True)
        histogram = histogram[histogram > 0]
        return -np.sum(histogram * np.log2(histogram))

    def extract_inter_organelle_features(self, cell_masks: np.ndarray, nucleus_masks: np.ndarray, 
                                        mito_masks: np.ndarray, golgi_masks: np.ndarray) -> pd.DataFrame:
        """Extract inter-organelle features using parallel processing."""
        print("Extracting inter-organelle features...")
        start_time = time.time()
        cells = measure.regionprops(cell_masks)
        
        cell_data = [(cell.label, cell.bbox, cell.image) for cell in cells]
        
        print(f"Number of cells for inter-organelle analysis: {len(cell_data)}")
        
        inter_features = Parallel(n_jobs=self.config['n_jobs'])(
            delayed(self.process_cell)(cell_info, nucleus_masks, mito_masks, golgi_masks) 
            for cell_info in cell_data
        )
        
        print(f"Number of processed cells: {len(inter_features)}")
        
        inter_features_df = pd.DataFrame(inter_features)
        print(f"Inter-organelle feature extraction completed in {time.time() - start_time:.2f} seconds.")
        print(f"Columns in inter_features_df: {inter_features_df.columns}")
        return inter_features_df

    def process_cell(self, cell_info: Tuple, nucleus_masks: np.ndarray, 
                     mito_masks: np.ndarray, golgi_masks: np.ndarray) -> Dict:
        """Process a single cell for inter-organelle feature extraction."""
        cell_label, bbox, cell_image = cell_info
        min_row, min_col, max_row, max_col = bbox
        
        nucleus_mask = nucleus_masks[min_row:max_row, min_col:max_col] * cell_image
        mito_mask = mito_masks[min_row:max_row, min_col:max_col] * cell_image
        golgi_mask = golgi_masks[min_row:max_row, min_col:max_col] * cell_image
        
        features = {
            'cell_label': cell_label,
            'mito_distance_to_nucleus': self.compute_centroid_distance(mito_mask, nucleus_mask),
            'golgi_distance_to_nucleus': self.compute_centroid_distance(golgi_mask, nucleus_mask),
            'mito_distance_to_golgi': self.compute_centroid_distance(mito_mask, golgi_mask),
            'mito_golgi_overlap': self.calculate_overlap(mito_mask, golgi_mask),
            'mito_nucleus_overlap': self.calculate_overlap(mito_mask, nucleus_mask),
            'golgi_nucleus_overlap': self.calculate_overlap(golgi_mask, nucleus_mask),
            'manders_mito_golgi': self.co_localization(mito_mask, golgi_mask),
            'pearson_mito_golgi': self.co_localization_pearson(mito_mask, golgi_mask),
            'distance_mito_membrane': self.compute_distance_to_membrane(cell_image, mito_mask),
            'distance_golgi_membrane': self.compute_distance_to_membrane(cell_image, golgi_mask),
            'distance_nucleus_membrane': self.compute_distance_to_membrane(cell_image, nucleus_mask)
        }
        return features

    def process_image(self, image_paths: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Complete image processing pipeline with biological feature extraction."""
        print("Starting image processing pipeline...")
        total_start_time = time.time()
        
        try:
            # Load and normalize images
            channels = [self.normalize_channel(io.imread(path)) for path in image_paths]
            img = np.stack(channels, axis=-1)
            print(f"Images loaded and normalized with shape: {img.shape}")
            
            # Store individual channels for biological analysis
            dapi_channel = channels[0]  # DAPI/Hoechst channel
            actin_channel = channels[2]  # Actin channel
            
            # Segment images (existing code)
            nuclei_masks = self.segment_nuclei(img[:, :, 0])
            cell_masks = self.segment_cells(img[:, :, 2])
            mito_masks = self.segment_organelle(img[:, :, 5], 'mitochondria')
            golgi_masks = self.segment_organelle(img[:, :, 4], 'golgi')
            
            # Extract basic features (existing code)
            nuclear_features = self.extract_features(nuclei_masks, img[:, :, 0], 'nucleus')
            cell_features = self.extract_features(cell_masks, img[:, :, 2], 'cell')
            mito_features = self.extract_features(mito_masks, img[:, :, 5], 'mitochondria')
            golgi_features = self.extract_features(golgi_masks, img[:, :, 4], 'golgi')
            
            # Convert features to DataFrames
            cell_df = pd.DataFrame(cell_features)
            nuclear_df = pd.DataFrame(nuclear_features)
            mito_df = pd.DataFrame(mito_features)
            golgi_df = pd.DataFrame(golgi_features)
            
            # Extract biological features
            biological_df = self.extract_biological_features(
                dapi_channel,
                actin_channel,
                cell_masks,
                nuclei_masks,
                mito_masks,
                golgi_masks
            )
            
            # Merge all features
            features_df = cell_df.merge(nuclear_df, on='cell_label', how='left', suffixes=('', '_nucleus'))
            features_df = features_df.merge(mito_df, on='cell_label', how='left', suffixes=('', '_mito'))
            features_df = features_df.merge(golgi_df, on='cell_label', how='left', suffixes=('', '_golgi'))
            features_df = features_df.merge(biological_df, on='cell_label', how='left')
            
            # Extract inter-organelle features
            inter_features_df = self.extract_inter_organelle_features(
                cell_masks, nuclei_masks, mito_masks, golgi_masks)
            
            # Combine all features
            combined_df = features_df.merge(inter_features_df, on='cell_label', how='left')
            
            print(f"Image processing pipeline completed in {time.time() - total_start_time:.2f} seconds.")
            return combined_df, cell_masks, nuclei_masks, mito_masks, golgi_masks
            
        except Exception as e:
            print(f"An error occurred during image processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None

    @staticmethod
    def compute_centroid_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute the minimum distance between centroids of regions in two masks."""
        props1 = measure.regionprops(measure.label(mask1))
        props2 = measure.regionprops(measure.label(mask2))
        
        if not props1 or not props2:
            return np.nan
        
        centroids1 = [prop.centroid for prop in props1]
        centroids2 = [prop.centroid for prop in props2]
        
        distances = cdist(centroids1, centroids2)
        return np.min(distances) if distances.size > 0 else np.nan

    @staticmethod
    def calculate_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate the percentage overlap between two binary masks."""
        overlap = mask1 & mask2
        return np.sum(overlap) / np.sum(mask1) * 100 if np.sum(mask1) > 0 else 0

    @staticmethod
    def co_localization(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Manders' Overlap Coefficient between two binary masks."""
        overlap = mask1 & mask2
        overlap_area = np.sum(overlap)
        return overlap_area / min(np.sum(mask1), np.sum(mask2)) if min(np.sum(mask1), np.sum(mask2)) > 0 else 0

    @staticmethod
    def co_localization_pearson(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Pearson's Correlation Coefficient between two binary masks."""
        if np.sum(mask1) == 0 or np.sum(mask2) == 0:
            return np.nan
        try:
            pearson_coeff, _ = pearsonr(mask1.flatten(), mask2.flatten())
            return pearson_coeff
        except:
            return np.nan

    @staticmethod
    def compute_distance_to_membrane(cell_mask: np.ndarray, organelle_mask: np.ndarray) -> float:
        """Compute the minimum distance from an organelle region to the cell membrane."""
        distance_map = distance_transform_edt(~cell_mask.astype(bool))
        organelle_coords = np.argwhere(organelle_mask > 0)
        
        if organelle_coords.size == 0:
            return np.nan
        
        distances = distance_map[organelle_mask > 0]
        return np.min(distances) if distances.size > 0 else np.nan

    def extract_biological_features(self, 
                                dapi_channel: np.ndarray,
                                actin_channel: np.ndarray,
                                cell_masks: np.ndarray, 
                                nuclei_masks: np.ndarray,
                                mito_masks: np.ndarray,
                                golgi_masks: np.ndarray) -> pd.DataFrame:
        """Extract comprehensive biological features for each cell."""
        print("Starting biological feature extraction...")
        biological_features = []
        
        # Get unique cell labels
        cell_labels = np.unique(cell_masks)
        cell_labels = cell_labels[cell_labels != 0]  # Remove background
        
        print(f"Processing {len(cell_labels)} cells...")
        
        # Process each cell individually
        for i, cell_label in enumerate(cell_labels):
            if i % 5 == 0:  # Print progress every 5 cells
                print(f"Processing cell {i+1}/{len(cell_labels)}")
                
            try:
                # Create masks for current cell
                current_cell_mask = (cell_masks == cell_label)
                
                # Get bounding box for the cell to reduce computation area
                props = measure.regionprops(current_cell_mask.astype(int))[0]
                bbox = props.bbox
                
                # Extract subregions using bounding box
                cell_nuclei = nuclei_masks[bbox[0]:bbox[2], bbox[1]:bbox[3]] * \
                            current_cell_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cell_mito = mito_masks[bbox[0]:bbox[2], bbox[1]:bbox[3]] * \
                        current_cell_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cell_golgi = golgi_masks[bbox[0]:bbox[2], bbox[1]:bbox[3]] * \
                            current_cell_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cell_actin = actin_channel[bbox[0]:bbox[2], bbox[1]:bbox[3]] * \
                            current_cell_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cell_dapi = dapi_channel[bbox[0]:bbox[2], bbox[1]:bbox[3]] * \
                        current_cell_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                
                # Initialize features dictionary
                features = {'cell_label': cell_label}
                
                # Extract basic shape features first
                print(f"Extracting shape features for cell {i+1}")
                shape_features = {
                    'area': props.area,
                    'perimeter': props.perimeter,
                    'eccentricity': props.eccentricity,
                    'major_axis_length': props.major_axis_length,
                    'minor_axis_length': props.minor_axis_length,
                    'orientation': props.orientation
                }
                features.update(shape_features)
                
                # Extract nuclear features
                print(f"Extracting nuclear features for cell {i+1}")
                nuclear_features = self.analyze_nuclear_morphology(cell_nuclei, cell_dapi)
                features.update(nuclear_features)
                
                # Extract mitochondrial features
                print(f"Extracting mitochondrial features for cell {i+1}")
                mito_features = self.analyze_mitochondrial_network(cell_mito)
                features.update(mito_features)
                
                # Extract Golgi features
                print(f"Extracting Golgi features for cell {i+1}")
                golgi_features = self.analyze_golgi_morphology(cell_golgi)
                features.update(golgi_features)
                
                # Extract cell junction features
                print(f"Extracting junction features for cell {i+1}")
                junction_features = self.analyze_cell_junctions(
                    current_cell_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]],
                    cell_actin
                )
                features.update(junction_features)
                
                biological_features.append(features)
                
            except Exception as e:
                print(f"Error processing cell {cell_label}: {str(e)}")
                continue
        
        print("Biological feature extraction completed.")
        
        if not biological_features:
            print("Warning: No features were extracted successfully.")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['cell_label'])
        
        return pd.DataFrame(biological_features)

    def visualize_segmentation(self, cell_masks: np.ndarray, nuclei_masks: np.ndarray, 
                               mito_masks: np.ndarray, golgi_masks: np.ndarray):
        """Visualize the segmentation results."""
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs[0, 0].imshow(nuclei_masks, cmap='nipy_spectral')
        axs[0, 0].set_title('Nuclei Segmentation')
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(cell_masks, cmap='nipy_spectral')
        axs[0, 1].set_title('Cell Segmentation')
        axs[0, 1].axis('off')
        
        axs[1, 0].imshow(mito_masks, cmap='nipy_spectral')
        axs[1, 0].set_title('Mitochondria Segmentation')
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(golgi_masks, cmap='nipy_spectral')
        axs[1, 1].set_title('Golgi Segmentation')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        return plt.gcf()

    def create_overlay_visualization(self, rgb_image: np.ndarray, 
                                  cell_masks: np.ndarray, 
                                  nuclei_masks: np.ndarray, 
                                  mito_masks: np.ndarray, 
                                  golgi_masks: np.ndarray) -> plt.Figure:
        """
        Create an overlay visualization of organelle masks on the combined RGB image.
        
        Parameters:
        -----------
        rgb_image : np.ndarray
            Combined RGB image of all channels
        *_masks : np.ndarray
            Binary masks for each organelle type
        
        Returns:
        --------
        matplotlib.figure.Figure
        """
        # Set up colors for each organelle type - matching the example image
        colors = {
            'Nucleus': '#FF0000',     # Red
            'Cell Boundary': '#00FF00',  # Bright Green
            'Mitochondria': '#0000FF',   # Blue
            'Golgi': '#FF00FF'        # Magenta
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Display base RGB image
        ax.imshow(rgb_image)
        
        # Refine masks using contour detection
        def get_refined_contours(mask):
            # Clean up the mask first
            mask = mask > 0  # Convert to binary
            mask = morphology.remove_small_objects(mask, min_size=50)  # Remove noise
            # Get contours with a more precise threshold
            contours = measure.find_contours(mask, 0.5)
            # Filter out very small contours
            return [c for c in contours if len(c) > 10]
        
        # Add contours for each organelle type
        contour_width = 0.8  # Thinner lines
        alpha = 1.0  # Full opacity for better visibility
        
        # Cell boundaries
        for contour in get_refined_contours(cell_masks):
            ax.plot(contour[:, 1], contour[:, 0], color=colors['Cell Boundary'], 
                   linewidth=contour_width, alpha=alpha)
        
        # Nuclei
        for contour in get_refined_contours(nuclei_masks):
            ax.plot(contour[:, 1], contour[:, 0], color=colors['Nucleus'], 
                   linewidth=contour_width, alpha=alpha)
        
        # Mitochondria
        for contour in get_refined_contours(mito_masks):
            ax.plot(contour[:, 1], contour[:, 0], color=colors['Mitochondria'], 
                   linewidth=contour_width, alpha=alpha)
        
        # Golgi
        for contour in get_refined_contours(golgi_masks):
            ax.plot(contour[:, 1], contour[:, 0], color=colors['Golgi'], 
                   linewidth=contour_width, alpha=alpha)
        
        # Create legend
        legend_elements = [
            mpatches.Patch(facecolor=color, edgecolor='black', label=label)
            for label, color in colors.items()
        ]
        ax.legend(handles=legend_elements, 
                 loc='center left', 
                 bbox_to_anchor=(1, 0.5),
                 fontsize=12)
        
        # Remove axes and make layout tight
        ax.axis('off')
        plt.tight_layout()
        
        return fig

    def process_single_directory(self, directory_path: Path) -> None:
        """Process a single directory with additional biological visualization."""
        print(f"Processing directory: {directory_path}")
        
        # Check for required images
        image_files = sorted(list(directory_path.glob("*.png")))
        if len(image_files) != 6:
            print(f"Error: Expected 6 images, found {len(image_files)} in {directory_path}")
            return
            
        # Sort files to ensure correct channel order
        image_files.sort()
        
        # Process the images
        results, cell_masks, nuclei_masks, mito_masks, golgi_masks = self.process_image(image_files)
        
        output_path = directory_path / 'output'
        if not output_path.exists():
            output_path.mkdir()

        if results is not None:
            # Save analysis results
            results_path = output_path / 'cell_analysis_results.csv'
            visualization_path = output_path / 'segmentation_visualization.png'
            combined_path = output_path / 'combined_channels.png'
            overlay_path = output_path / 'overlay_visualization.png'
            biological_path = output_path / 'biological_features.png'
            
            # Save CSV results
            results.to_csv(results_path, index=None)
            
            # Save existing visualizations
            fig = self.visualize_segmentation(cell_masks, nuclei_masks, mito_masks, golgi_masks)
            fig.savefig(visualization_path)
            plt.close(fig)
            
            # Create and save biological feature visualizations
            self.visualize_biological_features(
                results,
                output_path / 'nuclear_morphology.png',
                output_path / 'mitochondrial_network.png',
                output_path / 'golgi_structure.png',
                output_path / 'cell_junctions.png'
            )
            
            # Create summary report
            self.create_biological_summary(
                results,
                output_path / 'biological_summary.pdf'
            )
            
            print(f"Analysis complete. Results and visualizations saved to {output_path}")
        else:
            print(f"Image processing failed. Please check the error messages above.")

    def visualize_biological_features(self, results: pd.DataFrame, 
                                    nuclear_path: Path,
                                    mito_path: Path,
                                    golgi_path: Path,
                                    junction_path: Path):
        """Create visualizations for biological features."""
        # Nuclear morphology visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(results['nuclear_irregularity'], 
                    results['chromatin_homogeneity'],
                    c=results['is_mitotic'].astype(int))
        plt.xlabel('Nuclear Irregularity')
        plt.ylabel('Chromatin Homogeneity')
        plt.colorbar(label='Mitotic State')
        plt.savefig(nuclear_path)
        plt.close()

    def create_biological_summary(self, results: pd.DataFrame, output_path: Path):
        """Create a PDF summary of biological features with detailed biological metrics."""
        from fpdf import FPDF
        import numpy as np
        from datetime import datetime
        
        # First, let's print available columns for debugging
        print("Available columns in results:")
        print(results.columns.tolist())
        
        class SummaryPDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'HUVEC Cell Analysis Report', 0, 1, 'C')
                self.ln(5)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        pdf = SummaryPDF()
        pdf.add_page()
        
        # Title and Date
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Biological Feature Summary', 0, 1, 'C')
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 5, f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        pdf.ln(10)
        
        # Population Overview
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '1. Population Overview', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 5, 
            f'Total Cells Analyzed: {len(results)}\n'
            f'Viable Cells: {len(results[results["area"] > 0])}\n'  # Using area instead of nuclear_irregularity
            f'Multi-nucleated Cells: {len(results[results["nucleus_count"] > 1])}', 0)
        pdf.ln(5)
        
        # Nuclear Characteristics
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '2. Nuclear Characteristics', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        # Use area and perimeter from base measurements
        pdf.multi_cell(0, 5,
            f'Average Cell Area: {results["area"].mean():.2f} ± {results["area"].std():.2f} µm²\n'
            f'Mean Cell Perimeter: {results["perimeter"].mean():.2f} ± {results["perimeter"].std():.2f} µm\n'
            f'Average Eccentricity: {results["eccentricity"].mean():.2f}', 0)
        pdf.ln(5)
        
        # Mitochondrial Analysis
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '3. Mitochondrial Network Analysis', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        if 'network_branches' in results.columns:
            pdf.multi_cell(0, 5,
                f'Average Network Branches: {results["network_branches"].mean():.1f}\n'
                f'Mean Fragment Length: {results["mean_fragment_length"].mean():.2f}\n'
                f'Average Fragmentation Index: {results["fragmentation_index"].mean():.2f}', 0)
        else:
            pdf.multi_cell(0, 5, "Mitochondrial network metrics not available", 0)
        pdf.ln(5)
        
        # Golgi Analysis
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '4. Golgi Apparatus Analysis', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        if 'fragment_count' in results.columns:
            pdf.multi_cell(0, 5,
                f'Average Fragment Count: {results["fragment_count"].mean():.1f}\n'
                f'Mean Fragment Size: {results["mean_fragment_size"].mean():.2f}\n'
                f'Average Compactness: {results["compactness"].mean():.2f}', 0)
        else:
            pdf.multi_cell(0, 5, "Golgi apparatus metrics not available", 0)
        pdf.ln(5)
        
        # Cell Junction Analysis
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '5. Cell Junction Analysis', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        if 'junction_continuity' in results.columns:
            pdf.multi_cell(0, 5,
                f'Junction Continuity: {results["junction_continuity"].mean():.2f}\n'
                f'Junction Length: {results["junction_length"].mean():.2f}', 0)
        else:
            pdf.multi_cell(0, 5, "Cell junction metrics not available", 0)
        pdf.ln(5)
        
        # Cell Morphology
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '6. Cell Morphology', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        # Use major_axis_length and minor_axis_length for shape analysis
        if 'major_axis_length' in results.columns:
            elongation = results['major_axis_length'] / results['minor_axis_length']
            pdf.multi_cell(0, 5,
                f'Average Elongation: {elongation.mean():.2f}\n'
                f'Average Orientation: {np.abs(results["orientation"].mean()):.2f} radians', 0)
        else:
            pdf.multi_cell(0, 5, "Cell morphology metrics not available", 0)
        pdf.ln(5)
        
        # Quality Metrics
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '7. Quality Metrics', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        # Calculate quality metrics based on available columns
        valid_cells = len(results[results['area'] > 0])
        quality_score = valid_cells / len(results) if len(results) > 0 else 0
        
        pdf.multi_cell(0, 5,
            f'Valid Cells: {valid_cells} out of {len(results)}\n'
            f'Quality Score: {quality_score:.2f}', 0)
        
        # Save PDF
        pdf.output(str(output_path))

    def _calculate_quality_score(self, results: pd.DataFrame) -> float:
        """Calculate overall quality score based on biological metrics."""
        scores = []
        
        # Nuclear morphology score
        nuclear_score = len(results[results['nuclear_irregularity'] < 2]) / len(results)
        scores.append(nuclear_score)
        
        # Cell size score
        size_score = len(results[(results['cell_area'] > 500) & (results['cell_area'] < 3000)]) / len(results)
        scores.append(size_score)
        
        # Junction integrity score
        junction_score = results['junction_continuity'].mean()
        scores.append(junction_score)
        
        # Mitochondrial network score
        mito_score = 1 - (results['fragmentation_index'].mean() / results['fragmentation_index'].max())
        scores.append(mito_score)
        
        return np.mean(scores)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cell Image Analysis CLI')
    parser.add_argument('input_dir', type=str, help='Directory containing the six input images (*.png)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
    parser.add_argument('--min-cell-size', type=int, default=100, help='Minimum cell size')
    parser.add_argument('--min-nucleus-size', type=int, default=50, help='Minimum nucleus size')
    parser.add_argument('--min-mito-size', type=int, default=10, help='Minimum mitochondria size')
    parser.add_argument('--min-golgi-size', type=int, default=10, help='Minimum golgi size')
    parser.add_argument('--threads', type=int, default=-1, help='Number of threads to use (-1 for all cores)')
    
    args = parser.parse_args()

    # Configure analysis parameters
    config = {
        'use_gpu': args.gpu,
        'min_cell_size': args.min_cell_size,
        'min_nucleus_size': args.min_nucleus_size,
        'min_mitochondria_size': args.min_mito_size,
        'min_golgi_size': args.min_golgi_size,
        'default_cell_threshold': 0.5,
        'default_mitochondria_threshold': 0.5,
        'default_golgi_threshold': 0.5,
        'n_jobs': args.threads
    }

    # Create analyzer instance
    analyzer = CellAnalyzer(config)
    
    # Process the input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return
        
    analyzer.process_single_directory(input_dir)

if __name__ == "__main__":
    main()
