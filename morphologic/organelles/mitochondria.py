from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
from scipy import ndimage
from skimage import morphology, filters, measure
from .base import Organelle, OrganelleFeatures
from ..config.channels import ChannelMap

@dataclass
class MitochondrialFeatures(OrganelleFeatures):
    """Mitochondria-specific features for both individual and network properties."""
    # Network properties
    network_branches: int = 0
    network_junctions: int = 0
    mean_branch_length: float = 0.0
    total_network_length: float = 0.0
    network_connectivity: float = 0.0
    
    # Morphological properties
    fragmentation_index: float = 0.0
    mean_fragment_size: float = 0.0
    size_variance: float = 0.0
    
    # Distribution properties
    perinuclear_density: float = 0.0
    peripheral_density: float = 0.0
    spatial_distribution: float = 0.0
    
    # Dynamic properties
    membrane_potential_estimate: float = 0.0
    network_complexity: float = 0.0

class Mitochondria(Organelle):
    """Analyzer for mitochondrial features and network properties."""

    def _set_channel(self) -> None:
        """Set the mitochondrial channel from ChannelMap."""
        self.channel = ChannelMap.MITOCHONDRIA.value

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment mitochondria using adaptive thresholding and morphological operations.
        
        Args:
            image: Mitochondrial channel image
            
        Returns:
            Binary mask of segmented mitochondria
        """
        # Apply Gaussian blur to reduce noise
        smoothed = filters.gaussian(image, sigma=1.0)
        
        # Adaptive thresholding
        thresh = filters.threshold_local(smoothed, block_size=51, method='gaussian')
        binary = smoothed > thresh
        
        # Clean up mask
        cleaned = self.clean_mask(
            binary, 
            self.config.get('min_mitochondria_size', 10)
        )
        
        return cleaned

    def analyze(self, 
                mask: np.ndarray, 
                intensity_image: Optional[np.ndarray] = None) -> MitochondrialFeatures:
        """
        Analyze mitochondrial network and morphology.
        
        Args:
            mask: Binary mask of segmented mitochondria
            intensity_image: Mitochondrial channel intensity image
            
        Returns:
            MitochondrialFeatures containing analysis results
        """
        # Get basic features
        basic_features = self.extract_basic_features(mask, intensity_image)
        features = MitochondrialFeatures(**basic_features.__dict__)
        
        if not np.any(mask) or intensity_image is None:
            return features

        # Create skeleton for network analysis
        skeleton = morphology.skeletonize(mask)
        
        # Analyze network topology
        network_features = self._analyze_network_topology(skeleton)
        features.network_branches = network_features['branches']
        features.network_junctions = network_features['junctions']
        features.mean_branch_length = network_features['mean_branch_length']
        features.total_network_length = network_features['total_length']
        features.network_connectivity = network_features['connectivity']
        
        # Analyze morphology
        morph_features = self._analyze_morphology(mask)
        features.fragmentation_index = morph_features['fragmentation_index']
        features.mean_fragment_size = morph_features['mean_fragment_size']
        features.size_variance = morph_features['size_variance']
        
        # Analyze distribution
        dist_features = self._analyze_spatial_distribution(mask)
        features.perinuclear_density = dist_features['perinuclear_density']
        features.peripheral_density = dist_features['peripheral_density']
        features.spatial_distribution = dist_features['distribution_index']
        
        # Analyze membrane potential (estimated from intensity)
        if intensity_image is not None:
            features.membrane_potential_estimate = self._estimate_membrane_potential(
                intensity_image, mask
            )
        
        # Calculate network complexity
        features.network_complexity = self._calculate_network_complexity(
            skeleton, network_features
        )
        
        return features

    def _analyze_network_topology(self, skeleton: np.ndarray) -> Dict[str, float]:
        """Analyze the topology of the mitochondrial network."""
        # Convert skeleton to graph
        G = self._create_network_graph(skeleton)
        
        # Calculate basic network metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        # Find junction points (nodes with degree > 2)
        junctions = [n for n, d in G.degree() if d > 2]
        
        # Calculate branch lengths
        branch_lengths = [d['weight'] for _, _, d in G.edges(data=True)]
        mean_branch_length = np.mean(branch_lengths) if branch_lengths else 0
        
        # Calculate network connectivity
        if num_nodes > 1:
            connectivity = 2 * num_edges / (num_nodes * (num_nodes - 1))
        else:
            connectivity = 0
        
        return {
            'branches': num_edges,
            'junctions': len(junctions),
            'mean_branch_length': mean_branch_length,
            'total_length': np.sum(branch_lengths),
            'connectivity': connectivity
        }

    def _create_network_graph(self, skeleton: np.ndarray) -> nx.Graph:
        """Convert skeleton to networkx graph."""
        G = nx.Graph()
        
        # Find endpoints and junctions
        struct = ndimage.generate_binary_structure(2, 2)
        neighbors = ndimage.convolve(skeleton.astype(int), struct) * skeleton
        
        # Endpoints have 1 neighbor, junctions have >2 neighbors
        endpoints = np.argwhere(neighbors == 2)  # 2 because center point is counted
        junctions = np.argwhere(neighbors > 3)
        
        # Add all points as nodes
        points = np.vstack([endpoints, junctions])
        G.add_nodes_from(map(tuple, points))
        
        # Connect nodes by tracing paths
        visited = set()
        
        def trace_path(start: Tuple[int, int]) -> List[Tuple[int, int]]:
            """Trace a path from start point to next junction/endpoint."""
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
                if current in map(tuple, points):
                    break
            
            return path
        
        # Connect points with edges
        for start in map(tuple, points):
            if start not in visited:
                path = trace_path(start)
                if len(path) > 1 and path[-1] in map(tuple, points):
                    G.add_edge(path[0], path[-1], weight=len(path)-1)
        
        return G

    def _analyze_morphology(self, mask: np.ndarray) -> Dict[str, float]:
        """Analyze morphological properties of mitochondrial fragments."""
        # Label individual fragments
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        
        if not regions:
            return {
                'fragmentation_index': 0,
                'mean_fragment_size': 0,
                'size_variance': 0
            }
        
        # Calculate fragment sizes
        sizes = [r.area for r in regions]
        
        # Calculate metrics
        total_area = np.sum(sizes)
        mean_size = np.mean(sizes)
        size_variance = np.var(sizes) if len(sizes) > 1 else 0
        
        # Fragmentation index: ratio of number of fragments to total area
        fragmentation_index = len(regions) / total_area if total_area > 0 else 0
        
        return {
            'fragmentation_index': fragmentation_index,
            'mean_fragment_size': mean_size,
            'size_variance': size_variance
        }

    def _analyze_spatial_distribution(self, mask: np.ndarray) -> Dict[str, float]:
        """Analyze spatial distribution of mitochondria."""
        if not np.any(mask):
            return {
                'perinuclear_density': 0,
                'peripheral_density': 0,
                'distribution_index': 0
            }
        
        # Get mask centroid
        cy, cx = self.get_centroid(mask)
        
        # Create distance map from centroid
        y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
        distances = np.sqrt((y - cy)**2 + (x - cx)**2)
        
        # Define regions
        max_dist = np.max(distances[mask])
        perinuclear_mask = distances <= max_dist/3
        peripheral_mask = distances >= 2*max_dist/3
        
        # Calculate densities
        perinuclear_density = (np.sum(mask & perinuclear_mask) / 
                             np.sum(perinuclear_mask)) if np.any(perinuclear_mask) else 0
        
        peripheral_density = (np.sum(mask & peripheral_mask) / 
                            np.sum(peripheral_mask)) if np.any(peripheral_mask) else 0
        
        # Distribution index: ratio of peripheral to perinuclear density
        distribution_index = (peripheral_density / perinuclear_density 
                            if perinuclear_density > 0 else 0)
        
        return {
            'perinuclear_density': perinuclear_density,
            'peripheral_density': peripheral_density,
            'distribution_index': distribution_index
        }

    def _estimate_membrane_potential(self, 
                                   intensity_image: np.ndarray, 
                                   mask: np.ndarray) -> float:
        """
        Estimate membrane potential from intensity distribution.
        Returns normalized estimate between 0 and 1.
        """
        if not np.any(mask):
            return 0.0
        
        # Get masked intensities
        masked_intensities = intensity_image[mask > 0]
        
        if len(masked_intensities) == 0:
            return 0.0
        
        # Calculate metrics that correlate with membrane potential
        mean_intensity = np.mean(masked_intensities)
        intensity_variance = np.var(masked_intensities)
        
        # Normalize to [0, 1] range
        normalized_potential = (mean_intensity * np.sqrt(intensity_variance)) / \
                             (np.max(intensity_image) * np.max(masked_intensities))
        
        return np.clip(normalized_potential, 0, 1)

    def _calculate_network_complexity(self, 
                                    skeleton: np.ndarray, 
                                    network_features: Dict[str, float]) -> float:
        """
        Calculate overall network complexity score.
        Returns value between 0 (simple) and 1 (complex).
        """
        if not np.any(skeleton):
            return 0.0
        
        # Combine multiple metrics for complexity score
        num_branches = network_features['branches']
        num_junctions = network_features['junctions']
        total_length = network_features['total_length']
        
        # Normalize each component
        max_branches = 100  # Expected maximum values for normalization
        max_junctions = 50
        max_length = 1000
        
        normalized_branches = min(num_branches / max_branches, 1)
        normalized_junctions = min(num_junctions / max_junctions, 1)
        normalized_length = min(total_length / max_length, 1)
        
        # Combine metrics with weights
        complexity = (0.4 * normalized_branches + 
                     0.4 * normalized_junctions + 
                     0.2 * normalized_length)
        
        return np.clip(complexity, 0, 1)