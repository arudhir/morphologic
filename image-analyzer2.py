#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Dict, Optional
from skimage import io, filters, measure, morphology
from skimage.segmentation import clear_border
from skimage.exposure import rescale_intensity
from cellpose import models
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from scipy.ndimage import distance_transform_edt
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

class CellAnalyzer:
    def __init__(self, config: Dict):
        self.config = config

    def normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalize the intensity of a single image channel to the range [0, 1]."""
        return rescale_intensity(channel, in_range='image', out_range=(0, 1))

    def segment_nuclei(self, nuclei_channel: np.ndarray) -> np.ndarray:
        """Segment nuclei using Cellpose."""
        print("Starting nuclei segmentation...")
        start_time = time.time()
        model = models.Cellpose(gpu=self.config['use_gpu'], model_type='nuclei')
        masks, _, _, _ = model.eval(nuclei_channel, diameter=None, channels=[0, 0])
        print(f"Nuclei segmentation completed in {time.time() - start_time:.2f} seconds.")
        return masks

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
        """Generic function to segment organelles."""
        print(f"Starting {organelle_name} segmentation...")
        start_time = time.time()
        try:
            thresholded = filters.threshold_otsu(channel)
        except ValueError:
            print(f"Warning: Unable to compute Otsu threshold for {organelle_name}. Using a default threshold.")
            thresholded = self.config[f'default_{organelle_name}_threshold']
        binary_img = channel > thresholded
        binary_img = morphology.remove_small_objects(binary_img, min_size=self.config[f'min_{organelle_name}_size'])
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

    def process_image(self, image_paths: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Complete image processing pipeline: segmentation, feature extraction, distance calculation."""
        print("Starting image processing pipeline...")
        total_start_time = time.time()
        
        try:
            # Load and normalize images
            channels = [self.normalize_channel(io.imread(path)) for path in image_paths]
            img = np.stack(channels, axis=-1)
            print(f"Images loaded and normalized with shape: {img.shape}")
            
            # Segment images
            nuclei_masks = self.segment_nuclei(img[:, :, 0])
            cell_masks = self.segment_cells(img[:, :, 2])  # Using Actin channel for cell segmentation
            mito_masks = self.segment_organelle(img[:, :, 5], 'mitochondria')
            golgi_masks = self.segment_organelle(img[:, :, 4], 'golgi')
            
            # Extract features
            nuclear_features = self.extract_features(nuclei_masks, img[:, :, 0], 'nucleus')
            cell_features = self.extract_features(cell_masks, img[:, :, 2], 'cell')
            mito_features = self.extract_features(mito_masks, img[:, :, 5], 'mitochondria')
            golgi_features = self.extract_features(golgi_masks, img[:, :, 4], 'golgi')
            
            # Convert features to DataFrames
            cell_df = pd.DataFrame(cell_features)
            nuclear_df = pd.DataFrame(nuclear_features)
            mito_df = pd.DataFrame(mito_features)
            golgi_df = pd.DataFrame(golgi_features)

            # Ensure all DataFrames have 'cell_label' as integer
            for df in [cell_df, nuclear_df, mito_df, golgi_df]:
                if 'cell_label' not in df.columns:
                    label_col = [col for col in df.columns if col.endswith('_label')][0]
                    df['cell_label'] = df[label_col]
                df['cell_label'] = df['cell_label'].astype(int)

            # Print data types for debugging
            for df_name, df in zip(['cell', 'nuclear', 'mito', 'golgi'], [cell_df, nuclear_df, mito_df, golgi_df]):
                print(f"{df_name}_df dtypes:")
                print(df.dtypes)
                print(f"{df_name}_df shape: {df.shape}")

            # Merge features
            features_df = cell_df.merge(nuclear_df, on='cell_label', how='left', suffixes=('', '_nucleus'))
            features_df = features_df.merge(mito_df, on='cell_label', how='left', suffixes=('', '_mito'))
            features_df = features_df.merge(golgi_df, on='cell_label', how='left', suffixes=('', '_golgi'))

            # Extract inter-organelle features
            inter_features_df = self.extract_inter_organelle_features(cell_masks, nuclei_masks, mito_masks, golgi_masks)
            
            # Ensure 'cell_label' in inter_features_df is also integer
            if 'cell_label' in inter_features_df.columns:
                inter_features_df['cell_label'] = inter_features_df['cell_label'].astype(int)
            
            # Combine all features
            combined_df = features_df.merge(inter_features_df, on='cell_label', how='left')
            
            print(f"Image processing pipeline completed in {time.time() - total_start_time:.2f} seconds.")
            return combined_df, cell_masks, nuclei_masks, mito_masks, golgi_masks
        
        except Exception as e:
            print(f"An error occurred during image processing: {str(e)}")
            import traceback
            traceback.print_exc()  # This will print the full traceback for debugging
            return None, None, None, None, None
        
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
    
def address_to_images(image_directory):
    address_to_images = {}
    for treatment in image_directory.iterdir():
        addresses = [f.stem.split('_')[0] for f in list(treatment.glob('*w6.png'))]
        for address in addresses:
            images = [
                image_directory / f'{treatment}/{address}_s1_w1.png',
                image_directory / f'{treatment}/{address}_s1_w2.png',
                image_directory / f'{treatment}/{address}_s1_w3.png',
                image_directory / f'{treatment}/{address}_s1_w4.png',
                image_directory / f'{treatment}/{address}_s1_w5.png',
                image_directory / f'{treatment}/{address}_s1_w6.png',
            ]
            address_to_images[address] = images
    return address_to_images

if __name__ == "__main__":
    config = {
        'use_gpu': False,
        'min_cell_size': 100,
        'min_nucleus_size': 50,
        'min_mitochondria_size': 10,
        'min_golgi_size': 10,
        'default_cell_threshold': 0.5,
        'default_mitochondria_threshold': 0.5,
        'default_golgi_threshold': 0.5,
        'n_jobs': -1  # Use all available cores
    }
    
    analyzer = CellAnalyzer(config)
    
    
    image_directory = Path('/home/ubuntu/EFAAR_benchmarking/notebooks/egfr-images/ko')
    address_to_images = address_to_images(image_directory)
    
    for address, image_paths in address_to_images.items():
        print(f'Processing {address}')
        results, cell_masks, nuclei_masks, mito_masks, golgi_masks = analyzer.process_image(image_paths)
        if results is not None:
            # print(results.head())
            results.to_csv(f'{address}_cell_analysis_results.csv', index=False)
            fig = analyzer.visualize_segmentation(cell_masks, nuclei_masks, mito_masks, golgi_masks)
            fig.savefig(f'{address}_segmentation_visualization.png')
        else:
            print("Image processing failed. Please check the error messages above.")
