#!/usr/bin/env python
import argparse
from pathlib import Path
import logging
import sys
import pandas as pd
import json
from typing import List, Dict, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

try:
    from huvec_analyzer import HUVECAnalyzer, HUVECConfig
except ImportError as e:
    print(f"Error importing HUVECAnalyzer: {e}")
    print("Please ensure huvec_analyzer.py is in the same directory as cli.py")
    sys.exit(1)

class AnalysisCLI:
    def __init__(self):
        self.logger = self._setup_logger()
        self.config = HUVECConfig.default()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging with both file and console output."""
        logger = logging.getLogger('HUVECAnalysisCLI')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_file_logger(self, output_dir: Path):
        """Add file handler to logger."""
        file_handler = logging.FileHandler(output_dir / 'analysis.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
    
    def find_image_sets(self, base_dir: Path) -> List[Dict[str, Path]]:
        """Find all image sets in the directory structure."""
        image_sets = []
        
        # Find all w1 files as anchors for image sets
        for w1_file in base_dir.rglob('*_s1_w1.png'):
            dir_path = w1_file.parent
            
            # Check if all w1-w6 files exist in this directory
            w_files = {}
            valid_set = True
            for i in range(1, 7):
                # Get the corresponding w file using the same prefix as w1
                prefix = w1_file.name.replace('_s1_w1.png', '')
                w_file = dir_path / f"{prefix}_s1_w{i}.png"
                if not w_file.exists():
                    valid_set = False
                    break
                w_files[f'w{i}'] = w_file
            
            if valid_set:
                image_sets.append({
                    'directory': dir_path,
                    'files': w_files,
                    'set_id': dir_path.name
                })
        
        return image_sets
    
    def process_image_set(self, image_set: Dict) -> Optional[Dict]:
        """Process a single image set."""
        try:
            # Create output directory
            output_dir = image_set['directory'] / 'analysis_results'
            output_dir.mkdir(exist_ok=True)
            
            # Initialize analyzer
            analyzer = HUVECAnalyzer(self.config)
            
            # Prepare image paths in correct order
            image_paths = [str(image_set['files'][f'w{i}']) for i in range(1, 7)]
            
            # Process images
            features_df, quality_metrics = analyzer.process_image_set(image_paths)
            
            if features_df is None:
                self.logger.error(f"Failed to process {image_set['set_id']}")
                return None
            
            # Generate visualizations
            figures = analyzer.visualize_results(images, masks, features_df)
            
            # Save results
            analyzer.save_results(output_dir, features_df, quality_metrics, figures)
            
            return {
                'set_id': image_set['set_id'],
                'status': 'success',
                'cell_count': len(features_df),
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {image_set['set_id']}: {str(e)}")
            return {
                'set_id': image_set['set_id'],
                'status': 'failed',
                'error': str(e)
            }
    
    def run_batch_analysis(self, base_dir: Path, num_workers: int = None):
        """Run analysis on all image sets in the directory."""
        self.logger.info(f"Analyzing directory: {base_dir}")
        
        # Find image sets
        image_sets = self.find_image_sets(base_dir)
        self.logger.info(f"Found {len(image_sets)} image sets")
        
        if not image_sets:
            self.logger.error("No valid image sets found")
            return
        
        # Create output directory
        output_base = base_dir / 'analysis_output'
        output_base.mkdir(exist_ok=True)
        
        # Process image sets
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_image_set, image_set) 
                    for image_set in image_sets]
            
            for future in tqdm(futures, desc="Processing image sets"):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process: {str(e)}")
                    results.append({
                        'set_id': 'unknown',
                        'status': 'failed',
                        'error': str(e)
                    })
        
        if results:
            # Save summary
            summary_df = pd.DataFrame(results)
            summary_df.to_csv(output_base / 'analysis_summary.csv', index=False)
            
            # Calculate success rate only if we have a status column
            if 'status' in summary_df.columns:
                success_rate = (summary_df['status'] == 'success').mean() * 100
                self.logger.info(f"Success rate: {success_rate:.1f}%")
        
        self.logger.info(f"Analysis complete. Processed {len(results)} image sets")
        
    def _generate_summary_report(self, results: List[Dict], output_dir: Path):
        """Generate summary report of all processed image sets."""
        # Convert results to DataFrame
        summary_df = pd.DataFrame(results)
        
        # Calculate success rate
        success_rate = (summary_df['status'] == 'success').mean() * 100
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_image_sets': len(results),
            'successful_analyses': len(summary_df[summary_df['status'] == 'success']),
            'failed_analyses': len(summary_df[summary_df['status'] == 'failed']),
            'success_rate': success_rate,
            'average_cell_count': summary_df[summary_df['status'] == 'success']['cell_count'].mean()
        }
        
        # Save reports
        summary_df.to_csv(output_dir / 'analysis_summary.csv', index=False)
        with open(output_dir / 'analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='HUVEC Image Analysis CLI')
    parser.add_argument('directory', type=str, help='Base directory containing image sets')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    # Run analysis
    cli = AnalysisCLI()
    cli.run_batch_analysis(Path(args.directory), args.workers)

if __name__ == "__main__":
    main()
