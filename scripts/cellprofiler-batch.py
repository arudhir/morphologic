import os
import subprocess
from pathlib import Path
import logging

def setup_logging():
    """configure logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cellprofiler_batch.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_cellprofiler(pipeline_path, input_dir, output_dir, logger):
    """
    run cellprofiler in headless mode for a single dataset
    
    Args:
        pipeline_path (str): path to the .cppipe file
        input_dir (str): input directory with images
        output_dir (str): output directory for results
        logger (logging.Logger): logger instance
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            'cellprofiler',
            '-c',                   
            '-r',                   # to run headlessly
            '-p', pipeline_path,    
            '-i', input_dir,        
            '-o', output_dir,       
            '--conserve-memory'     # memory optimization
        ]
        
        logger.info(f"Starting CellProfiler analysis for {input_dir}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        
        rc = process.poll()
        if rc != 0:
            error = process.stderr.read()
            logger.error(f"CellProfiler process failed with return code {rc}")
            logger.error(f"Error message: {error}")
            raise subprocess.CalledProcessError(rc, cmd, error)
            
        logger.info(f"Successfully completed analysis for {input_dir}")
        
    except Exception as e:
        logger.error(f"Error processing {input_dir}: {str(e)}")
        raise

def process_all_datasets(base_dir, pipeline_path):
    """
    process all datasets (EGFR KO, control, & drug datasets)
    
    Args:
        base_dir (str): base directory with all datasets
        pipeline_path (str): path to the updated pipeline file
    """
    logger = setup_logging()
    
    
    datasets = {
        'egfr': ['ko', 'control'],
        'drug': ['dataset1', 'dataset2', 'dataset3']  # Modify it based on actual drug dataset names - i'm not familiar with the names
    }
    
    for study_type, conditions in datasets.items():
        for condition in conditions:
            input_dir = os.path.join(base_dir, study_type, condition, 'input')
            output_dir = os.path.join(base_dir, study_type, condition, 'output')
            
            if not os.path.exists(input_dir):
                logger.warning(f"Input directory not found: {input_dir}")
                continue
                
            try:
                run_cellprofiler(pipeline_path, input_dir, output_dir, logger)
            except Exception as e:
                logger.error(f"Failed to process {study_type}/{condition}: {str(e)}")
                continue

if __name__ == "__main__":
    # config
    BASE_DIR = "/path/to/your/data"  # modify this path
    PIPELINE_PATH = "/path/to/updated/pipeline.cppipe"  # modify this path
    
    try:
        process_all_datasets(BASE_DIR, PIPELINE_PATH)
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
