from cell_analyzer.core.analyzer import CellAnalyzer
from cell_analyzer.core.config import load_config

def main():
    config = load_config('config.yaml')
    analyzer = CellAnalyzer(config)
    
    results = analyzer.process_directory('input_dir')
    analyzer.save_results(results, 'output_dir')

if __name__ == '__main__':
    main()