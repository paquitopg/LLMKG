import os
import json
import time
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from enhanced_kg_builder import FinancialKGBuilder
from kg_builder_orchestrator import KGBuilderOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('KGBuilderBenchmark')

class KGBuilderBenchmark:
    """
    Benchmark different configurations of the knowledge graph builder.
    Compares performance and quality metrics across different approaches.
    """
    
    def __init__(self, project_name: str, model_name: str, deployment_name: str):
        """
        Initialize the benchmark with project and model information.
        
        Args:
            project_name: Name of the project
            model_name: Name of the model to use
            deployment_name: Name of the deployment in Azure
        """
        self.project_name = project_name
        self.model_name = model_name
        self.deployment_name = deployment_name
        
        self.base_dir = Path(__file__).resolve().parents[3]
        self.results_dir = self.base_dir / "benchmarks" / project_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def run_benchmark(self, configurations: List[Dict]) -> List[Dict]:
        """
        Run benchmarks for multiple configurations.
        
        Args:
            configurations: List of configuration dictionaries
            
        Returns:
            List of benchmark results
        """
        for config in configurations:
            # Add required fields if not present
            config.update({
                'project_name': self.project_name,
                'model_name': self.model_name,
                'deployment_name': self.deployment_name
            })
            
            # Run benchmark for this configuration
            result = self._run_single_benchmark(config)
            self.results.append(result)
        
        # Save and visualize results
        self._save_results()
        self._generate_visualizations()
        
        return self.results
    
    def _run_single_benchmark(self, config: Dict) -> Dict:
        """
        Run a benchmark for a single configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Benchmark result dictionary
        """
        config_name = f"{config['construction_mode']}"
        if config.get('memory_efficient'):
            config_name += "_memory_efficient"
        if config.get('resumable'):
            config_name += "_resumable"
        
        logger.info(f"Running benchmark for configuration: {config_name}")
        
        # Initialize orchestrator
        orchestrator = KGBuilderOrchestrator(config)
        
        # Measure execution time
        start_time = time.time()
        graph = orchestrator.run()
        elapsed = time.time() - start_time
        
        # Calculate memory usage (approximate)
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Get graph metrics
        entity_count = len(graph.get('entities', []))
        relationship_count = len(graph.get('relationships', []))
        
        result = {
            'config': config_name,
            'construction_mode': config['construction_mode'],
            'memory_efficient': config.get('memory_efficient', False),
            'resumable': config.get('resumable', False),
            'batch_size': config.get('batch_size', 1),
            'max_workers': config.get('max_workers', 1),
            'execution_time': elapsed,
            'memory_usage_mb': memory_mb,
            'entity_count': entity_count,
            'relationship_count': relationship_count
        }
        
        logger.info(f"Benchmark for {config_name} completed:")
        logger.info(f"  - Execution time: {elapsed:.2f} seconds")
        logger.info(f"  - Memory usage: {memory_mb:.2f} MB")
        logger.info(f"  - Entities: {entity_count}")
        logger.info(f"  - Relationships: {relationship_count}")
        
        return result
    
    def _save_results(self) -> None:
        """Save benchmark results to CSV and JSON files."""
        # Convert results to DataFrame for easier analysis
        df = pd.DataFrame(self.results)
        
        # Save as CSV
        csv_file = self.results_dir / f"benchmark_results_{self.project_name}.csv"
        df.to_csv(csv_file, index=False)
        
        # Save as JSON
        json_file = self.results_dir / f"benchmark_results_{self.project_name}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {csv_file} and {json_file}")
    
    def _generate_visualizations(self) -> None:
        """Generate visualizations of benchmark results."""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Execution time comparison
        axes[0, 0].bar(df['config'], df['execution_time'])
        axes[0, 0].set_title('Execution Time (seconds)')
        axes[0, 0].set_ylabel('Seconds')
        axes[0, 0].set_xticklabels(df['config'], rotation=45, ha='right')
        
        # Memory usage comparison
        axes[0, 1].bar(df['config'], df['memory_usage_mb'])
        axes[0, 1].set_title('Memory Usage (MB)')
        axes[0, 1].set_ylabel('MB')
        axes[0, 1].set_xticklabels(df['config'], rotation=45, ha='right')
        
        # Entity count comparison
        axes[1, 0].bar(df['config'], df['entity_count'])
        axes[1, 0].set_title('Entity Count')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticklabels(df['config'], rotation=45, ha='right')
        
        # Relationship count comparison
        axes[1, 1].bar(df['config'], df['relationship_count'])
        axes[1, 1].set_title('Relationship Count')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticklabels(df['config'], rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"benchmark_visualization_{self.project_name}.png"
        plt.savefig(plot_file)
        
        logger.info(f"Benchmark visualization saved to {plot_file}")


def create_benchmark_configurations() -> List[Dict]:
    """
    Create a list of configurations to benchmark.
    
    Returns:
        List of configuration dictionaries
    """
    configurations = []
    
    # Base configurations for different modes
    for mode in ['iterative', 'onego', 'batch', 'parallel', 'streaming']:
        configurations.append({
            'construction_mode': mode,
            'use_cache': True
        })
    
    # Memory-efficient configurations
    for mode in ['iterative', 'batch']:
        configurations.append({
            'construction_mode': mode,
            'memory_efficient': True,
            'use_cache': True
        })
    
    # Resumable configurations
    for mode in ['iterative', 'batch']:
        configurations.append({
            'construction_mode': mode,
            'resumable': True,
            'use_cache': True
        })
    
    # Parallel processing with different worker counts
    for workers in [2, 4, 8]:
        configurations.append({
            'construction_mode': 'parallel',
            'max_workers': workers,
            'use_cache': True
        })
    
    # Batch processing with different batch sizes
    for batch_size in [2, 5, 10]:
        configurations.append({
            'construction_mode': 'batch',
            'batch_size': batch_size,
            'use_cache': True
        })
    
    return configurations


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark knowledge graph builder configurations')
    
    parser.add_argument('--project', required=True, help='Project name')
    parser.add_argument('--model', default='gpt-4', help='Model name')
    parser.add_argument('--deployment', required=True, help='Deployment name in Azure')
    parser.add_argument('--custom-configs', type=str, help='Path to custom configuration JSON file')
    
    