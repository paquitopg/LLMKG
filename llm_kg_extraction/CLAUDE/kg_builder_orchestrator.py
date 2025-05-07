import os
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from enhanced_kg_builder import FinancialKGBuilder, IncrementalKGBuilder, KGEvaluator
from kg_cache_manager import KGSQLiteCache, KGOutputManager, MemoryEfficientKGBuilder, ChunkedTextProcessor
from llm_client import AzureOpenAIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kg_builder.log')
    ]
)
logger = logging.getLogger('KGBuilderOrchestrator')

class KGBuilderOrchestrator:
    """
    Orchestrates the knowledge graph building process.
    Provides a unified interface for different extraction approaches.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the orchestrator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.project_name = config['project_name']
        self.model_name = config['model_name']
        self.deployment_name = config['deployment_name']
        
        # Set up base directories
        self.base_dir = Path(__file__).resolve().parents[3]
        self.output_dir = self.base_dir / "outputs" / self.project_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up cache
        self.cache_dir = self.base_dir / "cache" / self.project_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if config.get('cache_type', 'sqlite') == 'sqlite':
            self.cache = KGSQLiteCache(self.cache_dir / f"{self.project_name}_cache.db")
        else:
            # Fall back to disk cache
            from kg_cache_manager import KGDiskCache
            self.cache = KGDiskCache(self.cache_dir)
        
        # Initialize LLM client
        self.llm = AzureOpenAIClient(model_name=self.model_name)
        
        # Initialize output manager
        self.output_manager = KGOutputManager(self.output_dir, self.project_name)
        
        # Initialize text processor
        chunk_size = config.get('chunk_size', 4000)
        overlap = config.get('overlap', 500)
        self.text_processor = ChunkedTextProcessor(chunk_size, overlap)
        
        logger.info(f"KGBuilderOrchestrator initialized for project '{self.project_name}'")
    
    def select_builder(self) -> object:
        """
        Select and initialize the appropriate builder based on configuration.
        
        Returns:
            Initialized builder object
        """
        construction_mode = self.config.get('construction_mode', 'iterative')
        memory_efficient = self.config.get('memory_efficient', False)
        resumable = self.config.get('resumable', False)
        
        if memory_efficient:
            logger.info("Using memory-efficient builder")
            return MemoryEfficientKGBuilder(self.cache, self.output_manager, self.llm)
        elif resumable:
            logger.info(f"Using incremental builder with {construction_mode} mode")
            return IncrementalKGBuilder(
                model_name=self.model_name,
                deployment_name=self.deployment_name,
                project_name=self.project_name,
                construction_mode=construction_mode,
                max_workers=self.config.get('max_workers', 4),
                batch_size=self.config.get('batch_size', 3),
                use_cache=self.config.get('use_cache', True)
            )
        else:
            logger.info(f"Using standard builder with {construction_mode} mode")
            return FinancialKGBuilder(
                model_name=self.model_name,
                deployment_name=self.deployment_name,
                project_name=self.project_name,
                construction_mode=construction_mode,
                max_workers=self.config.get('max_workers', 4),
                batch_size=self.config.get('batch_size', 3),
                use_cache=self.config.get('use_cache', True)
            )
    
    def run(self) -> Dict:
        """
        Run the knowledge graph extraction process.
        
        Returns:
            The final knowledge graph
        """
        start_time = time.time()
        logger.info(f"Starting knowledge graph extraction for project '{self.project_name}'")
        
        # Select and initialize the appropriate builder
        builder = self.select_builder()
        
        # Build the knowledge graph
        graph = builder.build_knowledge_graph_from_pdf()
        
        # Save the knowledge graph
        if hasattr(builder, 'save_knowledge_graph'):
            builder.save_knowledge_graph(graph)
        else:
            self.output_manager.save_complete_graph(graph)
        
        # Calculate and log metrics
        evaluator = KGEvaluator(self.llm)
        metrics = evaluator.calculate_metrics(graph)
        
        logger.info(f"Knowledge graph extraction completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Entity count: {metrics['entity_count']}")
        logger.info(f"Relationship count: {metrics['relationship_count']}")
        logger.info(f"Graph density: {metrics['density']:.2f}")
        logger.info(f"Completeness score: {metrics['completeness']:.2f}")
        
        return graph
    
    def compare_modes(self, modes: List[str]) -> Dict:
        """
        Run extraction with multiple modes and compare results.
        
        Args:
            modes: List of construction modes to compare
            
        Returns:
            Dictionary with comparison metrics
        """
        results = {}
        comparisons = {}
        
        # Initialize evaluator
        evaluator = KGEvaluator(self.llm)
        
        # Run extraction with each mode
        for mode in modes:
            logger.info(f"Running extraction with '{mode}' mode")
            
            # Update config for this run
            self.config['construction_mode'] = mode
            
            # Run extraction
            start_time = time.time()
            builder = self.select_builder()
            graph = builder.build_knowledge_graph_from_pdf()
            elapsed = time.time() - start_time
            
            # Calculate metrics
            metrics = evaluator.calculate_metrics(graph)
            metrics['elapsed_seconds'] = elapsed
            
            # Save results
            results[mode] = {
                'graph': graph,
                'metrics': metrics
            }
            
            # Save graph
            output_file = self.output_dir / f"knowledge_graph_{self.project_name}_{self.model_name}_{mode}.json"
            with open(output_file, 'w') as f:
                import json
                json.dump(graph, f, indent=2)
        
        # Generate comparisons between modes
        for i, mode1 in enumerate(modes):
            for mode2 in modes[i+1:]:
                comparison = evaluator.compare_graphs(
                    results[mode1]['graph'],
                    results[mode2]['graph']
                )
                comparisons[f"{mode1}_vs_{mode2}"] = comparison
        
        # Save comparison report
        comparison_file = self.output_dir / f"mode_comparison_{self.project_name}.json"
        with open(comparison_file, 'w') as f:
            import json
            json.dump({
                'mode_metrics': {mode: results[mode]['metrics'] for mode in modes},
                'comparisons': comparisons
            }, f, indent=2)
        
        return {
            'results': results,
            'comparisons': comparisons
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Build a financial knowledge graph from PDF documents')
    
    parser.add_argument('--project', required=True, help='Project name')
    parser.add_argument('--model', default='gpt-4', help='Model name')
    parser.add_argument('--deployment', required=True, help='Deployment name in Azure')
    parser.add_argument('--mode', default='iterative', choices=['iterative', 'onego', 'batch', 'parallel', 'streaming'],
                        help='Construction mode')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for parallel processing')
    parser.add_argument('--batch-size', type=int, default=3, help='Batch size for processing')
    parser.add_argument('--memory-efficient', action='store_true', help='Use memory-efficient processing')
    parser.add_argument('--resumable', action='store_true', help='Use resumable processing')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--chunk-size', type=int, default=4000, help='Text chunk size')
    parser.add_argument('--overlap', type=int, default=500, help='Overlap between chunks')
    parser.add_argument('--compare-modes', nargs='+', help='Compare multiple construction modes')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    config = {
        'project_name': args.project,
        'model_name': args.model,
        'deployment_name': args.deployment,
        'construction_mode': args.mode,
        'max_workers': args.workers,
        'batch_size': args.batch_size,
        'memory_efficient': args.memory_efficient,
        'resumable': args.resumable,
        'use_cache': not args.no_cache,
        'chunk_size': args.chunk_size,
        'overlap': args.overlap
    }
    
    orchestrator = KGBuilderOrchestrator(config)
    
    if args.compare_modes:
        orchestrator.compare_modes(args.compare_modes)
    else:
        orchestrator.run()


if __name__ == "__main__":
    main()
