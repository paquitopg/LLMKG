
import json
import re
import networkx as nx
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Any, Optional
from ontology_management.ontology_loader import PEKGOntology


class KGEvaluator:
    """
    A class to evaluate a knowledge graph (KG) based on its structure and 
    compliance with a given ontology.
    """
    def __init__(self, kg_json_path: str, ontology_path: str):
        with open(kg_json_path, 'r') as f:
            self.kg_data = json.load(f)

        self.ontology = PEKGOntology(ontology_path)
        self.ontology.transform_for_evaluation()

        self.entities = self.ontology.entities
        self.relations = self.ontology.relations
        self.attributes = self.ontology.attributes

        self.graph = self._create_nx_graph()

    def _create_nx_graph(self) -> nx.DiGraph:
        """
        Create a NetworkX directed graph from the KG data.
        Each node represents an entity, and each edge represents a relationship.
        """
        G = nx.DiGraph()
        for entity in self.kg_data.get('entities', []):
            node_id = entity['id']
            node_type = entity['type'].split(':')[-1]
            attributes = entity.get('attributes', {})
            node_attrs = {
                'type': node_type,
                'label': entity.get('name', ''),
                **attributes
            }
            G.add_node(node_id, **node_attrs)
        for relationship in self.kg_data.get('relationships', []):
            source = relationship['source']
            target = relationship['target']
            rel_type = relationship['type'].split(':')[-1]
            G.add_edge(source, target, type=rel_type)
        return G

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the knowledge graph against the ontology and return a dictionary
        of results.
        """
        results = {}
        results['basic_metrics'] = self.basic_metrics()
        results['ontology_compliance'] = self.evaluate_ontology_compliance()
        results['graph_quality'] = self.evaluate_graph_quality()
        results['relation_distribution'] = self.relation_distribution()
        results['entity_coverage'] = self.entity_coverage()
        results['overall_score'] = self.calculate_overall_score(results)
        return results

    def evaluate_ontology_compliance(self) -> Dict[str, Any]:
        """
        Evaluate the knowledge graph against the ontology.
        Returns:
            Dict with evaluation results
        """
        valid_entities = 0
        valid_relations = 0
        domain_range_valid = 0

        invalid_entities = []
        invalid_relations = []
        domain_range_violations = []

        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', '')
            if node_type in self.entities:
                valid_entities += 1
            else:
                invalid_entities.append({'node_id': node, 'invalid_type': node_type})

        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('type', '')
            if rel_type in self.relations:
                valid_relations += 1
                constraints = self.ontology.get_relation_constraints(rel_type)
                source_type = self.graph.nodes[u].get('type', '')
                target_type = self.graph.nodes[v].get('type', '')
                if (source_type in constraints.get('domain', [])) and (target_type in constraints.get('range', [])):
                    domain_range_valid += 1
                else:
                    domain_range_violations.append({
                        'relation_type': rel_type,
                        'source': {'id': u, 'type': source_type},
                        'target': {'id': v, 'type': target_type},
                        'expected_domain': constraints.get('domain', []),
                        'expected_range': constraints.get('range', [])
                    })
            else:
                invalid_relations.append({'source': u, 'target': v, 'invalid_type': rel_type})

        total_attributes = 0
        valid_attributes = 0

        for _, data in self.graph.nodes(data=True):
            node_type = data.get('type', '')
            expected_attrs = self.attributes.get(node_type, [])
            for attr, value in data.items():
                if attr not in ['type', 'label']:
                    total_attributes += 1
                    if attr in expected_attrs:
                        valid_attributes += 1

        results = {
            'entity_type_validity': valid_entities / max(1, self.graph.number_of_nodes()),
            'relation_type_validity': valid_relations / max(1, self.graph.number_of_edges()),
            'domain_range_validity': domain_range_valid / max(1, self.graph.number_of_edges()),
            'attribute_validity': valid_attributes / max(1, total_attributes) if total_attributes > 0 else 1.0,
            'overall_compliance': 0,
            'validation_details': {
                'invalid_entities': invalid_entities,
                'invalid_relations': invalid_relations,
                'domain_range_violations': domain_range_violations
            }
        }

        results['overall_compliance'] = (
            results['entity_type_validity'] +
            results['relation_type_validity'] +
            results['domain_range_validity'] +
            results['attribute_validity']
        ) / 4

        return results

    def basic_metrics(self) -> Dict[str, Any]:
        """
        Calculate basic metrics of the graph.
        Returns:
            Dict with basic metrics
        """
        metrics = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes()),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'node_type_distribution': dict(Counter([d.get('type', 'unknown') for _, d in self.graph.nodes(data=True)]))
        }
        return metrics

    def evaluate_graph_quality(self) -> Dict[str, Any]:
        """
        Evaluate the quality of the graph based on various metrics.
        Returns:
            Dict with graph quality metrics
        """
        results = {}
        if self.graph.number_of_nodes() > 0:
            isolated_nodes = list(nx.isolates(self.graph))
            results['isolated_nodes_ratio'] = len(isolated_nodes) / self.graph.number_of_nodes()

            nodes_with_complete_attrs = 0
            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', '')
                expected_attrs = self.attributes.get(node_type, [])
                if all(attr in data for attr in expected_attrs):
                    nodes_with_complete_attrs += 1
            results['attribute_completeness'] = nodes_with_complete_attrs / self.graph.number_of_nodes()

            financial_metrics = [n for n, d in self.graph.nodes(data=True)
                if d.get('type') == 'Company' and any(key in d for key in ['annualRevenue', 'EBITDA', 'headcount'])]
            results['financial_metrics_availability'] = len(financial_metrics) > 0

            companies = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'Company']
            avg_company_relations = sum(len(list(self.graph.edges(c))) + len(list(self.graph.in_edges(c))) for c in companies) / max(1, len(companies))
            results['avg_company_relations'] = avg_company_relations

            try:
                largest_cc = max(nx.weakly_connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                results['graph_diameter'] = nx.diameter(subgraph.to_undirected())
            except:
                results['graph_diameter'] = 0
        else:
            raise ValueError("Graph is empty, cannot evaluate quality metrics.")

        return results

    def relation_distribution(self) -> Dict[str, float]:
        """
        Calculate the distribution of relation types in the graph.
        Returns:
            Dict with relation type distribution
            Dict with relation type counts
        """
        relation_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get('type', 'unknown')
            relation_counts[rel_type] += 1
        total = sum(relation_counts.values())
        return {k: v / total for k, v in relation_counts.items()}, {k: v for k, v in relation_counts.items() if v > 0}

    def entity_coverage(self) -> Dict[str, float]:
        """
        Calculate the coverage of entity types in the graph.
        Returns:
            Dict with entity type coverage
            Dict with entity type counts
        """
        entity_counts = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            entity_type = data.get('type', 'unknown')
            entity_counts[entity_type] += 1
        coverage = {etype: entity_counts[etype] > 0 for etype in self.entities}
        coverage['overall_entity_coverage'] = sum(coverage.values()) / max(1, len(self.entities))
        return coverage, entity_counts

    def calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate the overall score of the knowledge graph based on various metrics.
        Returns:
            float: Overall score
        """
        weights = {
            'ontology_compliance': 0.4,
            'graph_quality': 0.3,
            'entity_coverage': 0.2,
            'basic_metrics': 0.1
        }

        score = 0
        score += weights['ontology_compliance'] * results['ontology_compliance']['overall_compliance']

        graph_quality = results['graph_quality']
        quality_score = (
            (1 - graph_quality['isolated_nodes_ratio']) * 0.3 +
            graph_quality['attribute_completeness'] * 0.3 +
            (1 if graph_quality['financial_metrics_availability'] else 0) * 0.2 +
            min(1, graph_quality['avg_company_relations'] / 10) * 0.2
        )
        score += weights['graph_quality'] * quality_score

        score += weights['entity_coverage'] * results['entity_coverage'][0]['overall_entity_coverage']

        basic_metrics = results['basic_metrics']
        connectivity_score = 1 / max(1, basic_metrics['connected_components'])
        basic_score = (
            min(1, basic_metrics['density'] * 100) * 0.3 +
            min(1, basic_metrics['avg_degree'] / 5) * 0.4 +
            connectivity_score * 0.3
        )
        score += weights['basic_metrics'] * basic_score

        return score
    
    def export_evaluation_json(self, output_path: str) -> None:
        """
        Export the evaluation results to a JSON file.
        Args:
            output_path (str): Path to save the evaluation results.
        """
        results = self.evaluate()
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)


    def main(self, output_path: Optional[str] = None) -> None:
        """
        Main function to run the evaluation.
        """
        if output_path:
            self.export_evaluation_json(output_path)
        else:
            results = self.evaluate()
            print(json.dumps(results, indent=4))

if __name__ == "__main__":
    import sys 
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Invalid number of arguments.")
        print("Usage: python KG_evaluator.py <kg_json_path> <ontology_path> [<output_path>]")
        sys.exit(1)

    kg_json_path = sys.argv[1]
    ontology_path = sys.argv[2]
    
    output_path = sys.argv[3] if len(sys.argv) == 4 else None

    evaluator = KGEvaluator(kg_json_path, ontology_path)
    evaluator.main(output_path)