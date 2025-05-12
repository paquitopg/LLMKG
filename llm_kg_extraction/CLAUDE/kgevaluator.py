import json
import re
import networkx as nx
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Any, Optional


class KGEvaluator:
    """
    Knowledge Graph Evaluator for financial teasers based on PE-KG ontology.
    Evaluates graph quality, completeness, and compliance with ontology.
    """
    
    def __init__(self, kg_json_path: str, ontology_path: Optional[str] = None):
        """
        Initialize the evaluator with a knowledge graph in JSON format and optional ontology.
        
        Args:
            kg_json_path: Path to the JSON file containing the constructed knowledge graph
            ontology_path: Optional path to the ontology file (YAML or TXT)
        """
        # Load the knowledge graph
        with open(kg_json_path, 'r') as f:
            self.kg_data = json.load(f)
        
        # Parse the ontology from the provided file or use the built-in definition
        self.entities = []
        self.relations = {}
        self.attributes = {}
        
        if ontology_path:
            self._load_ontology(ontology_path)
        else:
            # Using built-in ontology definition (based on the provided files)
            self._load_default_ontology()
        
        # Create a networkx graph for analysis
        self.graph = self._create_nx_graph()
        
    def _load_ontology(self, ontology_path: str):
        """Load ontology from file based on file extension"""
        if ontology_path.endswith('.yaml') or ontology_path.endswith('.yml'):
            self._load_yaml_ontology(ontology_path)
        elif ontology_path.endswith('.txt'):
            self._load_txt_ontology(ontology_path)
        else:
            raise ValueError(f"Unsupported ontology file format: {ontology_path}")
    
    def _load_yaml_ontology(self, yaml_path: str):
        """Load ontology from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            ontology = yaml.safe_load(f)
        
        self.entities = [entity.split(':')[1] if ':' in entity else entity 
                         for entity in ontology.get('entities', [])]
        
        # Parse relations and their domains/ranges
        for rel in ontology.get('relations', []):
            # Extract relation name and its domain/range
            match = re.match(r'pekg:(\w+)\s*\((.*?)\s*→\s*(.*?)\)', rel)
            if match:
                rel_name, domain, range_val = match.groups()
                self.relations[rel_name] = {
                    'domain': domain.split('|'),
                    'range': range_val.split('|')
                }
        
        # Parse attributes
        self.attributes = ontology.get('attributes', {})
        
    def _load_txt_ontology(self, txt_path: str):
        """Load ontology from Turtle/RDF TXT file"""
        with open(txt_path, 'r') as f:
            content = f.read()
        
        # Extract entity classes
        entity_matches = re.findall(r'pekg:(\w+)\s+rdf:type\s+owl:Class', content)
        self.entities = entity_matches
        
        # Extract relations
        relation_matches = re.findall(r'pekg:(\w+)\s+rdf:type\s+owl:ObjectProperty', content)
        for rel in relation_matches:
            # Try to find domain and range information
            domain_match = re.search(rf'pekg:{rel}.*?rdfs:domain\s+(.*?)\s*;', content, re.DOTALL)
            range_match = re.search(rf'pekg:{rel}.*?rdfs:range\s+(.*?)\s*\.', content, re.DOTALL)
            
            domain = domain_match.group(1).split(',') if domain_match else []
            range_val = range_match.group(1).split(',') if range_match else []
            
            # Clean up the domain and range
            domain = [d.strip().split(':')[1] if ':' in d else d.strip() for d in domain]
            range_val = [r.strip().split(':')[1] if ':' in r else r.strip() for r in range_val]
            
            self.relations[rel] = {
                'domain': domain,
                'range': range_val
            }
        
        # Extract attributes (datatype properties)
        attr_pattern = r'pekg:(\w+)\s+rdf:type\s+owl:DatatypeProperty\s*;\s*rdfs:domain\s+(.*?)\s*;'
        attr_matches = re.findall(attr_pattern, content)
        for attr, domains in attr_matches:
            domains = domains.split(',')
            for domain in domains:
                domain = domain.strip()
                if ':' in domain:
                    domain = domain.split(':')[1]
                
                if domain not in self.attributes:
                    self.attributes[domain] = []
                self.attributes[domain].append(attr)
    
    def _load_default_ontology(self):
        """Load the default ontology built into the class"""
        self.entities = [
            "Company", "LegalEntity", "Person", "Department", "Position", "Committee",
            "Product", "UseCase", "Investor", "Advisor", "GovernmentBody", "FinancialInstrument",
            "OwnershipStake", "Contract", "IntellectualProperty", "Litigation", "Risk",
            "RegulatoryRequirement", "PolicyDocument", "CorporateEvent", "FundingRound",
            "AcquisitionEvent", "MergerEvent", "IPOEvent", "ExitEvent", "LeadershipChangeEvent",
            "FinancialMetric", "HeadcountMetric", "MarketMetric", "TaxMetric", "KPI",
            "NewsItem", "Location"
        ]
        
        # Define relation domains and ranges
        self.relations = {
            "ownsEntity": {"domain": ["Company"], "range": ["LegalEntity"]},
            "hasExecutive": {"domain": ["Company", "LegalEntity"], "range": ["Person"]},
            "hasBoardMember": {"domain": ["Company"], "range": ["Person"]},
            "employs": {"domain": ["Company"], "range": ["Person"]},
            "hasDepartment": {"domain": ["Company"], "range": ["Department"]},
            "holdsPosition": {"domain": ["Person"], "range": ["Position"]},
            "isMemberOfCommittee": {"domain": ["Person"], "range": ["Committee"]},
            "hasCommittee": {"domain": ["Company"], "range": ["Committee"]},
            "targetsUseCase": {"domain": ["Product"], "range": ["UseCase"]},
            "operatesInMarket": {"domain": ["Company"], "range": ["UseCase"]},
            "competesWith": {"domain": ["Company"], "range": ["Company"]},
            "hasCustomer": {"domain": ["Company"], "range": ["Company"]},
            "hasPartnershipWith": {"domain": ["Company"], "range": ["Company"]},
            "hasContractor": {"domain": ["Company"], "range": ["Company"]},
            "hasHeadquartersIn": {"domain": ["Company"], "range": ["Location"]},
            "hasOfficeIn": {"domain": ["Company"], "range": ["Location"]},
            "operatesInLocation": {"domain": ["Company"], "range": ["Location"]},
            "registeredIn": {"domain": ["LegalEntity"], "range": ["Location"]},
            "hasOwnershipStake": {"domain": ["Company"], "range": ["OwnershipStake"]},
            "hasInvestor": {"domain": ["OwnershipStake"], "range": ["Investor"]},
            "inLegalEntity": {"domain": ["OwnershipStake"], "range": ["LegalEntity"]},
            "receivedInvestment": {"domain": ["Company"], "range": ["FundingRound"]},
            "investsIn": {"domain": ["Investor"], "range": ["Company"]},
            "acquired": {"domain": ["AcquisitionEvent"], "range": ["Company"]},
            "acquirer": {"domain": ["AcquisitionEvent"], "range": ["Company"]},
            "mergedWith": {"domain": ["Company"], "range": ["Company"]},
            "hasEvent": {"domain": ["Company"], "range": ["CorporateEvent"]},
            "signsContract": {"domain": ["Company"], "range": ["Contract"]},
            "holdsIP": {"domain": ["Company"], "range": ["IntellectualProperty"]},
            "partyToLitigation": {"domain": ["Company"], "range": ["Litigation"]},
            "hasRisk": {"domain": ["Company"], "range": ["Risk"]},
            "subjectTo": {"domain": ["Company"], "range": ["RegulatoryRequirement"]},
            "hasPolicy": {"domain": ["Company"], "range": ["PolicyDocument"]},
            "contractsWithGov": {"domain": ["Company"], "range": ["GovernmentBody"]},
            "reportsMetric": {"domain": ["Company"], "range": ["FinancialMetric"]},
            "hasHeadcount": {"domain": ["Department"], "range": ["HeadcountMetric"]},
            "hasKPI": {"domain": ["Company"], "range": ["KPI"]},
            "mentionsCompany": {"domain": ["NewsItem"], "range": ["Company"]},
            "hasSentiment": {"domain": ["NewsItem"], "range": ["Risk"]},
        }
        
        # Define attributes per entity type
        self.attributes = {
            "FinancialMetric": ["metricValue", "metricCurrency", "metricUnit", "percentageValue"],
            "MarketMetric": ["metricValue", "metricUnit"],
            "KPI": ["metricUnit"],
            "HeadcountMetric": ["headcountValue"],
            "FundingRound": ["roundDate", "roundAmount"],
            "Location": ["latitude", "longitude"]
        }
    
    def _create_nx_graph(self) -> nx.DiGraph:
        """Create a NetworkX graph from the KG data for analysis"""
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node_data in self.kg_data.get('nodes', {}).items():
            G.add_node(node_id, **node_data)
        
        # Add edges
        for edge_id, edge_data in self.kg_data.get('edges', {}).items():
            source = edge_data.get('source')
            target = edge_data.get('target')
            if source and target:
                G.add_edge(source, target, id=edge_id, **edge_data)
        
        return G
    
    def evaluate(self) -> Dict[str, Any]:
        """Run all evaluations and return aggregated results"""
        results = {}
        
        # Basic graph metrics
        results['basic_metrics'] = self.basic_metrics()
        
        # Ontology compliance
        results['ontology_compliance'] = self.evaluate_ontology_compliance()
        
        # Graph quality
        results['graph_quality'] = self.evaluate_graph_quality()
        
        # Distribution of relation types
        results['relation_distribution'] = self.relation_distribution()
        
        # Entity coverage
        results['entity_coverage'] = self.entity_coverage()
        
        # Calculate overall scores
        results['overall_score'] = self.calculate_overall_score(results)
        
        return results
    
    def basic_metrics(self) -> Dict[str, Any]:
        """Calculate basic graph metrics"""
        metrics = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes()),
            'connected_components': nx.number_weakly_connected_components(self.graph),
        }
        
        # Node type distribution
        node_types = [data.get('type', 'unknown') for _, data in self.graph.nodes(data=True)]
        metrics['node_type_distribution'] = dict(Counter(node_types))
        
        return metrics
    
    def evaluate_ontology_compliance(self) -> Dict[str, Any]:
        """Evaluate how well the KG complies with the ontology"""
        results = {
            'entity_type_validity': 0,
            'relation_type_validity': 0,
            'domain_range_validity': 0,
            'attribute_validity': 0,
        }
        
        # Check entity type validity
        valid_entities = 0
        for _, data in self.graph.nodes(data=True):
            node_type = data.get('type', '')
            if node_type in self.entities:
                valid_entities += 1
        
        results['entity_type_validity'] = valid_entities / max(1, self.graph.number_of_nodes())
        
        # Check relation type validity
        valid_relations = 0
        domain_range_valid = 0
        
        for source, target, data in self.graph.edges(data=True):
            rel_type = data.get('type', '')
            
            # Check if relation type is valid
            if rel_type in self.relations:
                valid_relations += 1
                
                # Check domain-range validity
                source_type = self.graph.nodes[source].get('type', '')
                target_type = self.graph.nodes[target].get('type', '')
                
                if (not self.relations[rel_type]['domain'] or 
                    source_type in self.relations[rel_type]['domain']) and \
                   (not self.relations[rel_type]['range'] or 
                    target_type in self.relations[rel_type]['range']):
                    domain_range_valid += 1
        
        results['relation_type_validity'] = valid_relations / max(1, self.graph.number_of_edges())
        results['domain_range_validity'] = domain_range_valid / max(1, self.graph.number_of_edges())
        
        # Check attribute validity
        total_attributes = 0
        valid_attributes = 0
        
        for _, data in self.graph.nodes(data=True):
            node_type = data.get('type', '')
            expected_attrs = self.attributes.get(node_type, [])
            
            for attr, value in data.items():
                if attr != 'type' and attr != 'id' and attr != 'label':
                    total_attributes += 1
                    if attr in expected_attrs:
                        valid_attributes += 1
        
        results['attribute_validity'] = valid_attributes / max(1, total_attributes) if total_attributes > 0 else 1.0
        
        # Overall ontology compliance score
        results['overall_compliance'] = (
            results['entity_type_validity'] +
            results['relation_type_validity'] +
            results['domain_range_validity'] +
            results['attribute_validity']
        ) / 4
        
        return results
    
    def evaluate_graph_quality(self) -> Dict[str, Any]:
        """Evaluate the quality of the knowledge graph"""
        results = {}
        
        # Completeness metrics
        if self.graph.number_of_nodes() > 0:
            # Check for isolated nodes
            isolated_nodes = list(nx.isolates(self.graph))
            results['isolated_nodes_ratio'] = len(isolated_nodes) / self.graph.number_of_nodes()
            
            # Consistency check - ratio of nodes with all required attributes
            nodes_with_complete_attrs = 0
            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', '')
                expected_attrs = self.attributes.get(node_type, [])
                
                has_all_attrs = all(attr in data for attr in expected_attrs)
                if has_all_attrs:
                    nodes_with_complete_attrs += 1
            
            results['attribute_completeness'] = nodes_with_complete_attrs / self.graph.number_of_nodes()
            
            # Financial metrics availability
            financial_metrics = [n for n, d in self.graph.nodes(data=True) 
                                if d.get('type') == 'FinancialMetric']
            results['financial_metrics_availability'] = len(financial_metrics) > 0
            
            # Company relationships
            companies = [n for n, d in self.graph.nodes(data=True) 
                        if d.get('type') == 'Company']
            avg_company_relations = 0
            if companies:
                company_relations = [len(list(self.graph.edges(c))) + len(list(self.graph.in_edges(c))) 
                                    for c in companies]
                avg_company_relations = sum(company_relations) / len(companies)
            
            results['avg_company_relations'] = avg_company_relations
            
            # Graph diameter estimation (using a sample if graph is large)
            if self.graph.number_of_nodes() < 1000:
                try:
                    # Get largest connected component
                    largest_cc = max(nx.weakly_connected_components(self.graph), key=len)
                    subgraph = self.graph.subgraph(largest_cc)
                    results['graph_diameter'] = nx.diameter(subgraph)
                except (nx.NetworkXError, ValueError):
                    results['graph_diameter'] = 0
            else:
                results['graph_diameter'] = 'N/A (graph too large)'
        else:
            results = {
                'isolated_nodes_ratio': 0,
                'attribute_completeness': 0,
                'financial_metrics_availability': False,
                'avg_company_relations': 0,
                'graph_diameter': 0
            }
        
        return results
    
    def relation_distribution(self) -> Dict[str, float]:
        """Calculate distribution of relation types in the graph"""
        relation_counts = defaultdict(int)
        
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get('type', 'unknown')
            relation_counts[rel_type] += 1
        
        total_relations = self.graph.number_of_edges()
        
        # Convert to percentage
        distribution = {rel: count / max(1, total_relations) 
                        for rel, count in relation_counts.items()}
        
        return distribution
    
    def entity_coverage(self) -> Dict[str, float]:
        """Calculate coverage of entity types in the graph"""
        entity_counts = defaultdict(int)
        
        for _, data in self.graph.nodes(data=True):
            entity_type = data.get('type', 'unknown')
            entity_counts[entity_type] += 1
        
        # Calculate coverage compared to ontology
        coverage = {}
        for entity_type in self.entities:
            coverage[entity_type] = entity_counts[entity_type] > 0
        
        # Calculate percentage of covered entity types
        coverage['overall_entity_coverage'] = sum(coverage.values()) / len(self.entities)
        
        return coverage
    
    def calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate an overall score based on all metrics"""
        weights = {
            'ontology_compliance': 0.4,
            'graph_quality': 0.3,
            'entity_coverage': 0.2,
            'basic_metrics': 0.1
        }
        
        score = 0
        
        # Ontology compliance score
        score += weights['ontology_compliance'] * results['ontology_compliance']['overall_compliance']
        
        # Graph quality score
        graph_quality = results['graph_quality']
        quality_score = (
            (1 - graph_quality['isolated_nodes_ratio']) * 0.3 +
            graph_quality['attribute_completeness'] * 0.3 +
            (1 if graph_quality['financial_metrics_availability'] else 0) * 0.2 +
            min(1, graph_quality['avg_company_relations'] / 10) * 0.2  # Normalize to [0,1]
        )
        score += weights['graph_quality'] * quality_score
        
        # Entity coverage score
        score += weights['entity_coverage'] * results['entity_coverage']['overall_entity_coverage']
        
        # Basic metrics score
        basic_metrics = results['basic_metrics']
        connectivity_score = 1 / max(1, basic_metrics['connected_components'])
        basic_score = (
            min(1, basic_metrics['density'] * 100) * 0.3 +  # Adjust for sparse graphs
            min(1, basic_metrics['avg_degree'] / 5) * 0.4 +  # Normalize to [0,1]
            connectivity_score * 0.3
        )
        score += weights['basic_metrics'] * basic_score
        
        return score
    
    def plot_entity_distribution(self, top_n: int = 10) -> None:
        """Plot distribution of entity types in the graph"""
        entity_counts = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            entity_type = data.get('type', 'unknown')
            entity_counts[entity_type] += 1
        
        # Sort and get top N entity types
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        top_entities = sorted_entities[:top_n]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.bar([e[0] for e in top_entities], [e[1] for e in top_entities])
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {top_n} Entity Types in Knowledge Graph')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    def plot_relation_distribution(self, top_n: int = 10) -> None:
        """Plot distribution of relation types in the graph"""
        relation_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get('type', 'unknown')
            relation_counts[rel_type] += 1
        
        # Sort and get top N relation types
        sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
        top_relations = sorted_relations[:top_n]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.bar([r[0] for r in top_relations], [r[1] for r in top_relations])
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {top_n} Relation Types in Knowledge Graph')
        plt.xlabel('Relation Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report"""
        results = self.evaluate()
        
        report = []
        report.append("# Knowledge Graph Evaluation Report")
        report.append("\n## 1. Basic Metrics")
        report.append(f"- Number of nodes: {results['basic_metrics']['num_nodes']}")
        report.append(f"- Number of edges: {results['basic_metrics']['num_edges']}")
        report.append(f"- Graph density: {results['basic_metrics']['density']:.4f}")
        report.append(f"- Average degree: {results['basic_metrics']['avg_degree']:.2f}")
        report.append(f"- Connected components: {results['basic_metrics']['connected_components']}")
        
        report.append("\n## 2. Ontology Compliance")
        report.append(f"- Entity type validity: {results['ontology_compliance']['entity_type_validity']:.2%}")
        report.append(f"- Relation type validity: {results['ontology_compliance']['relation_type_validity']:.2%}")
        report.append(f"- Domain-range validity: {results['ontology_compliance']['domain_range_validity']:.2%}")
        report.append(f"- Attribute validity: {results['ontology_compliance']['attribute_validity']:.2%}")
        report.append(f"- Overall compliance: {results['ontology_compliance']['overall_compliance']:.2%}")
        
        report.append("\n## 3. Graph Quality")
        report.append(f"- Isolated nodes ratio: {results['graph_quality']['isolated_nodes_ratio']:.2%}")
        report.append(f"- Attribute completeness: {results['graph_quality']['attribute_completeness']:.2%}")
        report.append(f"- Financial metrics available: {results['graph_quality']['financial_metrics_availability']}")
        report.append(f"- Average company relations: {results['graph_quality']['avg_company_relations']:.2f}")
        
        report.append("\n## 4. Entity Coverage")
        report.append(f"- Overall entity coverage: {results['entity_coverage']['overall_entity_coverage']:.2%}")
        report.append("- Entity types present:")
        for entity_type, present in results['entity_coverage'].items():
            if entity_type != 'overall_entity_coverage':
                report.append(f"  - {entity_type}: {'✓' if present else '✗'}")
        
        report.append("\n## 5. Top Relations")
        top_relations = sorted(
            results['relation_distribution'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        for rel, pct in top_relations:
            report.append(f"- {rel}: {pct:.2%}")
        
        report.append("\n## 6. Overall Score")
        report.append(f"- Quality score: {results['overall_score']:.2%}")
        
        # Save to file if requested
        report_text = "\n".join(report)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text


# Example usage
if __name__ == "__main__":
    # Replace with actual path to your KG JSON file
    evaluator = KGEvaluator("knowledge_graph.json")
    
    # Run evaluation
    results = evaluator.evaluate()
    print(f"Overall KG quality score: {results['overall_score']:.2%}")
    
    # Generate and save report
    report = evaluator.generate_report("kg_evaluation_report.md")
    
    # Plot visualizations
    evaluator.plot_entity_distribution()
    evaluator.plot_relation_distribution()
