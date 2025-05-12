import yaml
import json
import re
from typing import Dict, List, Any, Optional


class PEKGOntology:
    def __init__(self, yaml_path: str):
        """
        Initialize the ontology by loading from a YAML file.
        
        Args:
            yaml_path (str): Path to the YAML ontology file
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        self.entities = data.get("entities", [])
        self.relations = data.get("relations", [])
        self.attributes = data.get("attributes", {})
        
        self.raw_data = data

    def format_for_prompt(self, include_attrs=True) -> str:
        """
        Generate a human-readable prompt-friendly representation of the ontology.
        
        Args:
            include_attrs (bool): Whether to include attributes in the output
        
        Returns:
            str: Formatted ontology description
        """
        entity_block = "\n".join(f"- {e}" for e in self.entities)
        relation_block = "\n".join(f"- {r}" for r in self.relations)

        prompt = f"""
        ### ONTOLOGY ###

        Entities:
        {entity_block}

        Relations:
        {relation_block}
        """
        if include_attrs:
            attr_lines = []
            for cls, props in self.attributes.items():
                attr_lines.append(f"- {cls}: {', '.join(props)}")
            prompt += f"\nAttributes:\n" + "\n".join(attr_lines)

        return prompt.strip()
    
    def transform_for_evaluation(self):
        """
        Transform ontology data for easier evaluation and graph processing.
        Processes entity and relation names, extracts domains and ranges.
        """
        self.rel_dic = {}
        
        self.entities = [entity.split(':')[-1] if ':' in entity else entity 
                         for entity in self.entities]
        
        for rel in self.relations:
            match = re.match(r'(pekg:)?(\w+)\s*\((.*?)\s*→\s*(.*?)\)', rel)
            if match:
                rel_name = match.group(2)
                domain = match.group(3).split('|')
                range_val = match.group(4).split('|')
                
                domain = [d.split(':')[-1] if ':' in d else d for d in domain]
                range_val = [r.split(':')[-1] if ':' in r else r for r in range_val]
                
                self.rel_dic[rel_name] = {
                    'domain': domain,
                    'range': range_val
                }
        
        self.relations = self.rel_dic

    def export_to_json(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export ontology to a JSON format.
        
        Args:
            output_path (str, optional): Path to save the JSON file
        
        Returns:
            Dict containing ontology information
        """
        ontology_json = {
            "entities": self.entities,
            "relations": self.relations,
            "attributes": self.attributes
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(ontology_json, f, indent=2)
        
        return ontology_json

    def validate_graph_compatibility(self, kg_json_path: str) -> Dict[str, Any]:
        """
        Validate a knowledge graph against the ontology.
        
        Args:
            kg_json_path (str): Path to the knowledge graph JSON file
        
        Returns:
            Dict with validation results
        """
        with open(kg_json_path, 'r') as f:
            kg_data = json.load(f)
        
        validation_results = {
            "entity_type_validity": [],
            "relation_type_validity": [],
            "missing_entities": [],
            "missing_relations": []
        }
        
        # Validate entity types
        for entity in kg_data.get('entities', []):
            entity_type = entity.get('type', '').split(':')[-1]
            if entity_type not in self.entities:
                validation_results["entity_type_validity"].append(entity)
        
        # Validate relation types
        for relation in kg_data.get('relationships', []):
            rel_type = relation.get('type', '').split(':')[-1]
            if rel_type not in self.relations:
                validation_results["relation_type_validity"].append(relation)
        
        # Find missing expected entities and relations
        graph_entity_types = set(
            entity.get('type', '').split(':')[-1] 
            for entity in kg_data.get('entities', [])
        )
        graph_relation_types = set(
            relation.get('type', '').split(':')[-1] 
            for relation in kg_data.get('relationships', [])
        )
        
        validation_results["missing_entities"] = list(
            set(self.entities) - graph_entity_types
        )
        validation_results["missing_relations"] = list(
            set(self.relations.keys()) - graph_relation_types
        )
        
        return validation_results

    def get_relation_constraints(self, relation_type: str) -> Dict[str, List[str]]:
        """
        Get domain and range constraints for a specific relation type.
        
        Args:
            relation_type (str): The relation type to get constraints for
        
        Returns:
            Dict with domain and range constraints
        """
        return self.relations.get(relation_type, {})

    def get_entity_attributes(self, entity_type: str) -> List[str]:
        """
        Get expected attributes for a specific entity type.
        
        Args:
            entity_type (str): The entity type to get attributes for
        
        Returns:
            List of expected attribute names
        """
        return self.attributes.get(entity_type, [])
