import yaml
import json
import re
from typing import Dict, List, Any, Optional


class PEKGOntology:
    def __init__(self, yaml_path: str):
        """
        Initialize the ontology by loading from a YAML file.
        The new YAML structure has attributes defined within each entity.
        
        Args:
            yaml_path (str): Path to the YAML ontology file
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        self.raw_data = data # Keep raw data if needed elsewhere
        
        raw_entity_list = data.get("entities", [])
        self.relations_str_list = data.get("relations", []) # Raw list of relation strings

        self.entity_definitions: Dict[str, Dict[str, Any]] = {}
        self.all_entity_names_prefixed: List[str] = []

        for entity_entry in raw_entity_list:
            if isinstance(entity_entry, dict):
                # Expecting format like: - pekg:Company: {description: ..., attributes: [...]}
                # So, entity_entry is a dict with one key: the entity name
                if len(entity_entry) == 1:
                    entity_name_with_prefix = list(entity_entry.keys())[0]
                    definition = entity_entry[entity_name_with_prefix]
                    
                    self.all_entity_names_prefixed.append(entity_name_with_prefix)
                    self.entity_definitions[entity_name_with_prefix] = {
                        "description": definition.get("description", ""),
                        "attributes": definition.get("attributes", []) # List of dicts, e.g., [{'name': 'string'}]
                    }
                else:
                    print(f"Warning: Unexpected entity entry format in ontology YAML: {entity_entry}")
            elif isinstance(entity_entry, str): # Handles simple entity type string without definition
                self.all_entity_names_prefixed.append(entity_entry)
                self.entity_definitions[entity_entry] = {
                    "description": "",
                    "attributes": []
                }
            else:
                print(f"Warning: Skipping unrecognized entity entry in ontology YAML: {entity_entry}")

        # These will be populated by transform_for_evaluation
        self.entities: List[str] = []  # List of unprefixed entity names
        self.relations: Dict[str, Dict[str, List[str]]] = {} # Dict of unprefixed relation info

    def format_for_prompt(self, include_attrs=True) -> str:
        """
        Generate a human-readable prompt-friendly representation of the ontology.
        
        Args:
            include_attrs (bool): Whether to include attributes in the output
        
        Returns:
            str: Formatted ontology description
        """
        entity_lines = []
        for entity_name_w_prefix in self.all_entity_names_prefixed:
            line = f"- {entity_name_w_prefix}"
            definition = self.entity_definitions.get(entity_name_w_prefix)
            if definition and definition.get("description"):
                line += f': {definition["description"]}'
            entity_lines.append(line)
        entity_block = "\n".join(entity_lines)
        
        relation_block = "\n".join(f"- {r}" for r in self.relations_str_list)

        prompt = f"""
        ### ONTOLOGY ###

        Entities:
        {entity_block}

        Relations:
        {relation_block}
        """
        if include_attrs:
            attr_block_lines = []
            for entity_name_w_prefix, definition in self.entity_definitions.items():
                attributes_list = definition.get("attributes", [])
                if attributes_list:
                    formatted_attrs = []
                    for attr_dict in attributes_list:
                        if isinstance(attr_dict, dict) and len(attr_dict) == 1:
                            attr_name = list(attr_dict.keys())[0]
                            attr_type = attr_dict[attr_name]
                            formatted_attrs.append(f"{attr_name} ({attr_type})")
                        elif isinstance(attr_dict, str): # Fallback for simple attribute name string
                            formatted_attrs.append(attr_dict)
                        # Else: skip malformed attribute
                    if formatted_attrs:
                         attr_block_lines.append(f"- {entity_name_w_prefix} Attributes: {', '.join(formatted_attrs)}")
            if attr_block_lines:
                prompt += f"\n\nAttributes (per Entity Type):\n" + "\n".join(attr_block_lines)

        return prompt.strip()
    
    def transform_for_evaluation(self):
        """
        Transform ontology data for easier evaluation and graph processing.
        Processes entity and relation names, extracts domains and ranges.
        Populates self.entities (unprefixed) and self.relations (unprefixed).
        """
        self.entities = [name.split(':')[-1] if ':' in name else name 
                         for name in self.all_entity_names_prefixed]
        
        self.relations = {} # Reset and rebuild
        for rel_str in self.relations_str_list:
            # Regex to capture: "pekg:relationName (pekg:DomainType|pekg:OtherDomain -> pekg:RangeType|pekg:OtherRange)"
            # or "relationName (DomainType -> RangeType)"
            match = re.match(r'(?:pekg:)?(\w+)\s*\(([^)]+)\s*→\s*([^)]+)\)', rel_str)
            if match:
                rel_name = match.group(1) # Unprefixed relation name
                domain_str = match.group(2)
                range_str = match.group(3)
                
                # Split by '|' and then clean "pekg:" prefix
                domain_types = [d.strip().split(':')[-1] for d in domain_str.split('|')]
                range_types = [r.strip().split(':')[-1] for r in range_str.split('|')]
                
                self.relations[rel_name] = {
                    'domain': domain_types,
                    'range': range_types
                }
            else:
                print(f"Warning: Could not parse relation string: {rel_str}")
        
    def export_to_json(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export processed ontology (unprefixed names after transform_for_evaluation) to a JSON format.
        Make sure to call transform_for_evaluation() before this if you want the transformed version.
        
        Args:
            output_path (str, optional): Path to save the JSON file
        
        Returns:
            Dict containing ontology information
        """
        # Attributes for export: map unprefixed entity name to list of attribute names
        attributes_for_export = {}
        for prefixed_name, definition in self.entity_definitions.items():
            unprefixed_name = prefixed_name.split(':')[-1] if ':' in prefixed_name else prefixed_name
            attrs_list_of_dicts = definition.get("attributes", [])
            attr_names = []
            for attr_dict in attrs_list_of_dicts:
                if isinstance(attr_dict, dict) and len(attr_dict) == 1:
                    attr_names.append(list(attr_dict.keys())[0])
                elif isinstance(attr_dict, str): # if attribute is just a name string
                    attr_names.append(attr_dict)
            if attr_names: # Only add if there are attributes
                attributes_for_export[unprefixed_name] = attr_names

        ontology_json = {
            "entities": self.entities, # Assumes transform_for_evaluation() has been called
            "relations": self.relations, # Assumes transform_for_evaluation() has been called
            "attributes": attributes_for_export 
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(ontology_json, f, indent=2)
        
        return ontology_json

    def validate_graph_compatibility(self, kg_json_path: str) -> Dict[str, Any]:
        """
        Validate a knowledge graph against the ontology.
        Assumes transform_for_evaluation() has been called on the ontology instance.
        
        Args:
            kg_json_path (str): Path to the knowledge graph JSON file
        
        Returns:
            Dict with validation results
        """
        if not self.entities or not self.relations:
            print("Warning: Ontology not transformed. Call transform_for_evaluation() before validating graph compatibility.")
            # Optionally call it here: self.transform_for_evaluation()

        with open(kg_json_path, 'r', encoding="utf-8") as f: # Added encoding
            kg_data = json.load(f)
        
        validation_results = {
            "entity_type_validity": [], # List of entities with invalid types
            "relation_type_validity": [], # List of relations with invalid types
            "missing_entities_from_ontology": [], # Ontology entity types not in KG
            "missing_relations_from_ontology": []  # Ontology relation types not in KG
        }
        
        # Validate entity types
        for entity in kg_data.get('entities', []):
            entity_type = entity.get('type', '').split(':')[-1] # Get unprefixed type
            if entity_type not in self.entities: # self.entities are unprefixed after transform
                validation_results["entity_type_validity"].append(
                    {"id": entity.get("id"), "type": entity.get("type"), "name": entity.get("name")}
                )
        
        # Validate relation types
        for relation in kg_data.get('relationships', []):
            rel_type = relation.get('type', '').split(':')[-1] # Get unprefixed type
            if rel_type not in self.relations: # self.relations keys are unprefixed after transform
                validation_results["relation_type_validity"].append(relation)
        
        graph_entity_types = set(
            entity.get('type', '').split(':')[-1] 
            for entity in kg_data.get('entities', []) if entity.get('type')
        )
        graph_relation_types = set(
            relation.get('type', '').split(':')[-1] 
            for relation in kg_data.get('relationships', []) if relation.get('type')
        )
        
        validation_results["missing_entities_from_ontology"] = list(
            set(self.entities) - graph_entity_types
        )
        validation_results["missing_relations_from_ontology"] = list(
            set(self.relations.keys()) - graph_relation_types
        )
        
        return validation_results

    def get_relation_constraints(self, relation_type_unprefixed: str) -> Dict[str, List[str]]:
        """
        Get domain and range constraints for a specific unprefixed relation type.
        Assumes transform_for_evaluation() has been called.
        
        Args:
            relation_type_unprefixed (str): The unprefixed relation type
        
        Returns:
            Dict with domain and range constraints (unprefixed types)
        """
        if not self.relations: # Check if transform_for_evaluation was called
            # print("Warning: transform_for_evaluation() likely not called. Relation constraints might be based on raw data.")
            # Fallback or raise error, for now, try to parse on the fly if needed, or return empty
            temp_relations_transformed = {}
            for rel_str in self.relations_str_list:
                match = re.match(r'(?:pekg:)?(\w+)\s*\(([^)]+)\s*→\s*([^)]+)\)', rel_str)
                if match:
                    rel_name = match.group(1)
                    domain_str = match.group(2)
                    range_str = match.group(3)
                    domain_types = [d.strip().split(':')[-1] for d in domain_str.split('|')]
                    range_types = [r.strip().split(':')[-1] for r in range_str.split('|')]
                    temp_relations_transformed[rel_name] = {'domain': domain_types, 'range': range_types}
            return temp_relations_transformed.get(relation_type_unprefixed, {"domain": [], "range": []})

        return self.relations.get(relation_type_unprefixed, {"domain": [], "range": []})

    def get_entity_attributes(self, entity_type_with_prefix: str) -> List[str]:
        """
        Get expected attribute names for a specific entity type (with prefix).
        
        Args:
            entity_type_with_prefix (str): The entity type with prefix (e.g., "pekg:Company")
        
        Returns:
            List of expected attribute names
        """
        definition = self.entity_definitions.get(entity_type_with_prefix)
        if not definition:
            return []
        
        attrs_list_of_dicts = definition.get("attributes", [])
        attr_names = []
        for attr_dict in attrs_list_of_dicts:
            if isinstance(attr_dict, dict) and len(attr_dict) == 1:
                # Assumes format like {'attributeName': 'typeString'}
                attr_names.append(list(attr_dict.keys())[0])
            elif isinstance(attr_dict, str): # Handles case where attribute is just a name string
                attr_names.append(attr_dict)
        return attr_names

    def get_all_entity_types_unprefixed(self) -> List[str]:
        """Returns a list of all unprefixed entity type names."""
        if not self.entities and self.all_entity_names_prefixed: # If transform not called yet
            return [name.split(':')[-1] if ':' in name else name for name in self.all_entity_names_prefixed]
        return self.entities # Assumes transform_for_evaluation has been called

    def get_all_relationship_types_unprefixed(self) -> List[str]:
        """Returns a list of all unprefixed relationship type names."""
        if not self.relations and self.relations_str_list: # If transform not called yet
            temp_rel_keys = []
            for rel_str in self.relations_str_list:
                match = re.match(r'(?:pekg:)?(\w+)\s*\(.*\)', rel_str)
                if match:
                    temp_rel_keys.append(match.group(1))
            return temp_rel_keys
        return list(self.relations.keys()) # Assumes transform_for_evaluation has been called