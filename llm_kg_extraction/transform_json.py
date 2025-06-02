import json
import uuid
import argparse
import re
from pathlib import Path
import yaml
from typing import Optional

# --- Default Configuration ---
DEFAULT_REQUEST_ID = "124"
DEFAULT_META_TITLE = "Visualization software"
# --- End Default Configuration ---

# Global variable to hold the loaded and processed ontology
ONTOLOGY_DATA = {}
# Defines a preference for which attribute to use as a label for a given entity type.
# The first attribute in the list that has a non-empty value in the entity will be used.
ONTOLOGY_LABEL_PREFERENCE = {
    "pekg:Company": ["name", "alias"],
    "pekg:GovernmentBody": ["name"],
    "pekg:Person": ["fullName", "name"],
    "pekg:Position": ["titleName", "name"],
    "pekg:ProductOrService": ["name"],
    "pekg:Product": ["name", "alias"],
    "pekg:Technology": ["name"],
    "pekg:MarketContext": ["segmentName"],
    "pekg:UseCaseOrIndustry": ["name"],
    "pekg:UseCase": ["name"],
    "pekg:FinancialMetric": ["metricName", "name"],
    "pekg:OperationalKPI": ["kpiName", "name"],
    "pekg:Headcount": ["headcountName"],
    "pekg:Shareholder": ["name"],
    "pekg:Advisor": ["name"],
    "pekg:TransactionContext": ["contextName"],
    "pekg:HistoricalEvent": ["eventName", "name"],
    "pekg:CorporateEvent": ["name"],
    "pekg:Location": ["locationName", "name", "officeSpecificInfo", "address"],
    "pekg:LegalEntity": ["name"],
    "pekg:PolicyDocument": ["name"],
    "pekg:OwnershipStake": ["name"],
    "pekg:Award": ["name"],
    "pekg:Department": ["name"],
     "_default": ["name", "id"]
}

def clean_link_type(type_str: str) -> str:
    """Converts a pekg:CamelCaseType to 'camel case type'."""
    if not type_str or ':' not in type_str:
        return "unknown link type"
    type_name = type_str.split(':')[-1]
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1 \2', type_name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s1).lower()
    return s2

def load_and_process_ontology(ontology_file_path: str) -> dict:
    """Loads the YAML ontology file and processes it into a usable dictionary."""
    processed_ontology = {"_attribute_definitions": {}}
    try:
        with open(ontology_file_path, 'r', encoding='utf-8') as f:
            raw_ontology = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Ontology file '{ontology_file_path}' not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML ontology file '{ontology_file_path}': {e}")
        return {}

    if not raw_ontology or 'entities' not in raw_ontology:
        print(f"Error: Ontology YAML '{ontology_file_path}' must contain a top-level 'entities' key with a list of definitions.")
        return {}

    if not isinstance(raw_ontology['entities'], list):
        print(f"Error: The 'entities' key in ontology YAML '{ontology_file_path}' should contain a list.")
        return {}

    for item in raw_ontology.get('entities', []):
        if isinstance(item, dict) and len(item) == 1:
            type_name, type_def = list(item.items())[0]
            if not isinstance(type_def, dict):
                print(f"Warning: Malformed type definition for '{type_name}' in ontology. Expected a dictionary.")
                continue
            processed_ontology[type_name] = type_def

            attr_list = []
            if 'attributes' in type_def and isinstance(type_def['attributes'], list):
                for attr_item in type_def['attributes']:
                    if isinstance(attr_item, dict) and len(attr_item) == 1:
                        attr_list.append(list(attr_item.keys())[0])
            processed_ontology['_attribute_definitions'][type_name] = attr_list
        else:
            if isinstance(item, dict) and 'id' in item and 'type' in item:
                 print(f"Critical Error: Attempting to parse an entity instance as an ontology definition in '{ontology_file_path}'.")
                 print(f"Problematic item: {item}")
                 print("This usually means the Knowledge Graph JSON file was passed as the ontology_file argument.")
                 print("Please ensure the first argument is the KG JSON and the second is the ontology YAML.")
                 return {}
            else:
                print(f"Warning: Skipping malformed entity definition in ontology: {item}")
    return processed_ontology

def get_category_from_type(type_str: str) -> str: # This function remains as per user's file
    """Extracts a clean category name from the entity type string."""
    global ONTOLOGY_DATA
    if not type_str:
        return "unknown"

    type_definition = ONTOLOGY_DATA.get(type_str, {})
    if 'category' in type_definition and isinstance(type_definition['category'], str):
        return type_definition['category']

    if ':' in type_str:
        category_suffix = type_str.split(':')[-1]
        category = re.sub(r'(?<!^)(?=[A-Z])', '_', category_suffix).lower()
        return category

    return "unknown"

def remove_pekg_prefix(text: str) -> str:
    """Removes 'pekg:' prefix from a string if it exists."""
    if text.startswith("pekg:"):
        return text[len("pekg:"):]
    return text

def get_entity_label(entity: dict) -> str:
    """Determines the appropriate label for an entity based on the ontology and preferences."""
    global ONTOLOGY_DATA, ONTOLOGY_LABEL_PREFERENCE
    entity_type = entity.get("type", "")
    type_definition_ontology = ONTOLOGY_DATA.get(entity_type, {})
    if "fixed_label" in type_definition_ontology:
        return type_definition_ontology["fixed_label"]

    preferred_attrs = ONTOLOGY_LABEL_PREFERENCE.get(entity_type, ONTOLOGY_LABEL_PREFERENCE['_default'])
    for attr_name in preferred_attrs:
        label_value = entity.get(attr_name)
        if isinstance(label_value, list):
            if label_value: label_value = str(label_value[0]).strip()
            else: label_value = None
        if label_value and str(label_value).strip():
            return str(label_value).strip()

    fallback_label = type_definition_ontology.get('label_fallback')
    if fallback_label: return fallback_label

    if entity_type == "pekg:FinancialMetric": return "Unknown Financial Metric"
    if entity_type == "pekg:OperationalKPI": return "Unnamed KPI"
    if entity_type == "pekg:MarketMetric": return "Unknown Market Metric"

    return "Unknown Entity"

def _format_attribute_for_description(attr_key: str, attr_value):
    """Helper to format a single attribute for description. Returns None if value is not suitable."""
    if attr_value is None: return None
    value_str = ""
    if isinstance(attr_value, list):
        if not attr_value: return None
        value_str = ', '.join(map(str, attr_value))
    elif isinstance(attr_value, bool): value_str = str(attr_value)
    elif isinstance(attr_value, (int, float)): value_str = str(attr_value)
    elif isinstance(attr_value, str) and attr_value.strip(): value_str = attr_value.strip()
    else: return None
    display_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', attr_key).title()
    display_name = display_name.replace('_', ' ')
    return f"{display_name}: {value_str}"

def get_primary_label_attribute_key(entity: dict, entity_label: str) -> Optional[str]:
    """Finds the attribute key that was most likely used to generate the entity_label."""
    entity_type = entity.get("type", "")
    preferred_attrs = ONTOLOGY_LABEL_PREFERENCE.get(entity_type, ONTOLOGY_LABEL_PREFERENCE['_default'])
    for attr_name in preferred_attrs:
        val = entity.get(attr_name)
        processed_val_for_label_check = None
        if isinstance(val, list):
            if val: processed_val_for_label_check = str(val[0]).strip()
        elif val is not None:
            processed_val_for_label_check = str(val).strip()

        if processed_val_for_label_check == entity_label:
            return attr_name
    return None

def generate_node_description(entity: dict, entity_label: str, entity_type_str_for_fallback: str) -> str:
    """
    Generates a description by concatenating key-value pairs of attributes,
    excluding "id", "type", and the attribute used for the entity_label.
    """
    description_parts = []
    primary_label_key = get_primary_label_attribute_key(entity, entity_label)

    for key, value in entity.items():
        if key == "id" or key == "type":
            continue
        if key == primary_label_key:
            continue

        formatted_attr = _format_attribute_for_description(key, value)
        if formatted_attr:
            description_parts.append(formatted_attr)

    if not description_parts:
        # Fallback if no other attributes to describe
        # Uses the raw type string (potentially with "pekg:") for the fallback description here
        # before prefix removal for the final category field.
        return f"{entity_label} (Type: {remove_pekg_prefix(entity_type_str_for_fallback)})."

    return ". ".join(description_parts) + "."

def generate_composite_id(original_id: str, entity_label: str) -> str:
    """Generates a new ID by concatenating original_id and a sanitized entity_label."""
    sanitized_label_part = str(entity_label)
    sanitized_label_part = re.sub(r'\s+', '_', sanitized_label_part)
    sanitized_label_part = re.sub(r'[^\w-]', '', sanitized_label_part)
    if not sanitized_label_part:
        sanitized_label_part = "nolabel"
    return f"{original_id}_{sanitized_label_part}"


def transform_kg(input_file_path: str,
                 output_dir: str,
                 request_id_to_use: str,
                 meta_title_to_use: str,
                 extraction_mode: str,
                 model_name: str,
                 llm_provider: str,
                 construction_mode: str,
                 ontology_file: str
                 ):
    global ONTOLOGY_DATA
    ONTOLOGY_DATA = load_and_process_ontology(ontology_file)
    if not ONTOLOGY_DATA: return

    input_path = Path(input_file_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        with open(input_path, 'r', encoding='utf-8') as f: data = json.load(f)
    except FileNotFoundError:
        print(f"Error (transform): Input KG file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error (transform): Could not decode JSON from {input_path}")
        return

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    meta_file_id = str(uuid.uuid4())
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    filename_base = f"{extraction_mode}_{safe_model_name}_{llm_provider}_{construction_mode}"

    if not isinstance(entities, list):
        print(f"Error: 'entities' field in input KG {input_path} is not a list.")
        entities = []

    entity_id_map = {}
    for entity in entities:
        if not isinstance(entity, dict): continue
        original_id = entity.get("id")
        if not original_id:
            original_id = str(uuid.uuid4())
            entity["id"] = original_id
            print(f"Warning: Entity found without 'id'. Assigned temporary ID: {original_id}. Entity: {str(entity)[:100]}")

        entity_label_for_id = get_entity_label(entity)
        composite_id = generate_composite_id(original_id, entity_label_for_id)
        entity_id_map[original_id] = composite_id

    actor_references = []
    for entity in entities:
        if not isinstance(entity, dict): continue
        original_entity_id = entity.get("id", "")
        actor_references.append({
            "label": get_entity_label(entity),
            "id": entity_id_map.get(original_entity_id, generate_composite_id(original_entity_id, get_entity_label(entity)))
        })

    link_keys_list = [str(uuid.uuid4()) for _ in relationships]
    link_references = [{"label": clean_link_type(rel.get("type", "unknown link")), "key": link_keys_list[idx]}
                       for idx, rel in enumerate(relationships)]

    meta_data = {
        "request_id": request_id_to_use, "title": meta_title_to_use, "id": meta_file_id,
        "file_local_path": input_path.name, "actor_references": actor_references, "link_references": link_references
    }
    meta_filename = output_path / f"{filename_base}_meta.json"
    with open(meta_filename, 'w', encoding='utf-8') as f: json.dump(meta_data, f, indent=2, ensure_ascii=False)
    print(f"Successfully created transformed file: {meta_filename}")

    nodes_list = []
    for entity in entities:
        if not isinstance(entity, dict): continue
        original_entity_id = entity.get("id", "")
        entity_type_str = entity.get("type", "unknown_type_fallback")

        node_label = get_entity_label(entity)

        # New category logic
        intermediate_category = get_category_from_type(entity_type_str)
        if intermediate_category == "unknown":
            category_base = entity_type_str
        else:
            category_base = intermediate_category
        final_node_category = remove_pekg_prefix(category_base)
        # End new category logic
        
        node_description = generate_node_description(entity, node_label, entity_type_str)

        nodes_list.append({
            "request_id": request_id_to_use,
            "id": entity_id_map.get(original_entity_id, generate_composite_id(original_entity_id, node_label)),
            "label": node_label,
            "category": final_node_category, # Use the new final_node_category
            "description": node_description,
            "sources": [{"FileId": meta_file_id, "FileName": f"{meta_file_id}" + ".meta.json", "Title": meta_title_to_use, "sourceint1" : request_id_to_use}],
        })
    nodes_filename = output_path / f"{filename_base}_nodes.json"
    with open(nodes_filename, 'w', encoding='utf-8') as f: json.dump(nodes_list, f, indent=2, ensure_ascii=False)
    print(f"Successfully created transformed file: {nodes_filename}")

    links_list = []
    if isinstance(relationships, list):
        for idx, rel in enumerate(relationships):
            if not isinstance(rel, dict):
                print(f"Warning: Skipping non-dictionary item in relationships list: {rel}")
                continue

            source_original_id = rel.get("source")
            target_original_id = rel.get("target")
            links_list.append({
                "request_id": request_id_to_use, "key": link_keys_list[idx],
                "sourceId": entity_id_map.get(source_original_id, source_original_id),
                "targetId": entity_id_map.get(target_original_id, target_original_id),
                "type": clean_link_type(rel.get("type", "unknown link")),
                "sources": [{"FileId": meta_file_id, "Title": meta_title_to_use, "sourceint1" : request_id_to_use, "FileName": f"{meta_file_id}" + ".meta.json"}]
            })
    else:
        print(f"Error: 'relationships' field in input KG {input_path} is not a list.")

    links_filename = output_path / f"{filename_base}_links.json"
    with open(links_filename, 'w', encoding='utf-8') as f: json.dump(links_list, f, indent=2, ensure_ascii=False)
    print(f"Successfully created transformed file: {links_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Transform a knowledge graph JSON into meta, nodes, and links JSON files using an ontology.\n"
                    "Usage: python transform_json.py <input_kg.json> <ontology.yaml> [options]",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", help="Path to the input JSON knowledge graph file.")
    parser.add_argument("ontology_file", help="Path to the YAML ontology file (e.g., pekg_ontology3.yaml).")
    parser.add_argument("-o", "--output_dir", default="output_transformed_kg", help="Directory to save the output files (default: output_transformed_kg).")
    parser.add_argument("--request_id", default=DEFAULT_REQUEST_ID, help="Request ID for transformed files.")
    parser.add_argument("--meta_title", default=DEFAULT_META_TITLE, help="Meta title for transformed files.")
    parser.add_argument("--extraction_mode", default="text", help="Extraction mode for filename.")
    parser.add_argument("--model_name", default="test_model", help="Model name for filename (sanitized).")
    parser.add_argument("--llm_provider", default="test_provider", help="LLM provider for filename.")
    parser.add_argument("--construction_mode", default="iterative", help="Construction mode for filename.")

    args = parser.parse_args()

    sanitized_model_name_for_func = args.model_name.replace("/", "_").replace(":", "_")

    input_path_obj = Path(args.input_file)
    if not input_path_obj.exists():
        print(f"Test input file {args.input_file} not found. Creating a dummy file for testing.")
        dummy_data = {
            "entities": [
                {"id": "comp1", "type": "pekg:Company", "name": "Acme Corp", "summary": "A leading manufacturer.", "industry": "Manufacturing", "website": "www.acme.com"},
                {"id": "pers1", "type": "pekg:Person", "fullName": "John Doe", "jobTitle": "CEO", "affiliation": "Acme Corp"},
                {"id": "metric1", "type": "pekg:FinancialMetric", "metricName": "Total Revenue 2023", "valueString": "€10.5M", "fiscalPeriod": "FY2023", "isRecurring": True},
                {"id": "kpi1", "type": "pekg:OperationalKPI", "kpiName": "Active Monthly Users", "kpiValueString": "150,000", "kpiDateOrPeriod": "May 2025", "description": "Users active in the last 30 days."},
                {"id": "hc1", "type": "pekg:HeadcountMetric", "totalEmployees": 120, "headcountName": "Global Workforce", "dateOrYear": "Q1 2025", "breakdownDescription": "Includes all departments."},
                {"id": "prod1", "type": "pekg:ProductOrService", "name": "Super Product", "description":"An innovative product."},
                {"id": "tech1", "type": "pekg:Technology", "name":"AI Engine", "description": "Core AI algorithms."}
            ],
            "relationships": [
                {"source": "comp1", "target": "pers1", "type": "pekg:Employs"}
            ]
        }
        with open(input_path_obj, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=2)
        print(f"Created dummy input file: {input_path_obj}")

    ontology_path_obj = Path(args.ontology_file)
    if not ontology_path_obj.exists():
        print(f"Ontology file {args.ontology_file} not found. Creating a dummy 'pekg_ontology3.yaml'.")
        dummy_ontology_content = """
entities:
  - pekg:Company:
      description: "A business entity."
      attributes: [{name: string}, {alias: list_string}, {summary: string}, {industry: string}, {website: string}]
  - pekg:Person: {description: "An individual.", attributes: [{fullName: string}, {jobTitle: string}, {affiliation: string}]}
  - pekg:FinancialMetric: {description: "A financial data point.", attributes: [{metricName: string}, {valueString: string}, {fiscalPeriod: string}, {isRecurring: boolean}, {scope: string}, {sourceNote: string}]}
  - pekg:OperationalKPI: {description: "A key performance indicator.", attributes: [{kpiName: string}, {kpiValueString: string}, {kpiDateOrPeriod: string}, {description: string}]}
  - pekg:HeadcountMetric: {description: "Employee count.", attributes: [{totalEmployees: integer}, {headcountName: string}, {breakdownDescription: string}, {dateOrYear: string}]}
  - pekg:ProductOrService: {description: "A product or service.", attributes: [{name: string}, {description: string}]}
  - pekg:Technology: {description: "A technology.", attributes: [{name: string}, {description: string}]}
"""
        with open(ontology_path_obj, 'w', encoding='utf-8') as f: f.write(dummy_ontology_content)
        print(f"Created dummy ontology file: {ontology_path_obj} - please replace with your full ontology.")

    transform_kg(
        args.input_file, args.output_dir, args.request_id, args.meta_title,
        args.extraction_mode, sanitized_model_name_for_func, args.llm_provider,
        args.construction_mode, args.ontology_file
    )