import json
import uuid
import argparse
import re
from pathlib import Path

# --- Default Configuration (used if no overrides are provided to transform_kg) ---
DEFAULT_REQUEST_ID = "2" 
DEFAULT_META_TITLE = "Visualization software" 
# --- End Default Configuration ---

def get_category_from_type(type_str: str) -> str:
    """Extracts a clean category name from the entity type string."""
    if not type_str or ':' not in type_str:
        return "unknown"
    return type_str.split(':')[-1].lower()

def clean_link_type(type_str: str) -> str:
    """Converts a pekg:CamelCaseType to 'camel case type'."""
    if not type_str or ':' not in type_str:
        return "unknown link type"
    
    type_name = type_str.split(':')[-1]
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1 \2', type_name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s1).lower()
    return s2

def generate_node_description(entity: dict) -> str:
    """Generates a description for a node based on its attributes."""
    entity_type = entity.get("type", "")
    
    name_for_description = entity.get("name")
    if not name_for_description: 
        if entity_type == "pekg:MarketMetric":
            name_for_description = entity.get("metricUnit", "Unknown Market Metric")
        elif entity_type == "pekg:FinancialMetric":
            name_for_description = entity.get("metricUnit", "Unknown Financial Metric")
        elif entity_type == "pekg:HeadcountMetric":
            name_for_description = "Headcount"
        elif entity_type == "pekg:KPI":
            name_for_description = entity.get("name", "Unnamed KPI")
        else:
            name_for_description = "Unknown Entity"

    if "description" in entity and entity["description"] and str(entity["description"]).strip():
        if entity_type == "pekg:Attribute":
            attr_name = entity.get('name', 'Unnamed Attribute')
            return f"Attribute '{attr_name}': {str(entity['description'])}."
        return str(entity["description"])

    if entity_type == "pekg:FinancialMetric":
        val = entity.get("metricValue", "N/A")
        curr = entity.get("metricCurrency", "")
        unit_label = entity.get("metricUnit", "Financial Metric")
        desc_parts = [f"Value: {val} {curr}".strip()]
        if entity.get("percentageValue") is not None:
            desc_parts.append(f"Percentage: {entity.get('percentageValue')}%")
        return f"Details for '{unit_label}': {'. '.join(desc_parts)}."
    elif entity_type == "pekg:HeadcountMetric":
        val = entity.get("headcountValue", "N/A")
        return f"Total headcount value: {val}."
    elif entity_type == "pekg:MarketMetric":
        val = entity.get("metricValue", "N/A")
        unit_label = entity.get("metricUnit", "Market Metric")
        curr = entity.get("metricCurrency", "")
        value_str = f"{val}"
        if curr: value_str += f" {curr}"
        return f"Details for '{unit_label}': Value is {value_str.strip()}."
    elif entity_type == "pekg:KPI":
        kpi_name = entity.get("name", "KPI")
        val = entity.get("metricValue", "N/A")
        unit = entity.get("metricUnit", "")
        desc = f"Value: {val} {unit}".strip()
        return f"Details for KPI '{kpi_name}': {desc}."
    elif entity_type == "pekg:Product": return f"Product: {name_for_description}."
    elif entity_type == "pekg:Company": return f"Company: {name_for_description}."
    elif entity_type == "pekg:Person": return f"Person: {name_for_description}."
    elif entity_type == "pekg:Location": return f"Location: {name_for_description}."
    elif entity_type == "pekg:Department": return f"Department: {name_for_description}."
    elif entity_type == "pekg:Project": return f"Project: {name_for_description}."
    elif entity_type == "pekg:Advisor": return f"Advisor: {name_for_description}."
    elif entity_type == "pekg:UseCase": return f"Use Case: {name_for_description}."
    elif entity_type == "pekg:Contract": return f"Contract: {name_for_description}."
    elif entity_type == "pekg:Position": return f"Position: {name_for_description}."

    generic_description_parts = []
    for key, value in entity.items():
        if key not in ["id", "type", "name", "description", "metricUnit", "metricValue", "metricCurrency", "headcountValue", "percentageValue"] and value is not None:
            if isinstance(value, str) and not value.strip():
                continue
            generic_description_parts.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    if generic_description_parts:
        return f"{name_for_description}: " + ". ".join(generic_description_parts) + "."
    return f"{name_for_description} (Type: {get_category_from_type(entity_type)})."

def transform_kg(input_file_path: str, 
                 output_dir: str, 
                 request_id_to_use: str, 
                 meta_title_to_use: str,
                 extraction_mode: str,
                 model_name: str, # Expect sanitized model name
                 llm_provider: str,
                 construction_mode: str
                 ):
    """
    Transforms the input knowledge graph JSON into three separate JSON files
    with dynamic filenames.
    Args:
        input_file_path (str): Path to the main knowledge graph JSON file.
        output_dir (str): Directory to save the transformed files.
        request_id_to_use (str): The request ID to use in transformed files.
        meta_title_to_use (str): The title to use in meta.json.
        extraction_mode (str): e.g., "text", "multimodal"
        model_name (str): Sanitized model name (e.g., "gemini-1_5-pro-latest")
        llm_provider (str): e.g., "vertexai", "azure"
        construction_mode (str): e.g., "iterative", "onego", "parallel"
    """
    input_path = Path(input_file_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error (transform): Input KG file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error (transform): Could not decode JSON from {input_path}")
        return

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    meta_file_id = str(uuid.uuid4())

    # Construct the base for dynamic filenames
    filename_base = f"{extraction_mode}_{model_name}_{llm_provider}_{construction_mode}"

    actor_references = []
    for entity in entities:
        entity_type = entity.get("type", "")
        label = ""
        if entity_type == "pekg:MarketMetric": label = entity.get("metricUnit", "Unknown Market Metric")
        elif entity_type == "pekg:FinancialMetric": label = entity.get("metricUnit", "Unknown Financial Metric")
        elif entity_type == "pekg:HeadcountMetric": label = "Headcount"
        elif entity_type == "pekg:KPI": label = entity.get("name", "Unnamed KPI")
        else: label = entity.get("name", "Unknown Entity")
        actor_references.append({"label": label, "id": entity.get("id", str(uuid.uuid4()))})

    link_keys_list = [str(uuid.uuid4()) for _ in relationships]
    link_references = [{"label": clean_link_type(rel.get("type", "unknown link")), "key": link_keys_list[idx]} for idx, rel in enumerate(relationships)]

    meta_data = {
        "request_id": request_id_to_use, 
        "title": meta_title_to_use, 
        "id": meta_file_id,
        "file_local_path": input_path.name, 
        "actor_references": actor_references, 
        "link_references": link_references
    }
    meta_filename = output_path / f"{filename_base}_meta.json"
    with open(meta_filename, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)
    print(f"Successfully created transformed file: {meta_filename}")

    nodes_list = []
    for entity in entities:
        entity_type = entity.get("type", "")
        node_label = ""
        if entity_type == "pekg:MarketMetric": node_label = entity.get("metricUnit", "Unknown Market Metric")
        elif entity_type == "pekg:FinancialMetric": node_label = entity.get("metricUnit", "Unknown Financial Metric")
        elif entity_type == "pekg:HeadcountMetric": node_label = "Headcount"
        elif entity_type == "pekg:KPI": node_label = entity.get("name", "Unnamed KPI")
        else: node_label = entity.get("name", "Unknown Entity")
        
        nodes_list.append({
            "request_id": request_id_to_use, 
            "id": entity.get("id", str(uuid.uuid4())), 
            "label": node_label,
            "category": get_category_from_type(entity_type), 
            "description": generate_node_description(entity),
            "timeline": [""], 
            "sources": [{"FileId": meta_file_id, "Snipets": [""]}]
        })
    nodes_filename = output_path / f"{filename_base}_nodes.json"
    with open(nodes_filename, 'w', encoding='utf-8') as f:
        json.dump(nodes_list, f, indent=2, ensure_ascii=False)
    print(f"Successfully created transformed file: {nodes_filename}")

    links_list = []
    for idx, rel in enumerate(relationships):
        links_list.append({
            "request_id": request_id_to_use, 
            "key": link_keys_list[idx], 
            "sourceId": rel.get("source"),
            "targetId": rel.get("target"), 
            "type": clean_link_type(rel.get("type", "unknown link")),
            "is_currently_ongoing": True, 
            "date_of_first_occurrence": "",
            "sources": [{"FileId": meta_file_id, "Snipets": [""]}]
        })
    links_filename = output_path / f"{filename_base}_links.json"
    with open(links_filename, 'w', encoding='utf-8') as f:
        json.dump(links_list, f, indent=2, ensure_ascii=False)
    print(f"Successfully created transformed file: {links_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform a knowledge graph JSON into meta, nodes, and links JSON files.")
    parser.add_argument("input_file", help="Path to the input JSON knowledge graph file.")
    parser.add_argument("-o", "--output_dir", default="output_transformed_kg", help="Directory to save the output files (default: output_transformed_kg).")
    parser.add_argument("--request_id", default=DEFAULT_REQUEST_ID, help="Request ID for transformed files.")
    parser.add_argument("--meta_title", default=DEFAULT_META_TITLE, help="Meta title for transformed files.")
    # Added arguments for standalone testing with new filename components
    parser.add_argument("--extraction_mode", default="text", help="Extraction mode for filename.")
    parser.add_argument("--model_name", default="test_model", help="Model name for filename (sanitized).")
    parser.add_argument("--llm_provider", default="test_provider", help="LLM provider for filename.")
    parser.add_argument("--construction_mode", default="iterative", help="Construction mode for filename.")
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Test input file {args.input_file} not found. Please create it for standalone testing.")
    else:
        transform_kg(
            args.input_file, 
            args.output_dir, 
            args.request_id, 
            args.meta_title,
            args.extraction_mode,
            args.model_name.replace("/", "_"), # Ensure sanitization for standalone test
            args.llm_provider,
            args.construction_mode
        )
