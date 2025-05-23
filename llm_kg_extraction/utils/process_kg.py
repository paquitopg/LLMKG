import json

def link_isolated_market_metrics(data):
    """
    Finds isolated 'pekg:MarketMetric' nodes in a JSON graph and links them 
    to the 'Systran' company.

    Args:
        data (dict): The loaded JSON data representing the knowledge graph.

    Returns:
        dict: The modified JSON data with new relationships added.
    """
    if not isinstance(data, dict):
        raise TypeError("Input data must be a dictionary (parsed JSON).")
    if "entities" not in data or "relationships" not in data:
        raise ValueError("Input JSON must contain 'entities' and 'relationships' keys.")

    systran_id = None
    for entity in data["entities"]:
        if entity.get("name") == "Systran" and entity.get("type") == "pekg:Company":
            systran_id = entity.get("id")
            break

    if not systran_id:
        print("Error: 'Systran' company entity not found.")
        return data

    market_metric_ids = set()
    for entity in data["entities"]:
        if entity.get("type") == "pekg:MarketMetric":
            market_metric_ids.add(entity.get("id"))

    if not market_metric_ids:
        print("No 'pekg:MarketMetric' entities found.")
        return data

    # Collect all entity IDs that are already part of a relationship
    related_entity_ids = set()
    for rel in data["relationships"]:
        if "source" in rel:
            related_entity_ids.add(rel["source"])
        if "target" in rel:
            related_entity_ids.add(rel["target"])
        # Also consider if the metric is linked *from* something already
        # (though the request specifically asks to link *to* them from Systran)
        # For a truly isolated node, it shouldn't be a source or target.

    new_relationships_added = 0
    for metric_id in market_metric_ids:
        if metric_id not in related_entity_ids:
            # This market metric is isolated
            new_relationship = {
                "source": systran_id,
                "target": metric_id,
                "type": "pekg:hasMarketMetric"
            }
            data["relationships"].append(new_relationship)
            new_relationships_added +=1
            print(f"Added relationship: Systran ({systran_id}) -> {metric_id} (pekg:hasMarketMetric)")

    if new_relationships_added == 0:
        print("No isolated 'pekg:MarketMetric' nodes found to link.")
    else:
        print(f"\nSuccessfully added {new_relationships_added} new relationships.")
        
    return data

# Load the JSON data from the uploaded file
file_path = 'multimodal_kg_System_gemini-2.5-pro-preview-05-06_vertexai_iterative.json'

try:
    with open(file_path, 'r') as f:
        json_data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    json_data = None
except json.JSONDecodeError:
    print(f"Error: The file '{file_path}' contains invalid JSON.")
    json_data = None

if json_data:
    # Process the data
    modified_json_data = link_isolated_market_metrics(json_data)
    with open('modified_kg_data.json', 'w') as outfile:
        json.dump(modified_json_data, outfile, indent=2)
        print("\nModified data saved to 'modified_kg_data.json'")