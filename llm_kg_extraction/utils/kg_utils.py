from typing import List, Dict

def merge_knowledge_graphs(graph1: Dict, graph2: Dict) -> Dict:
    """
    Merge two knowledge graphs.
    Args:
        graph1 (dict): The first knowledge graph.
        graph2 (dict): The second knowledge graph.
    Returns:
        dict: The merged knowledge graph.
        """
    entities1 = graph1.get('entities', [])
    entities2 = graph2.get('entities', [])
    
    relationships1 = graph1.get('relationships', [])
    relationships2 = graph2.get('relationships', [])
        
    entity_dict = {}
    for entity in entities1 + entities2:
        entity_id = entity.get('id')
        if entity_id in entity_dict:
            entity_dict[entity_id].update(entity)
        else:
            entity_dict[entity_id] = entity
  
    all_relationships = relationships1 + relationships2
        
    return {
        "entities": list(entity_dict.values()),
        "relationships": all_relationships
    }

def merge_graphs(base_graph: Dict, graphs_to_add: List[Dict]) -> Dict:
    """
    Merge multiple knowledge graphs into the base graph.
    Used in iterative mode to maintain entity consistency across pages.
    Args:
        base_graph (Dict): The base knowledge graph to merge into.
        graphs_to_add (List[Dict]): List of knowledge graphs to merge into the base.
    Returns:
        Dict: The merged knowledge graph.
    """
    entities = base_graph.get('entities', [])
    relationships = base_graph.get('relationships', [])

    entity_dict = {entity.get('id'): entity for entity in entities}
    relationship_set = {
        (rel['source'], rel['target'], rel['type']) for rel in relationships
    }
    merged_relationships = relationships.copy()
    for graph in graphs_to_add:
        for entity in graph.get('entities', []):
            entity_id = entity.get('id')
            if entity_id in entity_dict:
                entity_dict[entity_id].update(entity)
            else:
                entity_dict[entity_id] = entity

        for rel in graph.get('relationships', []):
            key = (rel['source'], rel['target'], rel['type'])
            if key not in relationship_set:
                relationship_set.add(key)
                merged_relationships.append(rel)

    merged_graph = {
        "entities": list(entity_dict.values()),
        "relationships": merged_relationships
    }
    return merged_graph

def merge_multiple_knowledge_graphs(self, graphs: List[Dict]) -> Dict:
    """
    Merge multiple knowledge graphs into a single unified graph.
    Used in onego mode to merge independent page analysis results.
    Args:
        graphs (List[Dict]): List of knowledge graphs to merge.
    Returns:
        Dict: Merged knowledge graph.
    """
    merged_graph = {"entities": [], "relationships": []}
    
    entity_map = {}  
    next_entity_id = 1
    next_rel_id = 1
    
    for graph in graphs:
        if "entities" in graph:
            for entity in graph["entities"]:
                entity_key = None
                if "name" in entity:
                    entity_key = f"{entity['type']}:{entity['name'].lower()}"
                elif "id" in entity:
                    entity_key = f"{entity['type']}:{entity['id']}"
                
                if entity_key:
                    if entity_key not in entity_map:
                        new_id = f"e{next_entity_id}"
                        next_entity_id += 1
                        old_id = entity["id"]
                        entity["id"] = new_id
                        [entity_key] = {"new_id": new_id, "old_id": old_id}
                           
                        merged_graph["entities"].append(entity)
            
        if "relationships" in graph:
            for rel in graph["relationships"]:
                source_old = rel["source"]
                target_old = rel["target"]
                 
                source_new = None
                target_new = None
                    
                for key, value in entity_map.items():
                    if value["old_id"] == source_old:
                        source_new = value["new_id"]
                    if value["old_id"] == target_old:
                        target_new = value["new_id"]

                if source_new and target_new:
                    new_rel = {
                        "id": f"r{next_rel_id}",
                        "source": source_new,
                        "target": target_new,
                        "type": rel["type"]
                    }
                    next_rel_id += 1
                    merged_graph["relationships"].append(new_rel)
        
    return merged_graph