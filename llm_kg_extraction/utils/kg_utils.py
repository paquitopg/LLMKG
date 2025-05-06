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

