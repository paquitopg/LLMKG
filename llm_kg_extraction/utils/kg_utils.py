from typing import List, Dict, Tuple, Set

def create_entity_key(entity: Dict) -> str:
    """
    Create a unique key for an entity based on its semantic properties.
    
    Args:
        entity (Dict): The entity to create a key for.
        
    Returns:
        str: A unique key for the entity.
    """
    if not entity:
        return ""
    
    # Extract type and attributes
    entity_type = entity.get('type', '').lower()
    
    # Special handling for financial metrics to catch similar metrics with different names
    if entity_type == 'pekg:financialmetric':
        return create_financial_metric_key(entity)
    elif entity_type == 'pekg:kpi':
        return create_kpi_key(entity)
    elif entity_type == 'pekg:headcountmetric':
        return create_headcount_metric_key(entity)
    
    # Default handling for other entity types
    entity_name = entity.get('name', '')
    
    # If we have both type and name, use them as the key
    if entity_type and entity_name:
        # Normalize case for better matching - always use lowercase for the key
        return f"{entity_type}:{entity_name.lower()}"
    
    # Fallback to ID if name is not available
    entity_id = entity.get('id', '')
    if entity_type and entity_id:
        # Also normalize ID to lowercase for the key
        return f"{entity_type}:{entity_id.lower()}"
    
    # Last resort, just use the ID (also normalized)
    return entity_id.lower() if entity_id else ""

def create_financial_metric_key(entity: Dict) -> str:
    """
    Create a key for financial metrics that groups similar metrics together.
    
    Args:
        entity (Dict): The financial metric entity.
        
    Returns:
        str: A semantic key for the financial metric.
    """
    name = entity.get('name', '').lower()
    
    # Extract key components from the name
    fiscal_year = extract_fiscal_year(name)
    metric_type = extract_metric_type(name)
    
    # Include numerical values for better matching
    percentage_value = entity.get('percentageValue')
    metric_value = entity.get('metricValue')
    metric_currency = entity.get('metricCurrency', '').lower()
    metric_unit = entity.get('metricUnit', '').lower()
    
    # Create a composite key from the essential attributes
    key_components = []
    key_components.append('pekg:financialmetric')
    
    if metric_type:
        key_components.append(f"type:{metric_type}")
    
    if fiscal_year:
        key_components.append(f"year:{fiscal_year}")
    
    # Add value components - these are crucial for identifying identical metrics
    if percentage_value is not None:
        key_components.append(f"percent:{percentage_value}")
    elif metric_value is not None:
        # Normalize large numbers to K or M for better matching
        normalized_value = normalize_metric_value(metric_value)
        key_components.append(f"value:{normalized_value}")
        
        if metric_currency:
            key_components.append(f"currency:{metric_currency}")
    
    if metric_unit:
        key_components.append(f"unit:{metric_unit}")
    
    # Join all components to create a unique key
    return ":".join(key_components)

def extract_fiscal_year(name: str) -> str:
    """
    Extract fiscal year information from a metric name.
    
    Args:
        name (str): The metric name.
        
    Returns:
        str: The extracted fiscal year or empty string.
    """
    import re
    
    # Common patterns for fiscal years in financial documents
    patterns = [
        r'fy(\d{2,4})a',   # FY21A, FY2021A
        r'fy(\d{2,4})e',   # FY21E, FY2021E
        r'fy(\d{2,4})bp',  # FY21BP, FY2021BP
        r'fy(\d{2,4})l',   # FY21L, FY2021L
        r'fy(\d{2,4})',    # FY21, FY2021
        r'20(\d{2})'       # 2021
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(0)
    
    return ""

def extract_metric_type(name: str) -> str:
    """
    Extract the core metric type from a financial metric name.
    
    Args:
        name (str): The metric name.
        
    Returns:
        str: A normalized metric type.
    """
    name = name.lower()
    
    # Map similar metric terms to canonical forms
    metric_mappings = {
        'revenue mix recurring': 'recurring revenue share',
        'recurring revenue mix': 'recurring revenue share',
        'recurring revenue share': 'recurring revenue share',
        'revenue mix non-recurring': 'non-recurring revenue share',
        'non-recurring revenue mix': 'non-recurring revenue share',
        'non-recurring revenue share': 'non-recurring revenue share',
        'recurring revenue': 'recurring revenue',
        'total revenue': 'total revenue',
        'revenue': 'total revenue',
        'adj. ebitda': 'adj ebitda',
        'adjusted ebitda': 'adj ebitda',
        'ebitda': 'ebitda',
        'cagr': 'cagr'
    }
    
    # Try to match the name against known metric types
    for pattern, canonical_form in metric_mappings.items():
        if pattern in name:
            return canonical_form
    
    # Check for percentage metrics
    if 'share' in name or 'mix' in name or '%' in name or 'percent' in name or 'margin' in name:
        if 'recurring' in name:
            return 'recurring revenue share'
        elif 'non-recurring' in name or 'nonrecurring' in name:
            return 'non-recurring revenue share'
        elif 'margin' in name:
            return 'margin'
        else:
            return 'share'
    
    return name

def normalize_metric_value(value: float) -> str:
    """
    Normalize metric values to a standard format for comparison.
    
    Args:
        value (float): The metric value.
        
    Returns:
        str: A normalized string representation of the value.
    """
    # Very small values might be percentages without the % symbol
    if 0 < value < 100:
        return f"{int(value)}" if value.is_integer() else f"{value}"
    
    # Convert to K for thousands
    if 1000 <= value < 1000000:
        return f"{int(value/1000)}k"
    
    # Convert to M for millions
    if 1000000 <= value < 1000000000:
        return f"{int(value/1000000)}m"
    
    # Convert to B for billions
    if value >= 1000000000:
        return f"{int(value/1000000000)}b"
    
    # Return as is for other values
    return str(int(value)) if value.is_integer() else str(value)

def create_kpi_key(entity: Dict) -> str:
    """
    Create a semantic key for KPI entities.
    
    Args:
        entity (Dict): The KPI entity.
        
    Returns:
        str: A semantic key for the KPI.
    """
    name = entity.get('name', '').lower()
    metric_value = entity.get('metricValue')
    metric_unit = entity.get('metricUnit', '').lower()
    
    # Normalize common KPI names
    name_mappings = {
        'client count': 'clients',
        'number of clients': 'clients',
        'clients': 'clients',
        'annual billings per client': 'billings per client',
        'users trained per year': 'users trained',
        'licensing models': 'licensing model',
        'licensing model flexibility': 'licensing model'
    }
    
    normalized_name = None
    for pattern, canonical_form in name_mappings.items():
        if pattern in name:
            normalized_name = canonical_form
            break
    
    if not normalized_name:
        normalized_name = name
    
    # Create a composite key
    key_components = ['pekg:kpi', f"name:{normalized_name}"]
    
    if metric_value is not None:
        normalized_value = normalize_metric_value(metric_value)
        key_components.append(f"value:{normalized_value}")
    
    if metric_unit:
        key_components.append(f"unit:{metric_unit}")
    
    return ":".join(key_components)

def create_headcount_metric_key(entity: Dict) -> str:
    """
    Create a semantic key for headcount metric entities.
    
    Args:
        entity (Dict): The headcount metric entity.
        
    Returns:
        str: A semantic key for the headcount metric.
    """
    headcount_value = entity.get('headcountValue')
    name = entity.get('name', '').lower()
    
    # Extract year information if present
    fiscal_year = extract_fiscal_year(name)
    
    key_components = ['pekg:headcountmetric']
    
    if headcount_value is not None:
        key_components.append(f"value:{headcount_value}")
    
    if fiscal_year:
        key_components.append(f"year:{fiscal_year}")
    
    return ":".join(key_components)

def get_entity_id_mapping(entities1: List[Dict], entities2: List[Dict]) -> Dict[str, str]:
    """
    Create a mapping of entity IDs from the second list to matching entities in the first list.
    
    Args:
        entities1 (List[Dict]): First list of entities (the base entities).
        entities2 (List[Dict]): Second list of entities to map to the first list.
        
    Returns:
        Dict[str, str]: Mapping from entities2 IDs to matching entities1 IDs.
    """
    # Create a lookup dictionary for entities in the first list
    entity1_keys = {}
    for entity in entities1:
        key = create_entity_key(entity)
        if key:  # Only add if key is not empty
            entity1_keys[key] = entity.get('id', '')
    
    # Map entities from the second list to the first
    id_mapping = {}
    for entity in entities2:
        key = create_entity_key(entity)
        if key and key in entity1_keys:
            id_mapping[entity.get('id', '')] = entity1_keys[key]
    
    return id_mapping

def merge_entity_attributes(entity1: Dict, entity2: Dict) -> Dict:
    """
    Merge attributes from two entities, using a more sophisticated strategy
    to combine information from both.
    
    Args:
        entity1 (Dict): The primary entity.
        entity2 (Dict): The secondary entity with attributes to merge.
        
    Returns:
        Dict: A new entity with merged attributes.
    """
    # Start with a copy of the first entity
    result = entity1.copy()
    entity_type = entity1.get('type', '').lower()
    
    # For different entity types, use different merging strategies
    if entity_type == 'pekg:financialmetric':
        merge_financial_metric_attributes(result, entity2)
    elif entity_type == 'pekg:kpi':
        merge_kpi_attributes(result, entity2)
    elif entity_type == 'pekg:headcountmetric':
        merge_headcount_metric_attributes(result, entity2)
    else:
        # For other entity types, use a general merging strategy
        for key, value in entity2.items():
            # Skip merging the ID field
            if key == 'id':
                continue
                
            # If the attribute doesn't exist in result or is empty, add it
            if key not in result or not result[key]:
                result[key] = value
            # For name fields, prefer the longer or more specific name
            elif key == 'name' and value and len(value) > len(result[key]):
                result[key] = value
    
    return result

def merge_financial_metric_attributes(result: Dict, entity: Dict) -> None:
    """
    Merge attributes for financial metric entities.
    
    Args:
        result (Dict): The result entity being built (modified in place).
        entity (Dict): The entity to merge in.
    """
    # For financial metrics, we want to merge numerical values carefully
    for key, value in entity.items():
        if key == 'id':
            continue
            
        if key == 'name':
            # Keep the more descriptive name (often the longer one)
            current_name = result.get('name', '')
            new_name = value
            
            if new_name and (not current_name or len(new_name) > len(current_name)):
                result[key] = new_name
        elif key in ['metricValue', 'percentageValue']:
            # For metric values, keep the more precise value (the one with more digits)
            if key not in result or value is not None:
                if key not in result or (result[key] is None):
                    result[key] = value
                elif isinstance(value, (int, float)) and isinstance(result[key], (int, float)):
                    # Determine which value has more precision
                    current_str = str(result[key])
                    new_str = str(value)
                    
                    # If the new value has more digits, use it
                    if len(new_str) > len(current_str):
                        result[key] = value
                    # If same number of digits but one is more precise (has decimals)
                    elif len(new_str) == len(current_str) and '.' in new_str and '.' not in current_str:
                        result[key] = value
        elif key in ['metricCurrency', 'metricUnit']:
            # Keep the more specific unit or currency
            if key not in result or not result[key]:
                result[key] = value
            elif value and len(value) > len(result[key]):
                result[key] = value
        else:
            # For other attributes, only add if not present
            if key not in result or not result[key]:
                result[key] = value

def merge_kpi_attributes(result: Dict, entity: Dict) -> None:
    """
    Merge attributes for KPI entities.
    
    Args:
        result (Dict): The result entity being built (modified in place).
        entity (Dict): The entity to merge in.
    """
    for key, value in entity.items():
        if key == 'id':
            continue
            
        if key == 'name':
            # For KPIs, keep the more descriptive name
            current_name = result.get('name', '')
            new_name = value
            
            if new_name and (not current_name or len(new_name) > len(current_name)):
                result[key] = new_name
        elif key == 'metricValue':
            # Keep non-null metric values
            if key not in result or result[key] is None:
                result[key] = value
        elif key == 'metricUnit':
            # Keep the more specific unit
            if key not in result or not result[key]:
                result[key] = value
        else:
            # For other attributes, only add if not present
            if key not in result or not result[key]:
                result[key] = value

def merge_headcount_metric_attributes(result: Dict, entity: Dict) -> None:
    """
    Merge attributes for headcount metric entities.
    
    Args:
        result (Dict): The result entity being built (modified in place).
        entity (Dict): The entity to merge in.
    """
    for key, value in entity.items():
        if key == 'id':
            continue
            
        if key == 'name':
            # Keep the more descriptive name
            current_name = result.get('name', '')
            new_name = value
            
            if new_name and (not current_name or len(new_name) > len(current_name)):
                result[key] = new_name
        elif key == 'headcountValue':
            # Keep non-null headcount values
            if key not in result or result[key] is None:
                result[key] = value
        else:
            # For other attributes, only add if not present
            if key not in result or not result[key]:
                result[key] = value

def merge_knowledge_graphs(graph1: Dict, graph2: Dict) -> Dict:
    """
    Merge two knowledge graphs, combining entities with the same semantic key.
    
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
    
    # Create a dictionary of entities by their semantic key
    entity_dict = {}
    entity_id_to_key = {}  # Maps entity IDs to their semantic keys
    
    # First process entities from the first graph
    for entity in entities1:
        entity_key = create_entity_key(entity)
        if entity_key:
            entity_dict[entity_key] = entity.copy()
            entity_id_to_key[entity['id']] = entity_key
    
    # Map entity IDs from graph2 to corresponding entities in graph1
    id_mapping = {}
    
    # Process entities from the second graph
    for entity in entities2:
        entity_key = create_entity_key(entity)
        entity_id = entity.get('id', '')
        
        if entity_key:
            if entity_key in entity_dict:
                # We found a match - update the mapping and merge attributes
                original_entity_id = entity_dict[entity_key]['id']
                id_mapping[entity_id] = original_entity_id
                entity_dict[entity_key] = merge_entity_attributes(entity_dict[entity_key], entity)
            else:
                # This is a new entity
                entity_dict[entity_key] = entity.copy()
                entity_id_to_key[entity_id] = entity_key
    
    # Process relationships from both graphs
    all_relationships = []
    relationship_set = set()
    
    # First add relationships from the first graph
    for rel in relationships1:
        source = rel['source']
        target = rel['target']
        rel_type = rel['type']
        
        # Check if this relationship already exists
        key = (source, target, rel_type)
        if key not in relationship_set:
            relationship_set.add(key)
            all_relationships.append(rel.copy())
    
    # Then process relationships from the second graph
    for rel in relationships2:
        # Get the original entity IDs if the entity was merged
        source = id_mapping.get(rel['source'], rel['source'])
        target = id_mapping.get(rel['target'], rel['target'])
        rel_type = rel['type']
        
        # Check if this relationship already exists
        key = (source, target, rel_type)
        if key not in relationship_set:
            relationship_set.add(key)
            new_rel = rel.copy()
            new_rel['source'] = source
            new_rel['target'] = target
            all_relationships.append(new_rel)
    
    # Return the merged graph
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
    result_graph = base_graph.copy() if base_graph else {"entities": [], "relationships": []}
    
    for graph in graphs_to_add:
        result_graph = merge_knowledge_graphs(result_graph, graph)
    
    return result_graph

def merge_multiple_knowledge_graphs(graphs: List[Dict]) -> Dict:
    """
    Merge multiple knowledge graphs into a single unified graph.
    Used in onego mode to merge independent page analysis results.
    
    Args:
        graphs (List[Dict]): List of knowledge graphs to merge.
        
    Returns:
        Dict: Merged knowledge graph.
    """
    if not graphs:
        return {"entities": [], "relationships": []}
    
    # Start with the first graph and merge the rest into it
    result_graph = graphs[0].copy() if graphs else {"entities": [], "relationships": []}
    
    for graph in graphs[1:]:
        result_graph = merge_knowledge_graphs(result_graph, graph)
    
    # Normalize entity IDs for consistency
    return normalize_entity_ids(clean_knowledge_graph(result_graph))

def normalize_entity_ids(graph: Dict) -> Dict:
    """
    Normalize entity IDs to ensure they follow a consistent pattern (e.g., e1, e2, ...)
    and update all relationship references accordingly.
    
    Args:
        graph (Dict): The knowledge graph to normalize.
        
    Returns:
        Dict: The normalized knowledge graph.
    """
    entities = graph.get('entities', [])
    relationships = graph.get('relationships', [])
    
    # Create case-insensitive map for finding entities
    entity_map = {}
    for entity in entities:
        entity_id = entity.get('id', '')
        if entity_id:
            entity_map[entity_id.lower()] = entity_id
    
    # Create new IDs for entities
    id_mapping = {}
    new_entities = []
    
    for i, entity in enumerate(entities):
        old_id = entity.get('id', '')
        new_id = f"e{i+1}"  # Start from e1
        id_mapping[old_id] = new_id
        id_mapping[old_id.lower()] = new_id  # Also add lowercase mapping
        
        # Update the entity with the new ID
        new_entity = entity.copy()
        new_entity['id'] = new_id
        new_entities.append(new_entity)
    
    # Update relationship references
    new_relationships = []
    for rel in relationships:
        source = rel.get('source', '')
        target = rel.get('target', '')
        
        # Try to find the mapping using exact case first, then try lowercase
        source_new = id_mapping.get(source) or id_mapping.get(source.lower(), source)
        target_new = id_mapping.get(target) or id_mapping.get(target.lower(), target)
        
        new_rel = rel.copy()
        new_rel['source'] = source_new
        new_rel['target'] = target_new
        new_relationships.append(new_rel)
    
    return {
        "entities": new_entities,
        "relationships": new_relationships
    }

def clean_knowledge_graph(graph: Dict) -> Dict:
    """
    Clean a knowledge graph by removing any relationships referencing
    non-existent entities, removing duplicate relationships, and standardizing
    entity IDs in relationships to match the case of the entity IDs.
    
    Args:
        graph (Dict): The knowledge graph to clean.
        
    Returns:
        Dict: The cleaned knowledge graph.
    """
    entities = graph.get('entities', [])
    relationships = graph.get('relationships', [])
    
    # Get map of entity IDs (with case preservation)
    entity_ids = {entity.get('id'): entity.get('id') for entity in entities}
    
    # Create a case-insensitive lookup map
    entity_ids_lower = {id.lower(): id for id in entity_ids.values()}
    
    # Filter out invalid relationships and fix case issues
    valid_relationships = []
    relationship_set = set()
    
    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        rel_type = rel.get('type')
        
        # Try to resolve case-insensitive matches for source and target
        if source and source.lower() in entity_ids_lower:
            source = entity_ids_lower[source.lower()]
            
        if target and target.lower() in entity_ids_lower:
            target = entity_ids_lower[target.lower()]
            
        # Check if both source and target entities exist after case correction
        if source in entity_ids and target in entity_ids:
            # Check for duplicates
            key = (source, target, rel_type)
            if key not in relationship_set:
                relationship_set.add(key)
                # Update relationship with corrected IDs
                corrected_rel = rel.copy()
                corrected_rel['source'] = source
                corrected_rel['target'] = target
                valid_relationships.append(corrected_rel)
    
    return {
        "entities": entities,
        "relationships": valid_relationships
    }