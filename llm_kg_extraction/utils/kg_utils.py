from typing import List, Dict, Tuple, Set, Any, Optional
import re
from difflib import SequenceMatcher

def similarity_score(str1: str, str2: str) -> float:
    """
    Calculate string similarity using SequenceMatcher.
    
    Args:
        str1 (str): First string
        str2 (str): Second string
        
    Returns:
        float: Similarity score between 0 and 1
    """
    if not str1 or not str2:
        return 0.0
    
    # Convert to lowercase for better matching
    str1 = str1.lower()
    str2 = str2.lower()
    
    # Calc similarity score
    return SequenceMatcher(None, str1, str2).ratio()

def normalize_text(text: str) -> str:
    """
    Normalize text for better matching by removing punctuation,
    extra spaces, and converting to lowercase.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace punctuation and special characters with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces
    return text.strip()

def extract_numerical_values(entity: Dict) -> Dict[str, Any]:
    """
    Extract all numerical values from an entity.
    
    Args:
        entity (Dict): Entity to extract values from
        
    Returns:
        Dict[str, Any]: Dictionary of numerical values
    """
    numerical_values = {}
    
    # Common numerical fields in the ontology
    numerical_fields = [
        'metricValue', 'percentageValue', 'headcountValue', 
        'roundAmount', 'valuation', 'revenueAmount'
    ]
    
    for field in numerical_fields:
        if field in entity and entity[field] is not None:
            numerical_values[field] = entity[field]
    
    return numerical_values

def get_entity_temporal_info(entity: Dict) -> Dict[str, Any]:
    """
    Extract temporal information (years, dates, periods) from an entity.
    
    Args:
        entity (Dict): Entity to extract temporal info from
        
    Returns:
        Dict[str, Any]: Dictionary of temporal information
    """
    temporal_info = {}
    
    # Common temporal fields
    time_fields = ['year', 'date', 'period', 'roundDate', 'eventYear']
    
    for field in time_fields:
        if field in entity and entity[field]:
            temporal_info[field] = entity[field]
    
    # Try to extract year from name if it exists
    name = entity.get('name', '')
    if name:
        # Look for patterns like "FY21", "2020", "FY2021", etc.
        year_patterns = [
            r'FY(\d{2})([A-Za-z])',  # FY21A, FY22E, etc.
            r'FY(\d{4})([A-Za-z])',  # FY2021A, FY2022E, etc.
            r'FY(\d{2,4})',          # FY21, FY2021
            r'20(\d{2})'             # 2021, 2022, etc.
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, name)
            if match and 'extractedYear' not in temporal_info:
                temporal_info['extractedYear'] = match.group(0)
    
    return temporal_info

def create_generic_entity_key(entity: Dict, include_numerical: bool = False, 
                             include_temporal: bool = False) -> str:
    """
    Create a generic entity key based on entity type and name,
    with options to include numerical and temporal information.
    
    Args:
        entity (Dict): Entity to create key for
        include_numerical (bool): Whether to include numerical values in the key
        include_temporal (bool): Whether to include temporal information in the key
        
    Returns:
        str: Entity key
    """
    if not entity:
        return ""
    
    entity_type = entity.get('type', '').lower()
    entity_name = normalize_text(entity.get('name', ''))
    
    components = [entity_type]
    
    if entity_name:
        # For products, keep more words for better differentiation
        if 'product' in entity_type:
            components.append(f"name:{entity_name}")
        else:
            # Extract key terms from name to allow for more flexible matching
            key_terms = extract_key_terms(entity_name, entity_type)
            if key_terms:
                components.append(f"terms:{'-'.join(key_terms)}")
            else:
                components.append(f"name:{entity_name}")
    
    # Include numerical values if requested
    if include_numerical:
        numerical_values = extract_numerical_values(entity)
        for key, value in sorted(numerical_values.items()):
            components.append(f"{key}:{value}")
    
    # Include temporal information if requested
    if include_temporal:
        temporal_info = get_entity_temporal_info(entity)
        for key, value in sorted(temporal_info.items()):
            if isinstance(value, str):
                components.append(f"{key}:{value.lower()}")
            else:
                components.append(f"{key}:{value}")
    
    return ":".join(components)

def extract_key_terms(text: str, entity_type: str) -> List[str]:
    """
    Extract key terms from text based on entity type.
    
    Args:
        text (str): Text to extract terms from
        entity_type (str): Type of entity
        
    Returns:
        List[str]: List of key terms
    """
    if not text:
        return []
    
    # Split into words
    words = text.split()
    
    # For short names (1-3 words), use the whole name
    if len(words) <= 3:
        return [text]
    
    # For different entity types, extract relevant terms
    if 'financial' in entity_type or 'metric' in entity_type or 'kpi' in entity_type:
        # For financial metrics, extract the type of metric and fiscal year if present
        financial_terms = []
        
        # Common financial terms to look for
        key_financial_terms = [
            'revenue', 'recurring', 'non-recurring', 'ebitda', 'margin',
            'profit', 'sales', 'growth', 'cagr', 'share', 'mix'
        ]
        
        # Add key financial terms found in the text
        for term in key_financial_terms:
            if term in text:
                financial_terms.append(term)
        
        # Look for fiscal year patterns
        fiscal_year_match = re.search(r'(fy\d{2,4}[a-z]?|20\d{2})', text)
        if fiscal_year_match:
            financial_terms.append(fiscal_year_match.group(0))
        
        return financial_terms if financial_terms else [text]
    
    elif 'company' in entity_type:
        # For companies, use the whole name
        return [text]
    
    elif 'product' in entity_type:
        # For products, find important product terms
        product_terms = []
        important_product_words = [
            'platform', 'solution', 'software', 'service', 'api',
            'system', 'application', 'tool', 'framework', 'module'
        ]
        
        for word in important_product_words:
            if word in text:
                product_terms.append(word)
        
        return product_terms if product_terms else [text]
    
    # Default: return the first 2-3 significant words
    significant_words = [w for w in words if len(w) > 3][:3]
    return [' '.join(significant_words)] if significant_words else [text]

def are_entities_similar(entity1: Dict, entity2: Dict, 
                        threshold: float = 0.8, 
                        check_numerical: bool = True,
                        check_temporal: bool = True) -> bool:
    """
    Determine if two entities are similar enough to be considered the same.
    
    Args:
        entity1 (Dict): First entity
        entity2 (Dict): Second entity
        threshold (float): Similarity threshold (0-1)
        check_numerical (bool): Whether to check numerical values
        check_temporal (bool): Whether to check temporal information
        
    Returns:
        bool: True if entities are similar, False otherwise
    """
    # Must be same entity type
    if entity1.get('type', '').lower() != entity2.get('type', '').lower():
        return False
    
    entity_type = entity1.get('type', '').lower()
    
    # Get names
    name1 = normalize_text(entity1.get('name', ''))
    name2 = normalize_text(entity2.get('name', ''))
    
    # If both have names, check name similarity
    if name1 and name2:
        # Direct name similarity check
        name_similarity = similarity_score(name1, name2)
        
        # If names are very similar, they're likely the same entity
        if name_similarity >= threshold:
            return True
        
        # Extract key terms for more flexible matching
        terms1 = extract_key_terms(name1, entity_type)
        terms2 = extract_key_terms(name2, entity_type)
        
        # Check if any key terms match with high similarity
        for term1 in terms1:
            for term2 in terms2:
                term_similarity = similarity_score(term1, term2)
                if term_similarity >= threshold:
                    # If key terms match and we have numerical or temporal checks enabled
                    if check_numerical or check_temporal:
                        return check_additional_attributes(
                            entity1, entity2, 
                            check_numerical=check_numerical,
                            check_temporal=check_temporal
                        )
                    return True
    
    # Handle entities without names
    if not name1 and not name2:
        # For entities without names, rely on other attributes
        return check_additional_attributes(
            entity1, entity2,
            check_numerical=True,  # Always check numerical for unnamed entities
            check_temporal=True    # Always check temporal for unnamed entities
        )
    
    return False

def check_additional_attributes(entity1: Dict, entity2: Dict, 
                               check_numerical: bool = True,
                               check_temporal: bool = True) -> bool:
    """
    Check additional attributes (numerical values, temporal info) to 
    determine if entities are similar.
    
    Args:
        entity1 (Dict): First entity
        entity2 (Dict): Second entity
        check_numerical (bool): Whether to check numerical values
        check_temporal (bool): Whether to check temporal information
        
    Returns:
        bool: True if additional attributes indicate similarity
    """
    # Check numerical values if requested
    if check_numerical:
        num_values1 = extract_numerical_values(entity1)
        num_values2 = extract_numerical_values(entity2)
        
        # If both have numerical values, they should match
        if num_values1 and num_values2:
            # Check for matching fields
            common_fields = set(num_values1.keys()) & set(num_values2.keys())
            
            # If there are common fields, values should be similar
            if common_fields:
                for field in common_fields:
                    val1 = num_values1[field]
                    val2 = num_values2[field]
                    
                    # Allow small differences for floating point values
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # For percentages, round to nearest integer
                        if 'percentage' in field and abs(val1 - val2) <= 1:
                            continue
                        
                        # For monetary values, allow 5% difference
                        if 'value' in field or 'amount' in field:
                            max_val = max(abs(val1), abs(val2))
                            if max_val > 0 and abs(val1 - val2) / max_val <= 0.05:
                                continue
                        
                        # For counts, allow exact matches only
                        if val1 != val2:
                            return False
                
                # If we get here, all common numerical fields are similar enough
                return True
    
    # Check temporal information if requested
    if check_temporal:
        time_info1 = get_entity_temporal_info(entity1)
        time_info2 = get_entity_temporal_info(entity2)
        
        # If both have temporal info, they should match
        if time_info1 and time_info2:
            # Check for matching fields
            common_fields = set(time_info1.keys()) & set(time_info2.keys())
            
            # If there are common fields, values should match
            if common_fields:
                for field in common_fields:
                    if time_info1[field] != time_info2[field]:
                        return False
                
                # If we get here, all common temporal fields match
                return True
    
    # If we don't have enough additional attributes to compare, 
    # be conservative and return False
    return False

def find_matching_entity(entity: Dict, entities: List[Dict], 
                        threshold: float = 0.75) -> Optional[Dict]:
    """
    Find a matching entity in a list of entities.
    
    Args:
        entity (Dict): Entity to find a match for
        entities (List[Dict]): List of entities to search in
        threshold (float): Similarity threshold
        
    Returns:
        Optional[Dict]: Matching entity if found, None otherwise
    """
    entity_type = entity.get('type', '').lower()
    
    # Use different thresholds for different entity types
    if 'company' in entity_type:
        # Companies need a higher match threshold
        threshold = 0.9
    elif 'product' in entity_type:
        # Products need a higher match threshold too
        threshold = 0.85
    elif 'metric' in entity_type or 'kpi' in entity_type:
        # Financial metrics can use a lower threshold with additional checks
        threshold = 0.7
    
    for candidate in entities:
        if are_entities_similar(entity, candidate, threshold=threshold):
            return candidate
    
    return None

def merge_entity_attributes(entity1: Dict, entity2: Dict) -> Dict:
    """
    Merge two entities' attributes, keeping the most comprehensive information.
    
    Args:
        entity1 (Dict): First entity
        entity2 (Dict): Second entity
        
    Returns:
        Dict: Merged entity
    """
    # Start with a copy of entity1
    result = entity1.copy()
    
    # Merge attributes from entity2
    for key, value in entity2.items():
        # Skip ID field
        if key == 'id':
            continue
        
        # Handle name field specially - keep the more descriptive name
        if key == 'name' and value:
            current_name = result.get('name', '')
            new_name = value
            
            # Use the longer name, or keep the first one if equal length
            if new_name and (not current_name or len(new_name) > len(current_name)):
                result[key] = new_name
                
        # Handle numerical values - choose the most precise
        elif key in ['metricValue', 'percentageValue', 'headcountValue', 'valuation', 'roundAmount']:
            if key not in result or result[key] is None:
                result[key] = value
            elif isinstance(value, (int, float)) and isinstance(result[key], (int, float)):
                # Choose the more precise value (the one with more digits)
                str_val1 = str(result[key])
                str_val2 = str(value)
                
                if len(str_val2) > len(str_val1):
                    result[key] = value
        
        # For other fields, only add if not present or empty
        elif key not in result or not result[key]:
            result[key] = value
    
    return result

def merge_knowledge_graphs(graph1: Dict, graph2: Dict) -> Dict:
    """
    Merge two knowledge graphs, combining similar entities.
    
    Args:
        graph1 (Dict): First knowledge graph
        graph2 (Dict): Second knowledge graph
        
    Returns:
        Dict: Merged knowledge graph
    """
    # Get entities and relationships from both graphs
    entities1 = graph1.get('entities', [])
    entities2 = graph2.get('entities', [])
    
    relationships1 = graph1.get('relationships', [])
    relationships2 = graph2.get('relationships', [])
    
    # Build a list of merged entities
    merged_entities = []
    id_mapping = {}  # Maps entity IDs from graph2 to corresponding entities in merged_entities
    
    # First, add all entities from graph1
    for entity in entities1:
        merged_entities.append(entity.copy())
    
    # Then try to merge entities from graph2
    for entity in entities2:
        entity_id = entity.get('id', '')
        
        # Look for a matching entity in merged_entities
        matching_entity = find_matching_entity(entity, merged_entities)
        
        if matching_entity:
            # Found a match, update the mapping and merge attributes
            matching_id = matching_entity.get('id', '')
            id_mapping[entity_id] = matching_id
            
            # Find the matching entity in our merged_entities list and update it
            for i, e in enumerate(merged_entities):
                if e.get('id') == matching_id:
                    merged_entities[i] = merge_entity_attributes(e, entity)
                    break
        else:
            # No match found, add as a new entity
            merged_entities.append(entity.copy())
            # Note: we don't need to update id_mapping here as the ID remains the same
    
    # Merge relationships
    merged_relationships = []
    relationship_set = set()  # To track unique relationships
    
    # Add relationships from graph1
    for rel in relationships1:
        source = rel.get('source', '')
        target = rel.get('target', '')
        rel_type = rel.get('type', '')
        
        rel_key = (source, target, rel_type)
        if rel_key not in relationship_set:
            relationship_set.add(rel_key)
            merged_relationships.append(rel.copy())
    
    # Add relationships from graph2, updating entity references
    for rel in relationships2:
        source = rel.get('source', '')
        target = rel.get('target', '')
        rel_type = rel.get('type', '')
        
        # Update source and target if they've been mapped
        source_new = id_mapping.get(source, source)
        target_new = id_mapping.get(target, target)
        
        rel_key = (source_new, target_new, rel_type)
        if rel_key not in relationship_set:
            relationship_set.add(rel_key)
            
            new_rel = rel.copy()
            new_rel['source'] = source_new
            new_rel['target'] = target_new
            merged_relationships.append(new_rel)
    
    return {
        'entities': merged_entities,
        'relationships': merged_relationships
    }

def merge_multiple_knowledge_graphs(graphs: List[Dict]) -> Dict:
    """
    Merge multiple knowledge graphs into one.
    
    Args:
        graphs (List[Dict]): List of knowledge graphs
        
    Returns:
        Dict: Merged knowledge graph
    """
    if not graphs:
        return {'entities': [], 'relationships': []}
    
    # Start with the first graph
    result = graphs[0].copy() if graphs else {'entities': [], 'relationships': []}
    
    # Merge each remaining graph into the result
    for graph in graphs[1:]:
        result = merge_knowledge_graphs(result, graph)
    
    return result

def normalize_entity_ids(graph: Dict) -> Dict:
    """
    Normalize entity IDs to a consistent pattern (e1, e2, etc.) and
    update relationship references accordingly.
    
    Args:
        graph (Dict): Knowledge graph
        
    Returns:
        Dict: Knowledge graph with normalized IDs
    """
    entities = graph.get('entities', [])
    relationships = graph.get('relationships', [])
    
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
        'entities': new_entities,
        'relationships': new_relationships
    }

def clean_knowledge_graph(graph: Dict) -> Dict:
    """
    Clean a knowledge graph by removing invalid relationships and
    standardizing entity references.
    
    Args:
        graph (Dict): Knowledge graph
        
    Returns:
        Dict: Cleaned knowledge graph
    """
    entities = graph.get('entities', [])
    relationships = graph.get('relationships', [])
    
    # Create entity ID lookups
    entity_ids = {entity.get('id'): entity.get('id') for entity in entities}
    entity_ids_lower = {id.lower(): id for id in entity_ids.values()}
    
    # Filter and clean relationships
    valid_relationships = []
    relationship_set = set()
    
    for rel in relationships:
        source = rel.get('source', '')
        target = rel.get('target', '')
        rel_type = rel.get('type', '')
        
        # Try to resolve case-insensitive matches
        if source and source.lower() in entity_ids_lower:
            source = entity_ids_lower[source.lower()]
            
        if target and target.lower() in entity_ids_lower:
            target = entity_ids_lower[target.lower()]
        
        # Check if both source and target entities exist
        if source in entity_ids and target in entity_ids:
            key = (source, target, rel_type)
            if key not in relationship_set:
                relationship_set.add(key)
                
                corrected_rel = rel.copy()
                corrected_rel['source'] = source
                corrected_rel['target'] = target
                valid_relationships.append(corrected_rel)
    
    return {
        'entities': entities,
        'relationships': valid_relationships
    }