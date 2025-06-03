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
    
    str1 = str(str1).lower() # Ensure string conversion
    str2 = str(str2).lower() # Ensure string conversion
    
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
    text = str(text) # Ensure it's a string
    text = text.lower()
    text = re.sub(r'[^\w\s.-]', ' ', text) # Keep alphanumeric, whitespace, dots, hyphens
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_numerical_values(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all numerical values from an entity based on a predefined list of common numerical fields.
    
    Args:
        entity (Dict): Entity to extract values from
        
    Returns:
        Dict[str, Any]: Dictionary of numerical values
    """
    numerical_values = {}
    # Predefined list of common numerical fields from pekg_ontology_streamlined_v1_2 and other observations
    numerical_fields = [
        'metricValue', 'percentageValue', 'headcountValue', 
        'roundAmount', 'valuation', 'revenueAmount', 'amount', 
        'parsedValue', 'parsedPercentage' 
    ]
    
    for field in numerical_fields:
        if field in entity and entity[field] is not None:
            try:
                val_str = str(entity[field]).replace(',', '') # Handle commas in numbers
                val = float(val_str)
                numerical_values[field] = val
            except (ValueError, TypeError):
                # Optionally log a warning if parsing fails for a field expected to be numerical
                # print(f"Warning: Could not parse numerical value for field '{field}' with value '{entity[field]}'")
                pass 
    return numerical_values

def get_entity_temporal_info(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract temporal information (years, dates, periods) from an entity.
    Prioritizes specific temporal fields from the ontology, then tries to parse from name fields.
    
    Args:
        entity (Dict): Entity to extract temporal info from
        
    Returns:
        Dict[str, Any]: Dictionary of temporal information, with 'extractedPeriod' as the primary consolidated field.
    """
    temporal_info = {}
    
    # Specific ontology fields for temporal data based on pekg_ontology_streamlined_v1_2
    ontology_time_fields = {
        "financialmetric": "fiscalPeriod",
        "operationalkpi": "kpiDateOrPeriod",
        "headcount": "dateOrYear", 
        "historicalevent": "dateOrYear",
        "company": "foundedYear" 
    }
    entity_type_unprefixed = entity.get('type', '').lower().split(':')[-1]
    specific_period_field = ontology_time_fields.get(entity_type_unprefixed)

    if specific_period_field and specific_period_field in entity and entity[specific_period_field] is not None:
        temporal_info['extractedPeriod'] = str(entity[specific_period_field])
    
    # Fallback: Try to extract year/period from name if no specific field was found or populated
    if 'extractedPeriod' not in temporal_info:
        name_to_search = entity.get('name', entity.get('metricName', entity.get('kpiName', ''))) #
        if name_to_search and isinstance(name_to_search, str):
            # Regex patterns to capture various year and period formats
            year_patterns = [
                r'\b(FY|CY|CAL\s*|H[1-2]|Q[1-4])?\s*(20\d{2}|19\d{2})\s*([A-Za-zEBPFL]*)?\b', 
                r'\b(20\d{2}|19\d{2})\s*-\s*(20\d{2}|19\d{2})\b([A-Za-z]*)', 
            ]
            for pattern in year_patterns:
                match = re.search(pattern, name_to_search, re.IGNORECASE)
                if match:
                    temporal_info['extractedPeriod'] = match.group(0).strip() # Take full match as period
                    break
    return temporal_info


def are_entities_similar(entity1: Dict[str, Any], entity2: Dict[str, Any], 
                        threshold: float = 0.8, 
                        check_numerical_values_for_similarity: bool = False,
                        check_temporal_for_similarity: bool = True) -> bool:
    """
    Determine if two entities are similar enough to be considered the same based on their intrinsic attributes.
    For FinancialMetric, requires high name similarity AND other attributes to be identical or very close.
    This function DOES NOT consider relational context directly, but can be guided by check_xxx flags.
    """

    type1_full = entity1.get('type', '').lower() if entity1.get('type') is not None else ""
    type2_full = entity2.get('type', '').lower() if entity2.get('type') is not None else ""

    if type1_full != type2_full and (type1_full == "" or type2_full == ""): # If one is None/empty and other isn't
        if type1_full != "" or type2_full != "": # Ensure not both are empty strings
             return False 
    elif type1_full != type2_full: # Both have types but they are different
        return False

    type1_unprefixed = type1_full.split(':')[-1]
    type2_unprefixed = type2_full.split(':')[-1]

    if type1_unprefixed != type2_unprefixed:
        return False
    
    entity_type_unprefixed = type1_unprefixed

    # Determine primary name field based on common patterns in the original code
    name1_str = entity1.get('metricName', entity1.get('name', entity1.get('kpiName', entity1.get('productName', entity1.get('fullName', ''))))) #
    name2_str = entity2.get('metricName', entity2.get('name', entity2.get('kpiName', entity2.get('productName', entity2.get('fullName', ''))))) #
    
    name1 = normalize_text(name1_str) #
    name2 = normalize_text(name2_str) #
    
    if not name1 and not name2: # If both names are empty
        # For financial metrics, a name is crucial
        if entity_type_unprefixed == "financialmetric": return False #
        # For other types, if names are empty, rely on additional attribute checks
        if check_temporal_for_similarity or check_numerical_values_for_similarity:
             return check_additional_attributes(entity1, entity2, 
                                                check_numerical=check_numerical_values_for_similarity, 
                                                check_temporal=check_temporal_for_similarity,
                                                strict_numerical_value_check=check_numerical_values_for_similarity) #
        return True # No names and no additional checks means they are considered similar (e.g. two anonymous entities of same type)
    
    if not name1 or not name2: # If one name is empty and the other is not
        return False

    name_similarity = similarity_score(name1, name2) #
    
    # Stricter rules for FinancialMetric entities
    if entity_type_unprefixed == "financialmetric": #
        if name_similarity < 0.95: # High threshold for name similarity
            return False

        # All other attributes must be identical or numerically very close
        keys1 = set(k for k in entity1.keys() if k not in ['id', 'type']) #
        keys2 = set(k for k in entity2.keys() if k not in ['id', 'type']) #

        if keys1 != keys2: # Must have the same set of attributes
            return False

        for key in keys1:
            val1 = entity1.get(key)
            val2 = entity2.get(key)

            if val1 is None and val2 is None: continue #
            if (val1 is None and val2 is not None) or \
               (val1 is not None and val2 is None): #
                return False

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) > 1e-9: return False # Strict numerical equality
            elif isinstance(val1, str) and isinstance(val2, str):
                if normalize_text(val1) != normalize_text(val2): return False #
            elif isinstance(val1, bool) and isinstance(val2, bool):
                if val1 != val2: return False #
            elif isinstance(val1, list) and isinstance(val2, list): # Simple list comparison
                if val1 != val2: return False #
            elif type(val1) != type(val2): return False #
            elif val1 != val2: return False # General comparison for other types
        return True # If all checks pass for FinancialMetric

    # Rules for other metric-like types
    metric_like_types = {"operationalkpi", "headcount"} #
    if entity_type_unprefixed in metric_like_types or 'metric' in entity_type_unprefixed: # Broader catch for "metric"
        metric_name_threshold = 0.90  # Slightly lower threshold than FinancialMetric
        if name_similarity >= metric_name_threshold: #
            # Check temporal consistency
            time_info1 = get_entity_temporal_info(entity1) #
            time_info2 = get_entity_temporal_info(entity2) #
            period1 = time_info1.get('extractedPeriod') #
            period2 = time_info2.get('extractedPeriod') #
            
            if period1 and period2 and normalize_text(str(period1)) != normalize_text(str(period2)): #
                return False
            
            # If specified, check numerical values strictly for these types
            if check_numerical_values_for_similarity:
                 return check_additional_attributes(entity1, entity2, 
                                                   check_numerical=True, 
                                                   check_temporal=False, # Already checked period consistency
                                                   strict_numerical_value_check=True) #
            return True 
        else:
            return False # Name similarity too low for metric-like type

    # General case for other entity types
    if name_similarity >= threshold: #
        if check_temporal_for_similarity or check_numerical_values_for_similarity:
            return check_additional_attributes(entity1, entity2, 
                                               check_numerical=check_numerical_values_for_similarity, 
                                               check_temporal=check_temporal_for_similarity,
                                               strict_numerical_value_check=check_numerical_values_for_similarity) #
        return True # Name similarity is sufficient
            
    return False

def check_additional_attributes(entity1: Dict[str, Any], entity2: Dict[str, Any], 
                               check_numerical: bool,
                               check_temporal: bool,
                               strict_numerical_value_check: bool) -> bool:
    """
    Helper to check non-name attributes (numerical, temporal) for consistency.
    """
    if check_temporal: #
        time_info1 = get_entity_temporal_info(entity1) #
        time_info2 = get_entity_temporal_info(entity2) #
        period1 = time_info1.get('extractedPeriod') #
        period2 = time_info2.get('extractedPeriod') #
        # If both have periods, they must match
        if period1 and period2 and normalize_text(str(period1)) != normalize_text(str(period2)): #
            return False 
    
    if check_numerical: #
        num_values1 = extract_numerical_values(entity1) #
        num_values2 = extract_numerical_values(entity2) #
        
        if strict_numerical_value_check: #
            all_num_keys = set(num_values1.keys()) | set(num_values2.keys()) #
            if not all_num_keys and (num_values1 or num_values2): # If one has numbers and the other doesn't, not strictly same
                return False

            for key in all_num_keys:
                val1 = num_values1.get(key)
                val2 = num_values2.get(key)

                if val1 is None and val2 is None: continue #
                # If one has a value and the other doesn't for the same key, they are different
                if (val1 is None and val2 is not None) or \
                   (val1 is not None and val2 is None): #
                    return False 

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if abs(val1 - val2) > 1e-9: return False # Strict numerical equality
            # If all numerical values present in one are identical in the other (and vice-versa)
            return True #

        else: # Less strict numerical check (e.g., for non-FinancialMetric types)
            common_num_fields = set(num_values1.keys()) & set(num_values2.keys()) #
            # If both have numerical values but no common fields, they are considered different
            if not common_num_fields and (num_values1 and num_values2): #
                 return False 

            for field in common_num_fields:
                val1 = num_values1[field]
                val2 = num_values2[field]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Percentage tolerance
                    if 'percentage' in field.lower() or entity1.get('metricUnit') == '%' or entity2.get('metricUnit') == '%': #
                        if abs(val1 - val2) > 1.0: return False  # e.g., 1% difference
                    # Value/amount tolerance (relative difference)
                    elif ('value' in field.lower() or 'amount' in field.lower()): #
                        if max(abs(val1), abs(val2)) > 1e-9: # Avoid division by zero
                           if abs(val1 - val2) / max(abs(val1), abs(val2)) > 0.05: return False # 5% relative diff
                        elif abs(val1 - val2) > 1e-9 : return False # For small numbers, use absolute diff
                    elif abs(val1 - val2) > 1e-9: return False # Default strict for other numericals
                elif str(val1) != str(val2): return False # If not numerical, compare as string
        
    return True # If all checks pass

def find_matching_entity(entity: Dict[str, Any], entities: List[Dict[str, Any]], 
                        threshold: float = 0.75) -> Optional[Dict[str, Any]]:
    """
    Find a matching entity in a list of entities based on intrinsic similarity.
    The strictness of numerical checks depends on the entity type.
    """
    entity_type_value = entity.get('type') # Get the value, could be None
    entity_type_full = str(entity_type_value).lower() if entity_type_value is not None else "" # Handle None explicitly
    entity_type_unprefixed = entity_type_full.split(':')[-1]
    
    # Types for which numerical values should be checked strictly during similarity assessment
    metric_types_for_strict_num_check = {"financialmetric", "operationalkpi", "headcount"} #
    check_numericals_strictly_for_type = entity_type_unprefixed in metric_types_for_strict_num_check or \
                                         'metric' in entity_type_unprefixed #

    for candidate_entity in entities:
        # Pass the type-dependent strictness for numerical check to are_entities_similar
        if are_entities_similar(entity, candidate_entity, 
                                threshold=threshold, 
                                check_numerical_values_for_similarity=check_numericals_strictly_for_type,
                                check_temporal_for_similarity=True): # Always check temporal for similarity matching
            return candidate_entity #
            
    return None

def merge_entity_attributes(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge attributes from entity2 into entity1. entity1 is the existing entity, entity2 is new.
    Prioritizes completeness and sensible merging for known attribute types.
    """
    merged_entity = entity1.copy() #
    
    for key, val2 in entity2.items():
        if key == 'id': # Do not overwrite the ID of the existing entity
            continue
        
        val1 = merged_entity.get(key)
        
        # If val2 has content (not None and not empty string)
        if val2 is not None and (not isinstance(val2, str) or val2.strip()): 
            if val1 is None or (isinstance(val1, str) and not val1.strip()): # If val1 is empty or None, take val2
                merged_entity[key] = val2
            elif val1 != val2: # If both have content but are different
                # Specific merging logic for name-like keys: prefer longer or more specific names (simple heuristic)
                name_keys = ['name', 'metricName', 'kpiName', 'productName', 'fullName', 
                             'shareholderName', 'contextName', 'eventName', 'locationName'] #
                if key in name_keys:
                    if len(str(val2)) > len(str(val1)): #
                         merged_entity[key] = val2
                
                # For numerical values: if entity1's value is insubstantial (None, 0), prefer entity2's value.
                # This assumes `are_entities_similar` handled whether they *should* be merged based on value similarity.
                elif key in ['metricValue', 'percentageValue', 'headcountValue', 'amount', 'parsedValue', 'parsedPercentage']: #
                     if val1 is None or val1 == 0 or val1 == 0.0 : #
                         merged_entity[key] = val2
                     # else: val1 (existing, substantial) is kept if different.
                
                elif isinstance(val1, list) and isinstance(val2, list): # Merge lists by combining unique elements
                    temp_list = list(val1)  # Start with elements from entity1
                    for item in val2:
                        if item not in temp_list: # Add unique items from entity2
                            temp_list.append(item)
                    merged_entity[key] = temp_list #
                
                elif isinstance(val1, dict) and isinstance(val2, dict): # Shallow merge for dictionaries, val2 keys override val1
                    merged_entity[key] = {**val1, **val2} #
                else: 
                    # General default: if val1 exists and is different, keep val1 (existing).
                    # One could choose to prefer val2 (newer) here, but the original logic often kept existing.
                    # Let's add a case for preferring longer strings if both are strings.
                     if isinstance(val1, str) and isinstance(val2, str) and len(val2) > len(val1): #
                        merged_entity[key] = val2
                     # Example: if a boolean flag was True and new info says False, update it.
                     elif isinstance(val1, bool) and isinstance(val2, bool) and val1 != val2: #
                        merged_entity[key] = val2 # Prefer new boolean if different
                     # Default: keep existing if no other rule.
                     pass 

    return merged_entity

def _get_company_id_for_metric(metric_id: str,
                              relationships: List[Dict[str, Any]],
                              entities_list: List[Dict[str, Any]],
                              rel_types_comp_to_metric: Optional[List[str]] = None,
                              rel_types_metric_to_comp: Optional[List[str]] = None
                             ) -> Optional[str]:
    """
    Finds the ID of a company linked to a given metric ID within a single graph's context.
    Used for contextual similarity checks of metric entities.
    """
    # Default relationship types linking companies to metrics (adapt if ontology changes)
    if rel_types_comp_to_metric is None:
        rel_types_comp_to_metric = ["pekg:reportsMetric", "pekg:reportsHeadcount", 
                                    "pekg:reportsOperationalKPI", "reportsFinancialMetric", "reportsMetric"] #
    if rel_types_metric_to_comp is None: # Relationships from metric to company (less common)
        rel_types_metric_to_comp = [] 

    # Create a quick lookup for entity types by ID from the provided entities_list
    entity_id_to_type_map = {
        entity['id']: entity.get('type', '')
        for entity in entities_list if isinstance(entity, dict) and 'id' in entity
    } #

    for rel in relationships:
        if not isinstance(rel, dict): #
            continue

        source_id = rel.get('source') #
        target_id = rel.get('target') #
        rel_type = rel.get('type') #

        # Check Company --links--> Metric relationships
        if target_id == metric_id and rel_type in rel_types_comp_to_metric: #
            if source_id in entity_id_to_type_map and \
               entity_id_to_type_map[source_id].lower().endswith(':company'): # Make company check case-insensitive
                return source_id #

        # Check Metric --links--> Company relationships
        elif source_id == metric_id and rel_type in rel_types_metric_to_comp: #
            if target_id in entity_id_to_type_map and \
               entity_id_to_type_map[target_id].lower().endswith(':company'): # Make company check case-insensitive
                return target_id #
    return None

def normalize_entity_ids(graph: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Normalizes all entity IDs in the graph to "e1", "e2", ...,
    and updates relationships accordingly. Ensures entities are dicts and have 'id'.
    """
    if not isinstance(graph, dict): #
        print("Warning: Graph for ID normalization is not a dict. Returning as is.") #
        return graph

    entities = graph.get('entities', []) #
    relationships = graph.get('relationships', []) #
    
    id_remap: Dict[str, str] = {} 
    new_entities: List[Dict[str, Any]] = []
    
    # Filter for valid entities that are dictionaries and have an 'id' key
    valid_entities_for_remapping = [e for e in entities if isinstance(e, dict) and 'id' in e] #

    for i, entity_dict in enumerate(valid_entities_for_remapping):
        old_id = entity_dict['id'] # 'id' is confirmed to exist here
        new_id = f"e{i+1}" # Generate new sequential ID, e.g., "e1", "e2"
        
        # This mapping assumes old_ids are unique within the 'entities' list before this function.
        # If old_ids could be duplicated for different entity objects, this simple remap might be an issue,
        # but merge_knowledge_graphs should handle ID uniqueness before this stage.
        if old_id not in id_remap: # Standard case
            id_remap[old_id] = new_id #
        # else: if old_id is already in id_remap, it means it was shared by multiple entity objects.
        # The current logic (iterating with enumerate) assigns a unique new_id (e.g., e1, e2) to each entity object
        # regardless of old_id duplication. The id_remap will map the first encountered old_id.
        # Relationships referencing that old_id will map to the new_id of the first entity object.
        # This should ideally be cleaned before this stage if multiple distinct entities share an ID.

        new_entity_copy = entity_dict.copy() #
        new_entity_copy['id'] = new_id  # Assign the new sequential ID
        new_entities.append(new_entity_copy) #
    
    new_relationships: List[Dict[str, Any]] = []
    # Set of all newly assigned, valid entity IDs for quick lookup
    valid_new_entity_ids = {e['id'] for e in new_entities}  #

    for rel_dict in relationships:
        if not (isinstance(rel_dict, dict) and 'source' in rel_dict and 'target' in rel_dict): #
            continue

        old_source = rel_dict.get('source') #
        old_target = rel_dict.get('target') #
        
        # Map old source/target IDs to the new "eX" IDs
        new_source = id_remap.get(old_source) #
        new_target = id_remap.get(old_target) #
        
        # Ensure both remapped source and target IDs are valid and present in the new entity set
        if new_source and new_target and \
           new_source in valid_new_entity_ids and new_target in valid_new_entity_ids: #
            new_rel_copy = rel_dict.copy() #
            new_rel_copy['source'] = new_source #
            new_rel_copy['target'] = new_target #
            new_relationships.append(new_rel_copy) #
        # else:
            # Optionally log dropped relationships due to missing remapped source/target
            # print(f"Debug: Dropping relationship during ID normalization due to missing source/target in remap or final entities: S:{old_source}->{new_source}, T:{old_target}->{new_target}")
            
    return {
        'entities': new_entities,
        'relationships': new_relationships
    } #

def clean_knowledge_graph(graph: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Removes duplicate relationships and relationships with dangling entities (source or target not in entities list).
    Ensures entities are dicts and have 'id'.
    """
    if not isinstance(graph, dict): #
        print("Warning: Graph for cleaning is not a dict. Returning as is.") #
        return graph

    entities = graph.get('entities', []) #
    relationships = graph.get('relationships', []) #
    
    # Create a set of valid entity IDs from the provided entities list
    # Ensures that entities are dicts and have an 'id' to be considered valid
    valid_entity_ids = {entity.get('id') for entity in entities if isinstance(entity, dict) and entity.get('id')} #
    
    cleaned_relationships: List[Dict[str, Any]] = []
    # Use a set of fingerprints to track unique relationships (source_id, target_id, type)
    relationship_fingerprints: Set[Tuple[Any, Any, Any]] = set() #

    for rel in relationships:
        if not (isinstance(rel, dict) and 'source' in rel and 'target' in rel): # Basic validation of relationship structure
            continue

        source_id = rel.get('source') #
        target_id = rel.get('target') #
        rel_type = rel.get('type', "") # Use empty string if type is None for fingerprint consistency

        # Relationship is kept only if both source and target entities exist in the valid_entity_ids set
        if source_id in valid_entity_ids and target_id in valid_entity_ids: #
            fingerprint = (source_id, target_id, rel_type) # Create fingerprint
            
            if fingerprint not in relationship_fingerprints: # If relationship is unique
                relationship_fingerprints.add(fingerprint) # Add to set of seen fingerprints
                cleaned_relationships.append(rel.copy()) # Add a copy of the relationship
        # else:
            # Optionally log dropped relationships due to dangling entities
            # print(f"Debug: Dropping relationship during cleaning due to dangling S:{source_id} or T:{target_id}")

    # Return copies of entities to prevent modification of original list if it's mutable elsewhere
    return {
        'entities': [e.copy() for e in entities if isinstance(e, dict)], #
        'relationships': cleaned_relationships
    } 