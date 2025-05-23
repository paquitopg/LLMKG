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

def extract_numerical_values(entity: Dict) -> Dict[str, Any]:
    """
    Extract all numerical values from an entity based on a predefined list of common numerical fields.
    
    Args:
        entity (Dict): Entity to extract values from
        
    Returns:
        Dict[str, Any]: Dictionary of numerical values
    """
    numerical_values = {}
    numerical_fields = [
        'metricValue', 'percentageValue', 'headcountValue', 
        'roundAmount', 'valuation', 'revenueAmount', 'amount', # From pekg:FinancialValue
        'parsedValue', 'parsedPercentage' # From streamlined pekg:FinancialMetric
    ]
    
    for field in numerical_fields:
        if field in entity and entity[field] is not None:
            try:
                val_str = str(entity[field]).replace(',', '') # Handle commas in numbers
                val = float(val_str)
                numerical_values[field] = val
            except (ValueError, TypeError):
                pass 
    return numerical_values

def get_entity_temporal_info(entity: Dict) -> Dict[str, Any]:
    """
    Extract temporal information (years, dates, periods) from an entity.
    Prioritizes specific temporal fields from the ontology, then tries to parse from name fields.
    
    Args:
        entity (Dict): Entity to extract temporal info from
        
    Returns:
        Dict[str, Any]: Dictionary of temporal information, with 'extractedPeriod' as the primary consolidated field.
    """
    temporal_info = {}
    
    # Specific ontology fields for temporal data (unprefixed for easier access)
    # From pekg_ontology_streamlined_v1_2
    ontology_time_fields = {
        "financialmetric": "fiscalPeriod",
        "operationalkpi": "kpiDateOrPeriod",
        "headcount": "dateOrYear",
        "historicalevent": "dateOrYear",
        "company": "foundedYear" 
        # Add other types and their specific date/year fields if needed
    }
    entity_type_unprefixed = entity.get('type', '').lower().split(':')[-1]
    specific_period_field = ontology_time_fields.get(entity_type_unprefixed)

    if specific_period_field and specific_period_field in entity and entity[specific_period_field] is not None:
        temporal_info['extractedPeriod'] = str(entity[specific_period_field])
    
    # Fallback: Try to extract year/period from name if no specific field was found or populated
    if 'extractedPeriod' not in temporal_info:
        name_to_search = entity.get('name', entity.get('metricName', entity.get('kpiName', '')))
        if name_to_search and isinstance(name_to_search, str):
            year_patterns = [
                r'\b(FY|CY|CAL\s*|H[1-2]|Q[1-4])?\s*(20\d{2}|19\d{2})\s*([A-Za-zEBPFL]*)?\b', # FY21A, 2022, Q1 2023, H1-22
                r'\b(20\d{2}|19\d{2})\s*-\s*(20\d{2}|19\d{2})\b([A-Za-z]*)', # 2020-2022 CAGR
            ]
            for pattern in year_patterns:
                match = re.search(pattern, name_to_search, re.IGNORECASE)
                if match:
                    temporal_info['extractedPeriod'] = match.group(0).strip() # Take full match as period
                    break
    return temporal_info


def are_entities_similar(entity1: Dict, entity2: Dict, 
                        threshold: float = 0.8, 
                        check_numerical_values_for_similarity: bool = False, # This flag is now less relevant for FinancialMetric
                        check_temporal_for_similarity: bool = True) -> bool:
    """
    Determine if two entities are similar enough to be considered the same.
    For FinancialMetric, requires name similarity AND all other attributes to be identical.
    """
    type1_full = entity1.get('type', '').lower()
    type2_full = entity2.get('type', '').lower()
    type1_unprefixed = type1_full.split(':')[-1]
    type2_unprefixed = type2_full.split(':')[-1]

    if type1_unprefixed != type2_unprefixed:
        return False
    
    entity_type_unprefixed = type1_unprefixed
    
    name1_str = entity1.get('metricName', entity1.get('name', entity1.get('kpiName', entity1.get('productName', entity1.get('fullName', '')))))
    name2_str = entity2.get('metricName', entity2.get('name', entity2.get('kpiName', entity2.get('productName', entity2.get('fullName', '')))))
    
    name1 = normalize_text(name1_str)
    name2 = normalize_text(name2_str)
    
    if not name1 and not name2: # Both lack names
        # If types are FinancialMetric, they must have names to be compared this way.
        if entity_type_unprefixed == "financialmetric": return False
        # For other types, if no names, rely on other attribute checks if enabled
        if check_temporal_for_similarity or check_numerical_values_for_similarity:
             return check_additional_attributes(entity1, entity2, 
                                                check_numerical_values_for_similarity, 
                                                check_temporal_for_similarity,
                                                strict_numerical_value_check=check_numerical_values_for_similarity)
        return True # Or False, depending on policy for nameless entities of other types
    
    if not name1 or not name2: # One has name, other doesn't
        return False

    name_similarity = similarity_score(name1, name2)
    
    # Specific strict logic for FinancialMetric as per user request
    if entity_type_unprefixed == "financialmetric":
        if name_similarity < 0.95: # Step 1: High name similarity required
            return False

        # Step 2: Compare all other attributes for exact match
        keys1 = set(k for k in entity1.keys() if k not in ['id', 'type'])
        keys2 = set(k for k in entity2.keys() if k not in ['id', 'type'])

        if keys1 != keys2: # Must have the exact same set of attribute keys
            # print(f"DEBUG: FinancialMetric keys differ. e1: {keys1}, e2: {keys2}")
            return False

        for key in keys1: # Iterate through one set, as they are identical
            val1 = entity1.get(key)
            val2 = entity2.get(key)

            # Handle None values consistently
            if val1 is None and val2 is None:
                continue
            if (val1 is None and val2 is not None) or \
               (val1 is not None and val2 is None):
                # print(f"DEBUG: FinancialMetric attribute '{key}' None mismatch: '{val1}' vs '{val2}'")
                return False

            # Type-aware comparison
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) > 1e-9: # Tolerance for float comparison
                    # print(f"DEBUG: FinancialMetric float mismatch for '{key}': {val1} vs {val2}")
                    return False
            elif isinstance(val1, str) and isinstance(val2, str):
                # For valueString, direct comparison after normalization. For others, could be more lenient if needed.
                if normalize_text(val1) != normalize_text(val2):
                    # print(f"DEBUG: FinancialMetric string mismatch for '{key}': '{normalize_text(val1)}' vs '{normalize_text(val2)}'")
                    return False
            elif isinstance(val1, bool) and isinstance(val2, bool):
                if val1 != val2:
                    # print(f"DEBUG: FinancialMetric bool mismatch for '{key}': {val1} vs {val2}")
                    return False
            elif isinstance(val1, list) and isinstance(val2, list):
                # Simple list comparison: requires same elements in same order.
                # For financial metrics, lists are not common in the streamlined ontology.
                if val1 != val2: 
                    # print(f"DEBUG: FinancialMetric list mismatch for '{key}': {val1} vs {val2}")
                    return False
            elif type(val1) != type(val2): # Mismatched types (and not caught by None checks)
                # print(f"DEBUG: FinancialMetric type mismatch for '{key}': {type(val1)} vs {type(val2)}")
                return False
            elif val1 != val2: # Fallback for other types or if above checks didn't catch subtle differences
                # print(f"DEBUG: FinancialMetric general value mismatch for '{key}': {val1} vs {val2}")
                return False
        return True # All attributes matched for FinancialMetric

    # Existing logic for other metric/KPI types (less strict than FinancialMetric)
    elif 'metric' in entity_type_unprefixed or 'kpi' in entity_type_unprefixed or 'headcount' in entity_type_unprefixed:
        metric_name_threshold = 0.90 
        if name_similarity >= metric_name_threshold:
            # Check temporal info (e.g., fiscalPeriod, kpiDateOrPeriod, dateOrYear)
            time_info1 = get_entity_temporal_info(entity1) # Uses 'extractedPeriod'
            time_info2 = get_entity_temporal_info(entity2)
            period1 = time_info1.get('extractedPeriod')
            period2 = time_info2.get('extractedPeriod')
            
            if period1 and period2 and normalize_text(str(period1)) != normalize_text(str(period2)):
                return False
            
            # For other metrics (not FinancialMetric), we might still use check_additional_attributes
            # if `check_numerical_values_for_similarity` is true (which it is by default from find_matching_entity)
            if check_numerical_values_for_similarity:
                 return check_additional_attributes(entity1, entity2, 
                                                   check_numerical=True, 
                                                   check_temporal=False, # Period already checked
                                                   strict_numerical_value_check=True) # Values must be close
            return True 
        else:
            return False

    # General case for other entity types
    if name_similarity >= threshold:
        if check_temporal_for_similarity or check_numerical_values_for_similarity:
            return check_additional_attributes(entity1, entity2, 
                                               check_numerical_values_for_similarity, 
                                               check_temporal_for_similarity,
                                               strict_numerical_value_check=check_numerical_values_for_similarity)
        return True
            
    return False

def check_additional_attributes(entity1: Dict, entity2: Dict, 
                               check_numerical: bool,
                               check_temporal: bool,
                               strict_numerical_value_check: bool) -> bool:
    """
    Helper to check non-name attributes for consistency, primarily for non-FinancialMetric types
    or when a less strict check is needed.
    """
    if check_temporal:
        time_info1 = get_entity_temporal_info(entity1)
        time_info2 = get_entity_temporal_info(entity2)
        period1 = time_info1.get('extractedPeriod') # Relies on get_entity_temporal_info's consolidation
        period2 = time_info2.get('extractedPeriod')
        if period1 and period2 and normalize_text(str(period1)) != normalize_text(str(period2)):
            return False 
    
    if check_numerical:
        num_values1 = extract_numerical_values(entity1)
        num_values2 = extract_numerical_values(entity2)
        
        # If strict check, all numerical fields present in one must be in other and match.
        if strict_numerical_value_check:
            all_num_keys = set(num_values1.keys()) | set(num_values2.keys())
            if not all_num_keys and (num_values1 or num_values2): # One has numbers, other doesn't, but strict check
                return False

            for key in all_num_keys:
                val1 = num_values1.get(key)
                val2 = num_values2.get(key)

                if val1 is None and val2 is None: continue
                if (val1 is None and val2 is not None) or \
                   (val1 is not None and val2 is None):
                    return False # One has it, other doesn't, considered different under strict

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if abs(val1 - val2) > 1e-9: # Using a small tolerance for floats
                        return False
                # This function primarily focuses on numericals, string comparison of numericals is less robust
                # elif str(val1) != str(val2): 
                # return False 
            return True # All common or present numericals matched strictly

        # Less strict: only compare common numerical fields if not strict_numerical_value_check
        else:
            common_num_fields = set(num_values1.keys()) & set(num_values2.keys())
            if not common_num_fields and (num_values1 and num_values2): # Both have numericals, but no common fields
                 return False # Or True, depending on policy. False is safer for "similar".

            for field in common_num_fields:
                val1 = num_values1[field]
                val2 = num_values2[field]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if 'percentage' in field.lower() or entity1.get('metricUnit') == '%' or entity2.get('metricUnit') == '%':
                        if abs(val1 - val2) > 1.0: return False # Allow 1% diff
                    elif ('value' in field.lower() or 'amount' in field.lower()):
                        if max(abs(val1), abs(val2)) > 1e-9:
                           if abs(val1 - val2) / max(abs(val1), abs(val2)) > 0.05: return False # 5% tolerance
                        elif abs(val1 - val2) > 1e-9 : return False # Both near zero but different
                    elif abs(val1 - val2) > 1e-9: return False
                elif str(val1) != str(val2): return False
        
    return True

def find_matching_entity(entity: Dict, entities: List[Dict], 
                        threshold: float = 0.75) -> Optional[Dict]:
    """
    Find a matching entity in a list of entities.
    The check_numerical_values_for_similarity flag passed to are_entities_similar
    will be True for metrics/KPIs, triggering stricter checks within are_entities_similar.
    """
    entity_type_full = entity.get('type','').lower()
    entity_type_unprefixed = entity_type_full.split(':')[-1]
    
    # For FinancialMetric, are_entities_similar now has its own very strict "all attributes must match" logic.
    # For other metrics/KPIs, we still want strict numerical value comparison.
    check_numericals_strictly = 'metric' in entity_type_unprefixed or \
                                'kpi' in entity_type_unprefixed or \
                                'headcount' in entity_type_unprefixed
        
    for candidate_entity in entities:
        if are_entities_similar(entity, candidate_entity, 
                                threshold=threshold, 
                                check_numerical_values_for_similarity=check_numericals_strictly,
                                check_temporal_for_similarity=True): # Always check temporal for relevant types
            return candidate_entity
            
    return None

def merge_entity_attributes(entity1: Dict, entity2: Dict) -> Dict:
    """
    Merge two entities' attributes (entity1 is existing, entity2 is new).
    Prioritizes completeness. For conflicting numerical values where entities were deemed similar
    (e.g., for non-FinancialMetric types), entity1's value is kept.
    """
    merged_entity = entity1.copy()
    
    for key, val2 in entity2.items():
        if key == 'id':
            continue
        
        val1 = merged_entity.get(key)
        
        if val2 is not None and (not isinstance(val2, str) or val2.strip()):
            if val1 is None or (isinstance(val1, str) and not val1.strip()):
                merged_entity[key] = val2
            elif val1 != val2:
                # Name merging: prefer longer or more specific
                name_keys = ['name', 'metricName', 'kpiName', 'productName', 'fullName', 'shareholderName', 'contextName', 'eventName', 'locationName']
                if key in name_keys:
                    name1_norm = normalize_text(str(val1))
                    name2_norm = normalize_text(str(val2))
                    if name1_norm in name2_norm and len(name2_norm) > len(name1_norm): merged_entity[key] = val2
                    elif name2_norm in name1_norm and len(name1_norm) > len(name2_norm): pass 
                    elif re.search(r'(fy\d{2,4}|20\d{2})', name2_norm, re.IGNORECASE) and \
                         not re.search(r'(fy\d{2,4}|20\d{2})', name1_norm, re.IGNORECASE): merged_entity[key] = val2
                    elif len(name2_norm) > len(name1_norm): merged_entity[key] = val2
                
                # For FinancialMetric, this branch should ideally not be hit for differing core attributes
                # because are_entities_similar would have returned False.
                # This merge logic is more for other entity types or for non-critical/descriptive attributes.
                elif key in ['metricValue', 'percentageValue', 'headcountValue', 'amount', 'parsedValue', 'parsedPercentage', 'valueString', 'kpiValueString']:
                    # If entity1 has a substantive value, keep it. Otherwise, take entity2's.
                    # This is a general policy; for FinancialMetrics, they wouldn't have merged if these differed significantly.
                    if val1 is None or val1 == 0 or val1 == 0.0 or (isinstance(val1, str) and not val1.strip()):
                        merged_entity[key] = val2
                    # else, val1 is kept.
                
                elif isinstance(val1, list) and isinstance(val2, list):
                    temp_list = list(val1) 
                    for item in val2:
                        if item not in temp_list:
                            temp_list.append(item)
                    merged_entity[key] = temp_list
                
                elif isinstance(val1, dict) and isinstance(val2, dict):
                    merged_entity[key] = {**val1, **val2}
                else: 
                    # For other differing scalar types, generally prefer val2 if val1 is generic or val2 is more specific
                    # Or if val1 is just a boolean placeholder and val2 has more info.
                    # If val1 is a meaningful string and val2 is different, might prefer longer or more descriptive.
                    if isinstance(val1, str) and isinstance(val2, str) and len(val2) > len(val1):
                        merged_entity[key] = val2
                    elif val1 is True and val2 is False: # Example: if a flag was true and now it's false
                        merged_entity[key] = val2
                    elif val1 is False and val2 is True:
                         merged_entity[key] = val2
                    else: # Default to entity2's value if no other rule applies for conflicting scalars
                        merged_entity[key] = val2
    
    return merged_entity

# --- Graph Merging and Cleaning (largely unchanged, but reviewed for compatibility) ---

def merge_knowledge_graphs(graph1: Dict, graph2: Dict) -> Dict:
    """
    Merge two knowledge graphs (graph1 is existing, graph2 is new).
    Combines similar entities and their relationships.
    """
    entities1 = graph1.get('entities', [])
    entities2 = graph2.get('entities', [])
    relationships1 = graph1.get('relationships', [])
    relationships2 = graph2.get('relationships', [])
    
    merged_entities_map = {entity['id']: entity.copy() for entity in entities1 if 'id' in entity}
    id_g2_to_merged_id_map = {} 
    
    for entity_g2 in entities2:
        if 'id' not in entity_g2:
            print(f"Warning: Entity in graph2 missing ID, skipping: {entity_g2.get('name', 'Unnamed')}")
            continue 
        g2_id = entity_g2['id']
        
        matching_entity_in_merged = find_matching_entity(entity_g2, list(merged_entities_map.values()))
        
        if matching_entity_in_merged:
            merged_id = matching_entity_in_merged['id']
            id_g2_to_merged_id_map[g2_id] = merged_id
            merged_entities_map[merged_id] = merge_entity_attributes(merged_entities_map[merged_id], entity_g2)
        else:
            current_new_id = g2_id
            id_counter = 0
            while current_new_id in merged_entities_map: # Handle potential ID collision for distinct entities
                id_counter += 1
                current_new_id = f"{g2_id}_dup{id_counter}"
                print(f"Warning: ID clash for new distinct entity {g2_id}. Remapped to {current_new_id}")
            
            new_entity_g2_copy = entity_g2.copy()
            new_entity_g2_copy['id'] = current_new_id # Assign new ID if it was remapped
            merged_entities_map[current_new_id] = new_entity_g2_copy
            id_g2_to_merged_id_map[g2_id] = current_new_id

    final_merged_entities = list(merged_entities_map.values())
    
    merged_relationship_fingerprints = set()
    final_merged_relationships = []

    all_final_entity_ids = {e['id'] for e in final_merged_entities} # For validation

    for rel_g1 in relationships1:
        if 'source' in rel_g1 and 'target' in rel_g1 and 'type' in rel_g1:
            s_id, t_id, r_type = rel_g1['source'], rel_g1['target'], rel_g1['type']
            # Ensure source and target from g1 still exist (they should, as merged_entities_map started with them)
            if s_id in all_final_entity_ids and t_id in all_final_entity_ids:
                fingerprint = (s_id, t_id, r_type)
                if fingerprint not in merged_relationship_fingerprints:
                    final_merged_relationships.append(rel_g1.copy())
                    merged_relationship_fingerprints.add(fingerprint)
    
    for rel_g2 in relationships2:
        if 'source' in rel_g2 and 'target' in rel_g2 and 'type' in rel_g2:
            original_s_g2, original_t_g2 = rel_g2['source'], rel_g2['target']
            
            s_id_new = id_g2_to_merged_id_map.get(original_s_g2) 
            t_id_new = id_g2_to_merged_id_map.get(original_t_g2) 
            
            # If original IDs were not found in the map (e.g. entity_g2 was added as new with a potentially remapped ID)
            # this means the relationship might be orphaned if its source/target entity from g2 was problematic.
            # However, id_g2_to_merged_id_map should contain all original g2_ids that had corresponding entities.
            if not s_id_new: s_id_new = original_s_g2 # Fallback, though less ideal
            if not t_id_new: t_id_new = original_t_g2 # Fallback

            r_type = rel_g2['type']
            
            fingerprint = (s_id_new, t_id_new, r_type)
            if fingerprint not in merged_relationship_fingerprints:
                if s_id_new in all_final_entity_ids and t_id_new in all_final_entity_ids:
                    new_rel = rel_g2.copy()
                    new_rel['source'] = s_id_new
                    new_rel['target'] = t_id_new
                    final_merged_relationships.append(new_rel)
                    merged_relationship_fingerprints.add(fingerprint)
                # else:
                    # print(f"Debug: Dropping rel from g2: S:{original_s_g2}->{s_id_new}, T:{original_t_g2}->{t_id_new} not in final entities.")
    
    return {
        'entities': final_merged_entities,
        'relationships': final_merged_relationships
    }


def merge_multiple_knowledge_graphs(graphs: List[Dict]) -> Dict:
    if not graphs:
        return {'entities': [], 'relationships': []}
    
    result_graph = graphs[0].copy() if graphs else {'entities': [], 'relationships': []}
    
    for i in range(1, len(graphs)):
        result_graph = merge_knowledge_graphs(result_graph, graphs[i])
    
    return result_graph

def normalize_entity_ids(graph: Dict) -> Dict:
    entities = graph.get('entities', [])
    relationships = graph.get('relationships', [])
    
    id_remap = {} 
    new_entities = []
    
    for i, entity_dict in enumerate(entities):
        if not isinstance(entity_dict, dict):
            continue

        old_id = entity_dict.get('id', f"missing_id_{i}") 
        new_id = f"e{i+1}" 
        id_remap[old_id] = new_id

        new_entity_copy = entity_dict.copy()
        new_entity_copy['id'] = new_id 
        new_entities.append(new_entity_copy)
    
    new_relationships = []
    valid_new_entity_ids = {e['id'] for e in new_entities} 

    for rel_dict in relationships:
        if not isinstance(rel_dict, dict):
            continue

        old_source = rel_dict.get('source', '')
        old_target = rel_dict.get('target', '')
        
        new_source = id_remap.get(old_source)
        new_target = id_remap.get(old_target)
        
        if new_source and new_target and new_source in valid_new_entity_ids and new_target in valid_new_entity_ids:
            new_rel_copy = rel_dict.copy()
            new_rel_copy['source'] = new_source
            new_rel_copy['target'] = new_target
            new_relationships.append(new_rel_copy)
            
    return {
        'entities': new_entities,
        'relationships': new_relationships
    }

def clean_knowledge_graph(graph: Dict) -> Dict:
    entities = graph.get('entities', [])
    relationships = graph.get('relationships', [])
    
    valid_entity_ids = {entity.get('id') for entity in entities if isinstance(entity, dict) and entity.get('id')}
    
    cleaned_relationships = []
    relationship_fingerprints = set() 

    for rel in relationships:
        if not isinstance(rel, dict): 
            continue

        source_id = rel.get('source')
        target_id = rel.get('target')
        rel_type = rel.get('type') 

        if source_id in valid_entity_ids and target_id in valid_entity_ids:
            fingerprint_type = rel_type if rel_type is not None else ""
            fingerprint = (source_id, target_id, fingerprint_type)
            
            if fingerprint not in relationship_fingerprints:
                relationship_fingerprints.add(fingerprint)
                cleaned_relationships.append(rel.copy()) 

    return {
        'entities': entities, 
        'relationships': cleaned_relationships
    }