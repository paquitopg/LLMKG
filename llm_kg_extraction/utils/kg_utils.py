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
        "headcount": "dateOrYear", # Assuming headcount entities might have a dateOrYear
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
                        check_numerical_values_for_similarity: bool = False,
                        check_temporal_for_similarity: bool = True) -> bool:
    """
    Determine if two entities are similar enough to be considered the same based on their intrinsic attributes.
    For FinancialMetric, requires name similarity AND all other attributes to be identical.
    This function DOES NOT consider relational context (e.g., parent company for a metric).
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
    
    if not name1 and not name2:
        if entity_type_unprefixed == "financialmetric": return False
        if check_temporal_for_similarity or check_numerical_values_for_similarity:
             return check_additional_attributes(entity1, entity2, 
                                                check_numerical_values_for_similarity, 
                                                check_temporal_for_similarity,
                                                strict_numerical_value_check=check_numerical_values_for_similarity)
        return True 
    
    if not name1 or not name2:
        return False

    name_similarity = similarity_score(name1, name2)
    
    if entity_type_unprefixed == "financialmetric":
        if name_similarity < 0.95:
            return False

        keys1 = set(k for k in entity1.keys() if k not in ['id', 'type'])
        keys2 = set(k for k in entity2.keys() if k not in ['id', 'type'])

        if keys1 != keys2:
            return False

        for key in keys1:
            val1 = entity1.get(key)
            val2 = entity2.get(key)

            if val1 is None and val2 is None: continue
            if (val1 is None and val2 is not None) or \
               (val1 is not None and val2 is None):
                return False

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) > 1e-9: return False
            elif isinstance(val1, str) and isinstance(val2, str):
                if normalize_text(val1) != normalize_text(val2): return False
            elif isinstance(val1, bool) and isinstance(val2, bool):
                if val1 != val2: return False
            elif isinstance(val1, list) and isinstance(val2, list):
                if val1 != val2: return False
            elif type(val1) != type(val2): return False
            elif val1 != val2: return False
        return True

    metric_like_types = {"operationalkpi", "headcount"} # Add other specific metric types if needed
    if entity_type_unprefixed in metric_like_types or 'metric' in entity_type_unprefixed: # Broader catch for "metric"
        metric_name_threshold = 0.90 
        if name_similarity >= metric_name_threshold:
            time_info1 = get_entity_temporal_info(entity1)
            time_info2 = get_entity_temporal_info(entity2)
            period1 = time_info1.get('extractedPeriod')
            period2 = time_info2.get('extractedPeriod')
            
            if period1 and period2 and normalize_text(str(period1)) != normalize_text(str(period2)):
                return False
            
            if check_numerical_values_for_similarity: # True by default from find_matching_entity for these
                 return check_additional_attributes(entity1, entity2, 
                                                   check_numerical=True, 
                                                   check_temporal=False, 
                                                   strict_numerical_value_check=True)
            return True 
        else:
            return False

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
    Helper to check non-name attributes for consistency.
    """
    if check_temporal:
        time_info1 = get_entity_temporal_info(entity1)
        time_info2 = get_entity_temporal_info(entity2)
        period1 = time_info1.get('extractedPeriod')
        period2 = time_info2.get('extractedPeriod')
        if period1 and period2 and normalize_text(str(period1)) != normalize_text(str(period2)):
            return False 
    
    if check_numerical:
        num_values1 = extract_numerical_values(entity1)
        num_values2 = extract_numerical_values(entity2)
        
        if strict_numerical_value_check:
            all_num_keys = set(num_values1.keys()) | set(num_values2.keys())
            if not all_num_keys and (num_values1 or num_values2):
                return False

            for key in all_num_keys:
                val1 = num_values1.get(key)
                val2 = num_values2.get(key)

                if val1 is None and val2 is None: continue
                if (val1 is None and val2 is not None) or \
                   (val1 is not None and val2 is None):
                    return False 

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if abs(val1 - val2) > 1e-9: return False
            return True

        else: # Less strict
            common_num_fields = set(num_values1.keys()) & set(num_values2.keys())
            if not common_num_fields and (num_values1 and num_values2):
                 return False 

            for field in common_num_fields:
                val1 = num_values1[field]
                val2 = num_values2[field]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if 'percentage' in field.lower() or entity1.get('metricUnit') == '%' or entity2.get('metricUnit') == '%':
                        if abs(val1 - val2) > 1.0: return False 
                    elif ('value' in field.lower() or 'amount' in field.lower()):
                        if max(abs(val1), abs(val2)) > 1e-9:
                           if abs(val1 - val2) / max(abs(val1), abs(val2)) > 0.05: return False
                        elif abs(val1 - val2) > 1e-9 : return False 
                    elif abs(val1 - val2) > 1e-9: return False
                elif str(val1) != str(val2): return False
        
    return True

def find_matching_entity(entity: Dict, entities: List[Dict], 
                        threshold: float = 0.75) -> Optional[Dict]:
    """
    Find a matching entity in a list of entities based on intrinsic similarity.
    """
    entity_type_full = entity.get('type','').lower()
    entity_type_unprefixed = entity_type_full.split(':')[-1]
    
    metric_types_for_strict_num_check = {"financialmetric", "operationalkpi", "headcount"}  # Add all relevant KPI/Metric types
    check_numericals_strictly = entity_type_unprefixed in metric_types_for_strict_num_check or \
                                'metric' in entity_type_unprefixed # Broader "metric" catch

    for candidate_entity in entities:
        if are_entities_similar(entity, candidate_entity, 
                                threshold=threshold, 
                                check_numerical_values_for_similarity=check_numericals_strictly,
                                check_temporal_for_similarity=True):
            return candidate_entity
            
    return None

def merge_entity_attributes(entity1: Dict, entity2: Dict) -> Dict:
    """
    Merge two entities' attributes (entity1 is existing, entity2 is new).
    Prioritizes completeness.
    """
    merged_entity = entity1.copy()
    
    for key, val2 in entity2.items():
        if key == 'id':
            continue
        
        val1 = merged_entity.get(key)
        
        if val2 is not None and (not isinstance(val2, str) or val2.strip()): # val2 has content
            if val1 is None or (isinstance(val1, str) and not val1.strip()): # val1 is empty
                merged_entity[key] = val2
            elif val1 != val2:
                name_keys = ['name', 'metricName', 'kpiName', 'productName', 'fullName', 
                             'shareholderName', 'contextName', 'eventName', 'locationName']
                if key in name_keys:
                    # Prefer longer or more specific names (simple heuristic)
                    if len(str(val2)) > len(str(val1)):
                         merged_entity[key] = val2
                
                # For numerical values, if they were deemed similar enough to merge (e.g. non-FinancialMetric types)
                # the are_entities_similar would have handled the tolerance.
                # If FinancialMetrics made it here, their values should be identical.
                # For other types, this merge can be a choice. Defaulting to keeping val1 (existing) if different.
                # Consider a more sophisticated merge for specific attributes if needed.
                # For now, if val1 is substantial, keep it. Otherwise, val2 can override.
                elif key in ['metricValue', 'percentageValue', 'headcountValue', 'amount', 'parsedValue', 'parsedPercentage']:
                     if val1 is None or val1 == 0 or val1 == 0.0 : # if val1 is not substantial
                         merged_entity[key] = val2
                     # else: val1 (existing) is kept.
                
                elif isinstance(val1, list) and isinstance(val2, list):
                    # Simple list merge: combine unique elements
                    temp_list = list(val1) 
                    for item in val2:
                        if item not in temp_list:
                            temp_list.append(item)
                    merged_entity[key] = temp_list
                
                elif isinstance(val1, dict) and isinstance(val2, dict):
                    # Shallow merge for dicts, val2 overrides val1 keys
                    merged_entity[key] = {**val1, **val2}
                else: 
                    # General override policy for other differing scalars: prefer entity2's value (new info)
                    # unless specific rules are added.
                     if isinstance(val1, str) and isinstance(val2, str) and len(val2) > len(val1):
                        merged_entity[key] = val2
                     elif val1 is True and val2 is False: # Example: if a flag was true and now it's false
                        merged_entity[key] = val2
                     elif val1 is False and val2 is True: # Example: if a flag was true and now it's false
                         merged_entity[key] = val2
                     # Default: keep existing if no other rule (or prefer new: merged_entity[key] = val2)
                     # Let's be cautious and keep existing if not explicitly preferring new.
                     pass # Keep val1

    return merged_entity

# --- New Helper Function ---
def _get_company_id_for_metric(metric_id: str,
                              relationships: List[Dict],
                              entities_list: List[Dict],
                              rel_types_comp_to_metric: Optional[List[str]] = None,
                              rel_types_metric_to_comp: Optional[List[str]] = None
                             ) -> Optional[str]:
    """
    Finds the ID of a company linked to a given metric ID within a single graph's context.
    """
    if rel_types_comp_to_metric is None:
        rel_types_comp_to_metric = ["pekg:reportsMetric", "pekg:reportsHeadcount", "pekg:reportsOperationalKPI", "reportsFinancialMetric", "reportsMetric"] # e.g., "pekg:reportsMetric" if it exists
    if rel_types_metric_to_comp is None:
        rel_types_metric_to_comp = [] # e.g., "pekg:metricReportedBy" if it exists

    # Create a quick lookup for entity types by ID from the provided entities_list
    entity_id_to_type_map = {
        entity['id']: entity.get('type', '')
        for entity in entities_list if isinstance(entity, dict) and 'id' in entity
    }

    for rel in relationships:
        if not isinstance(rel, dict):
            continue

        source_id = rel.get('source')
        target_id = rel.get('target')
        rel_type = rel.get('type')

        # Check Company --links--> Metric relationships
        if target_id == metric_id and rel_type in rel_types_comp_to_metric:
            if source_id in entity_id_to_type_map and \
               entity_id_to_type_map[source_id].lower().endswith(':company'): # Make company check case-insensitive
                return source_id

        # Check Metric --links--> Company relationships
        elif source_id == metric_id and rel_type in rel_types_metric_to_comp:
            if target_id in entity_id_to_type_map and \
               entity_id_to_type_map[target_id].lower().endswith(':company'): # Make company check case-insensitive
                return target_id
    return None

# --- Graph Merging and Cleaning ---

def merge_knowledge_graphs(graph1: Dict, graph2: Dict) -> Dict:
    """
    Merge two knowledge graphs (graph1 is existing, graph2 is new).
    Combines similar entities and their relationships.
    For metric-type entities, similarity also depends on being associated with the same company.
    """
    entities1 = graph1.get('entities', [])
    entities2 = graph2.get('entities', [])
    relationships1 = graph1.get('relationships', [])
    relationships2 = graph2.get('relationships', [])
    
    # Ensures that all entities in merged_entities_map are full copies
    merged_entities_map = {entity['id']: entity.copy() for entity in entities1 if isinstance(entity, dict) and 'id' in entity}
    id_g2_to_merged_id_map = {} 
    
    for entity_g2 in entities2:
        if not isinstance(entity_g2, dict) or 'id' not in entity_g2:
            print(f"Warning: Entity in graph2 missing ID or malformed, skipping: {str(entity_g2)[:100]}")
            continue 
        g2_id = entity_g2['id']
        
        # Find potential match based on intrinsic attributes
        matching_entity_in_merged = find_matching_entity(entity_g2, list(merged_entities_map.values()))
        
        # --- START: Contextual check for metric-type entities ---
        if matching_entity_in_merged:
            entity_g2_type = entity_g2.get('type', '').lower()
            merged_entity_type = matching_entity_in_merged.get('type', '').lower() # Already lowercased in are_entities_similar

            # Define your metric types (unprefixed, lowercase)
            # Ensure these align with the output of `entity.get('type', '').lower().split(':')[-1]` if used for type checking
            metric_type_suffixes = {"pekg:financialmetric", "pekg:operationalkpi", "pekg:headcount"} # Add more as needed

            is_g2_metric = entity_g2_type in metric_type_suffixes or "metric" in entity_g2_type # Broader check
            is_merged_metric = merged_entity_type in metric_type_suffixes or "metric" in merged_entity_type


            if is_g2_metric and is_merged_metric:
                # Both are metrics and were found intrinsically similar. Now, check company association.
                g2_company_orig_id = _get_company_id_for_metric(entity_g2['id'], relationships2, entities2)
                merged_company_orig_id = _get_company_id_for_metric(matching_entity_in_merged['id'], relationships1, entities1)

                companies_are_effectively_the_same = False
                if g2_company_orig_id is None and merged_company_orig_id is None:
                    # Both metrics are "global" / not tied to a specific company.
                    companies_are_effectively_the_same = True
                elif g2_company_orig_id is not None and merged_company_orig_id is not None:
                    # Both metrics are tied to companies. Check if these companies are the same.
                    company_g2_entity = next((e for e in entities2 if isinstance(e, dict) and e.get('id') == g2_company_orig_id), None)
                    company_merged_entity = next((e for e in entities1 if isinstance(e, dict) and e.get('id') == merged_company_orig_id), None)

                    if company_g2_entity and company_merged_entity:
                        # Use are_entities_similar for companies. Adjust threshold as needed.
                        # Ensure this call doesn't lead to infinite recursion if companies also have complex checks.
                        if are_entities_similar(company_g2_entity, company_merged_entity,
                                                threshold=0.90, # High threshold for company name/type
                                                check_numerical_values_for_similarity=False, # Typically not needed for company matching
                                                check_temporal_for_similarity=False): # Typically not needed for company matching
                            companies_are_effectively_the_same = True
                
                # If one metric has a company and the other doesn't, they are not the same.
                # companies_are_effectively_the_same remains False in this case.

                if not companies_are_effectively_the_same:
                    # Metrics intrinsically similar, but their company contexts differ.
                    # => Do NOT merge. Treat entity_g2 as new.
                    # print(f"Debug: Metric {g2_id} ({entity_g2.get('name')}) not merged with {matching_entity_in_merged['id']} ({matching_entity_in_merged.get('name')}) due to different company context.")
                    matching_entity_in_merged = None # Override previous match finding
        # --- END: Contextual check ---

        if matching_entity_in_merged:
            merged_id = matching_entity_in_merged['id']
            id_g2_to_merged_id_map[g2_id] = merged_id # Map g2's original ID to the ID in the merged graph
            # Merge attributes into the existing entity in merged_entities_map
            merged_entities_map[merged_id] = merge_entity_attributes(merged_entities_map[merged_id], entity_g2)
        else:
            # Add entity_g2 as a new entity, ensuring its ID is unique in the merged_entities_map
            current_new_id = g2_id
            id_counter = 0
            # Check if the original g2_id (or its modified version) is already a key in merged_entities_map
            while current_new_id in merged_entities_map: 
                id_counter += 1
                # Create a new unique ID if collision, e.g., by appending a suffix
                # This suffix helps distinguish it if g2_id was, for example, "e1" and "e1" already exists from graph1.
                new_suffixed_id = f"{g2_id}_dup{id_counter}"
                # However, if g2_id was ALREADY something like "e5_dup1" from a previous internal merge,
                # this could create "e5_dup1_dup1". A flat counter or UUID might be more robust for temporary unique IDs if this becomes an issue.
                # For now, this sequential dup counter is kept from original.
                print(f"Warning: ID clash for new distinct entity {g2_id} (or new due to context). Remapped to {new_suffixed_id}")
                current_new_id = new_suffixed_id
            
            new_entity_g2_copy = entity_g2.copy()
            new_entity_g2_copy['id'] = current_new_id # Assign the truly unique ID
            merged_entities_map[current_new_id] = new_entity_g2_copy
            id_g2_to_merged_id_map[g2_id] = current_new_id # Map g2's original ID to its new ID in the merged graph

    final_merged_entities = list(merged_entities_map.values())
    
    merged_relationship_fingerprints: Set[Tuple[str, str, str]] = set()
    final_merged_relationships: List[Dict] = []
    all_final_entity_ids = {e['id'] for e in final_merged_entities if isinstance(e, dict) and 'id' in e}


    # Process relationships from graph1
    for rel_g1 in relationships1:
        if not (isinstance(rel_g1, dict) and 'source' in rel_g1 and 'target' in rel_g1 and 'type' in rel_g1):
            continue
        s_id, t_id, r_type = rel_g1['source'], rel_g1['target'], rel_g1['type']
        # IDs from graph1 are already the keys in merged_entities_map (or should be)
        if s_id in all_final_entity_ids and t_id in all_final_entity_ids:
            fingerprint = (s_id, t_id, r_type)
            if fingerprint not in merged_relationship_fingerprints:
                final_merged_relationships.append(rel_g1.copy()) # Add copy
                merged_relationship_fingerprints.add(fingerprint)
    
    # Process relationships from graph2, remapping source/target IDs
    for rel_g2 in relationships2:
        if not (isinstance(rel_g2, dict) and 'source' in rel_g2 and 'target' in rel_g2 and 'type' in rel_g2):
            continue
            
        original_s_g2, original_t_g2 = rel_g2['source'], rel_g2['target']
        r_type = rel_g2['type']
        
        # Get the new IDs for source and target from the map created during entity merging
        s_id_new = id_g2_to_merged_id_map.get(original_s_g2) 
        t_id_new = id_g2_to_merged_id_map.get(original_t_g2)
        
        if s_id_new and t_id_new: # Both source and target entities were successfully mapped/added
            if s_id_new in all_final_entity_ids and t_id_new in all_final_entity_ids: # Double check they are in the final list
                fingerprint = (s_id_new, t_id_new, r_type)
                if fingerprint not in merged_relationship_fingerprints:
                    new_rel = rel_g2.copy()
                    new_rel['source'] = s_id_new
                    new_rel['target'] = t_id_new
                    final_merged_relationships.append(new_rel)
                    merged_relationship_fingerprints.add(fingerprint)
            # else:
                # print(f"Debug: Dropping rel from g2 as remapped S:{s_id_new} or T:{t_id_new} not in final entities. Orig S:{original_s_g2}, T:{original_t_g2}")
        # else:
            # print(f"Debug: Dropping rel from g2 as S:{original_s_g2} or T:{original_t_g2} not found in id_g2_to_merged_id_map.")
            
    return {
        'entities': final_merged_entities,
        'relationships': final_merged_relationships
    }


def merge_multiple_knowledge_graphs(graphs: List[Dict]) -> Dict:
    if not graphs:
        return {'entities': [], 'relationships': []}
    
    # Ensure the first graph is copied to avoid modifying the input list's content if it's mutable
    result_graph = graphs[0].copy() if graphs and isinstance(graphs[0], dict) else {'entities': [], 'relationships': []}
    # Deep copy entities and relationships if they exist to prevent modification of original graph objects
    if 'entities' in result_graph:
        result_graph['entities'] = [e.copy() for e in result_graph['entities'] if isinstance(e, dict)]
    if 'relationships' in result_graph:
        result_graph['relationships'] = [r.copy() for r in result_graph['relationships'] if isinstance(r, dict)]

    for i in range(1, len(graphs)):
        current_graph_to_merge = graphs[i]
        if isinstance(current_graph_to_merge, dict):
            # Similarly, ensure entities/relationships of the graph to merge are copied if they will be modified
            # or if merge_knowledge_graphs might modify them (though it tries to work on copies).
            # For safety, pass copies if there's any doubt.
            graph_to_merge_copy = {
                'entities': [e.copy() for e in current_graph_to_merge.get('entities', []) if isinstance(e, dict)],
                'relationships': [r.copy() for r in current_graph_to_merge.get('relationships', []) if isinstance(r, dict)]
            }
            result_graph = merge_knowledge_graphs(result_graph, graph_to_merge_copy)
        else:
            print(f"Warning: Item at index {i} in graphs list is not a dictionary, skipping merge.")

    return result_graph

def normalize_entity_ids(graph: Dict) -> Dict:
    """
    Normalizes all entity IDs in the graph to "e1", "e2", ...,
    and updates relationships accordingly.
    """
    if not isinstance(graph, dict):
        print("Warning: Graph for ID normalization is not a dict.")
        return graph

    entities = graph.get('entities', [])
    relationships = graph.get('relationships', [])
    
    id_remap = {} 
    new_entities = []
    
    # Ensure entities list contains dicts
    valid_entities_for_remapping = [e for e in entities if isinstance(e, dict) and 'id' in e]

    for i, entity_dict in enumerate(valid_entities_for_remapping):
        old_id = entity_dict['id'] # 'id' is confirmed to exist here
        new_id = f"e{i+1}" # Generate new sequential ID
        
        # It's possible for multiple original entities to share an old_id if the graph wasn't perfectly unique before this.
        # This remapping will assign the new_id based on the first encounter of old_id if old_ids are duplicated.
        # However, merge_knowledge_graphs should aim to make IDs unique before this stage.
        # If old_id is already in id_remap from a *different entity* that shared the same original ID, this is problematic.
        # The current logic assumes entities passed to normalize_entity_ids have unique IDs from the merging step.
        if old_id not in id_remap: # Standard case
            id_remap[old_id] = new_id
        else:
            # This case implies two different entities in valid_entities_for_remapping had the same old_id.
            # The first one processed got the id_remap entry. The current entity will use that same new_id.
            # This would lead to multiple entities getting the same new_id, which is bad.
            # A better approach if old_ids can be duplicated in the input 'entities' list:
            # Ensure each *entity object* gets a unique new ID.
            # This requires iterating and assigning regardless of old_id uniqueness.
            # The current code structure (enumerate valid_entities_for_remapping) inherently gives a unique new_id per entity object.
            # The potential issue is if `id_remap` is used to look up an `old_id` that was shared by multiple distinct entities.
            # Let's refine this: the key to id_remap should be the unique identifier of the entity *before* this function.
            pass # The current logic is: new_id is `f"e{i+1}"` unique to this iteration/entity object. id_remap maps its original_id to this.

        new_entity_copy = entity_dict.copy()
        new_entity_copy['id'] = new_id 
        new_entities.append(new_entity_copy)
    
    new_relationships = []
    # The new_entities list now has entities with IDs "e1", "e2", etc.
    valid_new_entity_ids = {e['id'] for e in new_entities} 

    for rel_dict in relationships:
        if not (isinstance(rel_dict, dict) and 'source' in rel_dict and 'target' in rel_dict):
            continue

        old_source = rel_dict.get('source')
        old_target = rel_dict.get('target')
        
        # Map old source/target to the new "eX" IDs
        new_source = id_remap.get(old_source)
        new_target = id_remap.get(old_target)
        
        if new_source and new_target and \
           new_source in valid_new_entity_ids and new_target in valid_new_entity_ids:
            new_rel_copy = rel_dict.copy()
            new_rel_copy['source'] = new_source
            new_rel_copy['target'] = new_target
            new_relationships.append(new_rel_copy)
        # else:
            # print(f"Debug: Dropping relationship during ID normalization due to missing source/target in remap or final entities: S:{old_source}->{new_source}, T:{old_target}->{new_target}")
            
    return {
        'entities': new_entities,
        'relationships': new_relationships
    }

def clean_knowledge_graph(graph: Dict) -> Dict:
    """
    Removes duplicate relationships and relationships with dangling entities.
    """
    if not isinstance(graph, dict):
        print("Warning: Graph for cleaning is not a dict.")
        return graph

    entities = graph.get('entities', [])
    relationships = graph.get('relationships', [])
    
    # Ensure entities are dicts and have 'id'
    valid_entity_ids = {entity.get('id') for entity in entities if isinstance(entity, dict) and entity.get('id')}
    
    cleaned_relationships = []
    relationship_fingerprints: Set[Tuple[Any, Any, Any]] = set()

    for rel in relationships:
        if not (isinstance(rel, dict) and 'source' in rel and 'target' in rel):
            continue

        source_id = rel.get('source')
        target_id = rel.get('target')
        rel_type = rel.get('type', "") # Use empty string if type is None for fingerprint consistency

        if source_id in valid_entity_ids and target_id in valid_entity_ids:
            fingerprint = (source_id, target_id, rel_type) # rel_type can be None
            
            if fingerprint not in relationship_fingerprints:
                relationship_fingerprints.add(fingerprint)
                cleaned_relationships.append(rel.copy()) # Add copy
        # else:
            # print(f"Debug: Dropping relationship during cleaning due to dangling S:{source_id} or T:{target_id}")


    return {
        'entities': [e.copy() for e in entities if isinstance(e, dict)], # Return copies of entities
        'relationships': cleaned_relationships
    }