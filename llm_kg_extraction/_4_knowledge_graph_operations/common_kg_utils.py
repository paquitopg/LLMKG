# Enhanced similarity functions for your PEKG ontology
# These replace the functions in common_kg_utils.py

from typing import Dict, Any, List, Optional, Tuple, Set
import re
from difflib import SequenceMatcher

def similarity_score(str1: str, str2: str) -> float:
    """Calculate string similarity using SequenceMatcher."""
    if not str1 or not str2:
        return 0.0
    
    str1 = str(str1).lower()
    str2 = str(str2).lower()
    
    sim = SequenceMatcher(None, str1, str2).ratio()
    return sim if isinstance(sim, float) else 0.0

def normalize_text(text: str) -> str:
    """Normalize text for better matching."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_entity_primary_name(entity: Dict[str, Any]) -> str:
    """Extract the primary name field based on PEKG ontology entity types."""
    entity_type = entity.get('type', '').lower().split(':')[-1]
    
    # Map entity types to their primary name fields based on your ontology
    name_field_mapping = {
        "company": "name",
        "person": "fullName", 
        "advisor": "name",
        "shareholder": "name",
        "productorservice": "name",
        "technology": "name",
        "financialmetric": "metricName",
        "operationalkpi": "kpiName", 
        "headcount": "headcountName",
        "transactioncontext": "contextName",
        "historicalevent": "eventName",
        "location": "locationName",
        "marketcontext": "segmentName",
        "usecaseorindustry": "name",
        "position": "titleName",
        "governmentbody": "name"
    }
    
    primary_field = name_field_mapping.get(entity_type, "name")
    return str(entity.get(primary_field, entity.get("name", "")))

def find_matching_entity_pekg(entity: Dict[str, Any], entities: list, 
                             threshold: float = 0.75) -> Dict[str, Any]:
    """
    Find a matching entity using PEKG-specific similarity logic.
    This is the main public function for entity matching.
    """
    for candidate_entity in entities:
        if _are_entities_similar_pekg(entity, candidate_entity, threshold):
            return candidate_entity
    return None

def _are_entities_similar_pekg(entity1: Dict[str, Any], entity2: Dict[str, Any], 
                              threshold: float = 0.8) -> bool:
    """
    Private helper function for PEKG ontology entity similarity.
    This is used internally by find_matching_entity_pekg().
    """
    # Type must match
    type1_full = entity1.get('type', '').lower()
    type2_full = entity2.get('type', '').lower()
    
    if type1_full != type2_full:
        return False
    
    entity_type = type1_full.split(':')[-1] if ':' in type1_full else type1_full
    
    # Use entity-type specific similarity functions
    if entity_type == "financialmetric":
        return _are_financial_metrics_similar(entity1, entity2)
    elif entity_type == "transactioncontext":
        return _are_transaction_contexts_similar(entity1, entity2)
    elif entity_type == "company":
        return _are_companies_similar(entity1, entity2, threshold)
    elif entity_type == "operationalkpi":
        return _are_operational_kpis_similar(entity1, entity2)
    elif entity_type == "headcount":
        return _are_headcount_similar(entity1, entity2)
    elif entity_type == "historicalevent":
        return _are_historical_events_similar(entity1, entity2)
    elif entity_type == "advisor":
        return _are_advisors_similar(entity1, entity2)
    elif entity_type == "shareholder":
        return _are_shareholders_similar(entity1, entity2)
    elif entity_type == "person":
        return _are_persons_similar(entity1, entity2)
    else:
        # For other types, use general name-based similarity
        return _are_general_entities_similar(entity1, entity2, threshold)

def _are_financial_metrics_similar(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
    """Special similarity check for FinancialMetric entities."""
    name1 = normalize_text(entity1.get("metricName", ""))
    name2 = normalize_text(entity2.get("metricName", ""))
    
    if not name1 or not name2:
        return False
    
    name_similarity = similarity_score(name1, name2)
    if name_similarity < 0.95:
        return False
    
    # DateOrPeriod must match exactly if both present
    period1 = entity1.get("DateOrPeriod")
    period2 = entity2.get("DateOrPeriod") 
    if period1 and period2:
        if normalize_text(str(period1)) != normalize_text(str(period2)):
            return False
    
    # Scope must match if both present
    scope1 = entity1.get("scope")
    scope2 = entity2.get("scope")
    if scope1 and scope2:
        if normalize_text(str(scope1)) != normalize_text(str(scope2)):
            return False
    
    return True

def _are_transaction_contexts_similar(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
    """Special similarity check for TransactionContext entities - very strict."""
    name1 = normalize_text(entity1.get("contextName", ""))
    name2 = normalize_text(entity2.get("contextName", ""))
    
    if not name1 or not name2:
        return False
    
    # Very high threshold for transaction contexts
    name_similarity = similarity_score(name1, name2)
    if name_similarity < 0.98:
        return False
    
    # Transaction type must match exactly
    type1 = entity1.get("typeSought", "")
    type2 = entity2.get("typeSought", "")
    if type1 and type2 and type1 != type2:
        return False
    
    # Status should match
    status1 = entity1.get("status", "")
    status2 = entity2.get("status", "")
    if status1 and status2:
        if normalize_text(str(status1)) != normalize_text(str(status2)):
            return False
    
    return True

def _are_companies_similar(entity1: Dict[str, Any], entity2: Dict[str, Any], threshold: float) -> bool:
    """Enhanced similarity check for Company entities considering aliases."""
    name1 = normalize_text(entity1.get("name", ""))
    name2 = normalize_text(entity2.get("name", ""))
    
    # Check primary name similarity
    if name1 and name2:
        if similarity_score(name1, name2) >= threshold:
            return True
    
    # Check aliases 
    aliases1 = entity1.get("alias", [])
    aliases2 = entity2.get("alias", [])
    
    # Ensure aliases are lists
    if not isinstance(aliases1, list):
        aliases1 = [aliases1] if aliases1 else []
    if not isinstance(aliases2, list):
        aliases2 = [aliases2] if aliases2 else []
    
    # Check if any alias matches main name or other aliases
    all_names1 = [name1] + [normalize_text(str(alias)) for alias in aliases1 if alias]
    all_names2 = [name2] + [normalize_text(str(alias)) for alias in aliases2 if alias]
    
    for n1 in all_names1:
        for n2 in all_names2:
            if n1 and n2 and similarity_score(n1, n2) >= threshold:
                return True
    
    return False

def _are_operational_kpis_similar(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
    """Similarity check for OperationalKPI entities."""
    name1 = normalize_text(entity1.get("kpiName", ""))
    name2 = normalize_text(entity2.get("kpiName", ""))
    
    if not name1 or not name2:
        return False
    
    name_similarity = similarity_score(name1, name2)
    if name_similarity < 0.90:
        return False
    
    # Period must match if both present
    period1 = entity1.get("kpiDateOrPeriod")
    period2 = entity2.get("kpiDateOrPeriod")
    if period1 and period2:
        if normalize_text(str(period1)) != normalize_text(str(period2)):
            return False
    
    return True

def _are_headcount_similar(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
    """Similarity check for Headcount entities."""
    name1 = normalize_text(entity1.get("headcountName", ""))
    name2 = normalize_text(entity2.get("headcountName", ""))
    
    if not name1 or not name2:
        return False
    
    name_similarity = similarity_score(name1, name2)
    if name_similarity < 0.90:
        return False
    
    # Date must match if both present
    date1 = entity1.get("dateOrYear")
    date2 = entity2.get("dateOrYear")
    if date1 and date2:
        if normalize_text(str(date1)) != normalize_text(str(date2)):
            return False
    
    return True

def _are_historical_events_similar(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
    """Similarity check for HistoricalEvent entities."""
    name1 = normalize_text(entity1.get("eventName", ""))
    name2 = normalize_text(entity2.get("eventName", ""))
    
    if not name1 or not name2:
        return False
    
    name_similarity = similarity_score(name1, name2)
    if name_similarity < 0.85:
        return False
    
    # Event type should match
    type1 = entity1.get("eventType", "")
    type2 = entity2.get("eventType", "")
    if type1 and type2 and type1 != type2:
        return False
    
    # Date should match if both present
    date1 = entity1.get("dateOrYear")
    date2 = entity2.get("dateOrYear")
    if date1 and date2:
        if normalize_text(str(date1)) != normalize_text(str(date2)):
            return False
    
    return True

def _are_advisors_similar(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
    """Similarity check for Advisor entities."""
    name1 = normalize_text(entity1.get("name", ""))
    name2 = normalize_text(entity2.get("name", ""))
    
    if not name1 or not name2:
        return False
    
    name_similarity = similarity_score(name1, name2)
    if name_similarity < 0.80:
        return False
    
    # Type should be similar
    type1 = entity1.get("type", "")
    type2 = entity2.get("type", "")
    if type1 and type2:
        type_similarity = similarity_score(normalize_text(type1), normalize_text(type2))
        if type_similarity < 0.70:
            return False
    
    return True

def _are_shareholders_similar(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
    """Similarity check for Shareholder entities."""
    name1 = normalize_text(entity1.get("name", ""))
    name2 = normalize_text(entity2.get("name", ""))
    
    if not name1 or not name2:
        return False
    
    name_similarity = similarity_score(name1, name2)
    if name_similarity < 0.85:
        return False
    
    # Type should match (PE, VC, Management, etc.)
    type1 = entity1.get("type", "")
    type2 = entity2.get("type", "")
    if type1 and type2 and type1 != type2:
        return False
    
    return True

def _are_persons_similar(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
    """Similarity check for Person entities."""
    name1 = normalize_text(entity1.get("fullName", ""))
    name2 = normalize_text(entity2.get("fullName", ""))
    
    if not name1 or not name2:
        return False
    
    name_similarity = similarity_score(name1, name2)
    if name_similarity >= 0.75:
        return True
    
    # Check if one might be a shorter version of the other
    name1_parts = name1.split()
    name2_parts = name2.split()
    
    if len(name1_parts) >= 2 and len(name2_parts) >= 2:
        # Check if first and last names match
        if (name1_parts[0] == name2_parts[0] and 
            name1_parts[-1] == name2_parts[-1]):
            return True
    
    return False

def _are_general_entities_similar(entity1: Dict[str, Any], entity2: Dict[str, Any], threshold: float) -> bool:
    """General similarity check for entity types not specifically handled."""
    primary_name1 = get_entity_primary_name(entity1)
    primary_name2 = get_entity_primary_name(entity2)
    
    if not primary_name1 or not primary_name2:
        return False
    
    name_similarity = similarity_score(normalize_text(primary_name1), 
                                     normalize_text(primary_name2))
    return name_similarity >= threshold

# All helper functions for entity-specific similarity checks

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