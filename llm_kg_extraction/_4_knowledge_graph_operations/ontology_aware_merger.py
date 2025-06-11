# Enhanced PageLevelMerger with ontology-aware strategies

from typing import List, Dict, Any, Set, Tuple, Optional
from .common_kg_utils import (
    find_matching_entity,
    merge_entity_attributes,
    are_entities_similar,
    normalize_text,
    similarity_score
)

class PageLevelMerger:
    """
    Enhanced PageLevelMerger that uses ontology-specific merging strategies
    to preserve important context entities while appropriately merging reference entities.
    """

    def __init__(self, ontology, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self.ontology = ontology
        self.merge_strategies = self._define_ontology_strategies()

    def _define_ontology_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Define merge strategies based on your PEKG ontology.
        """
        return {
            # Context entities - should rarely be merged
            "transactioncontext": {
                "strategy": "preserve_distinct",
                "threshold": 0.95,  # Very high threshold
                "key_differentiators": ["contextName", "typeSought", "status"],
                "description": "Each transaction context represents a distinct deal scenario"
            },
            
            # Financial metrics - need strict matching
            "financialmetric": {
                "strategy": "strict_temporal_match",
                "threshold": 0.95,
                "key_differentiators": ["metricName", "DateOrPeriod", "scope"],
                "description": "Financial metrics must match exactly on name, period, and scope"
            },
            
            "operationalkpi": {
                "strategy": "strict_temporal_match", 
                "threshold": 0.90,
                "key_differentiators": ["kpiName", "kpiDateOrPeriod"],
                "description": "KPIs must match on name and time period"
            },
            
            "headcount": {
                "strategy": "strict_temporal_match",
                "threshold": 0.90,
                "key_differentiators": ["headcountName", "dateOrYear"],
                "description": "Headcount must match on description and time"
            },
            
            # Reference entities - can be merged more liberally
            "company": {
                "strategy": "liberal_merge",
                "threshold": 0.70,
                "key_differentiators": ["name", "alias"],
                "description": "Companies mentioned across pages are likely the same"
            },
            
            "person": {
                "strategy": "liberal_merge",
                "threshold": 0.75,
                "key_differentiators": ["fullName"],
                "description": "People mentioned across pages are likely the same"
            },
            
            "advisor": {
                "strategy": "moderate_merge",
                "threshold": 0.80,
                "key_differentiators": ["name", "type", "roleInContext"],
                "description": "Advisors may have different roles in different contexts"
            },
            
            "shareholder": {
                "strategy": "moderate_merge",
                "threshold": 0.85,
                "key_differentiators": ["name", "type"],
                "description": "Shareholders should be merged carefully due to ownership changes"
            },
            
            # Products and services
            "productorservice": {
                "strategy": "moderate_merge",
                "threshold": 0.80,
                "key_differentiators": ["name", "category"],
                "description": "Products/services need moderate similarity"
            },
            
            "technology": {
                "strategy": "liberal_merge",
                "threshold": 0.75,
                "key_differentiators": ["name"],
                "description": "Technologies mentioned across pages are likely the same"
            },
            
            # Events and temporal entities
            "historicalevent": {
                "strategy": "preserve_with_temporal_context",
                "threshold": 0.90,
                "key_differentiators": ["eventName", "eventType", "dateOrYear"],
                "description": "Events may be mentioned multiple times with different details"
            },
            
            # Market and location context
            "marketcontext": {
                "strategy": "moderate_merge",
                "threshold": 0.80,
                "key_differentiators": ["segmentName", "geography"],
                "description": "Market contexts should be merged if segment and geography match"
            },
            
            "location": {
                "strategy": "liberal_merge",
                "threshold": 0.75,
                "key_differentiators": ["locationName", "locationType"],
                "description": "Locations mentioned across pages are likely the same"
            },
            
            "usecaseorindustry": {
                "strategy": "liberal_merge",
                "threshold": 0.75,
                "key_differentiators": ["name"],
                "description": "Use cases/industries are likely the same across mentions"
            },
            
            "position": {
                "strategy": "strict_merge",
                "threshold": 0.90,
                "key_differentiators": ["titleName", "department"],
                "description": "Positions should match exactly on title and department"
            },
            
            "governmentbody": {
                "strategy": "liberal_merge",
                "threshold": 0.80,
                "key_differentiators": ["name", "type"],
                "description": "Government bodies mentioned across pages are likely the same"
            }
        }

    def _get_entity_type_unprefixed(self, entity: Dict[str, Any]) -> str:
        """Extract the unprefixed entity type."""
        entity_type = entity.get('type', '')
        if ':' in entity_type:
            return entity_type.split(':')[-1].lower()
        return entity_type.lower()

    def _should_merge_entities_ontology_aware(self, entity1: Dict[str, Any], 
                                            entity2: Dict[str, Any]) -> bool:
        """
        Determine if two entities should be merged based on ontology-specific strategies.
        """
        entity_type = self._get_entity_type_unprefixed(entity1)
        
        # Get strategy for this entity type
        strategy_config = self.merge_strategies.get(entity_type, {
            "strategy": "default_merge",
            "threshold": self.similarity_threshold,
            "key_differentiators": ["name"],
            "description": "Default strategy for unknown entity types"
        })
        
        strategy = strategy_config["strategy"]
        threshold = strategy_config["threshold"]
        key_differentiators = strategy_config["key_differentiators"]
        
        # Check if types match
        if self._get_entity_type_unprefixed(entity1) != self._get_entity_type_unprefixed(entity2):
            return False
        
        if strategy == "preserve_distinct":
            return self._should_merge_preserve_distinct(entity1, entity2, key_differentiators, threshold)
        
        elif strategy == "strict_temporal_match":
            return self._should_merge_strict_temporal(entity1, entity2, key_differentiators, threshold)
        
        elif strategy == "liberal_merge":
            return self._should_merge_liberal(entity1, entity2, key_differentiators, threshold)
        
        elif strategy == "moderate_merge":
            return self._should_merge_moderate(entity1, entity2, key_differentiators, threshold)
        
        elif strategy == "strict_merge":
            return self._should_merge_strict(entity1, entity2, key_differentiators, threshold)
        
        elif strategy == "preserve_with_temporal_context":
            return self._should_merge_temporal_context(entity1, entity2, key_differentiators, threshold)
        
        else:  # default_merge
            return are_entities_similar(entity1, entity2, threshold=threshold)

    def _should_merge_preserve_distinct(self, entity1: Dict[str, Any], entity2: Dict[str, Any], 
                                      key_differentiators: List[str], threshold: float) -> bool:
        """
        Strategy for entities that should rarely be merged (like TransactionContext).
        Only merge if they are nearly identical across all key differentiators.
        """
        # Check if ALL key differentiators are identical
        for diff_field in key_differentiators:
            val1 = entity1.get(diff_field)
            val2 = entity2.get(diff_field)
            
            # If both have values, they must be identical
            if val1 and val2:
                if normalize_text(str(val1)) != normalize_text(str(val2)):
                    return False
            # If one has a value and the other doesn't, don't merge
            elif val1 or val2:
                return False
        
        # If all key differentiators match, check overall similarity
        return are_entities_similar(entity1, entity2, threshold=threshold,
                                  check_numerical_values_for_similarity=True,
                                  check_temporal_for_similarity=True)

    def _should_merge_strict_temporal(self, entity1: Dict[str, Any], entity2: Dict[str, Any],
                                    key_differentiators: List[str], threshold: float) -> bool:
        """
        Strategy for metrics that need strict temporal matching.
        """
        # First check name similarity
        name_fields = ["metricName", "kpiName", "headcountName", "name"]
        name1 = name2 = ""
        
        for field in name_fields:
            if entity1.get(field):
                name1 = normalize_text(str(entity1[field]))
                break
        
        for field in name_fields:
            if entity2.get(field):
                name2 = normalize_text(str(entity2[field]))
                break
        
        if not name1 or not name2:
            return False
        
        name_similarity = similarity_score(name1, name2)
        if name_similarity < threshold:
            return False
        
        # Check temporal fields must match exactly
        temporal_fields = ["DateOrPeriod", "kpiDateOrPeriod", "dateOrYear"]
        for field in temporal_fields:
            val1 = entity1.get(field)
            val2 = entity2.get(field)
            
            if val1 and val2:
                if normalize_text(str(val1)) != normalize_text(str(val2)):
                    return False
        
        # Check scope if present
        scope1 = entity1.get("scope")
        scope2 = entity2.get("scope")
        if scope1 and scope2:
            if normalize_text(str(scope1)) != normalize_text(str(scope2)):
                return False
        
        return True

    def _should_merge_liberal(self, entity1: Dict[str, Any], entity2: Dict[str, Any],
                            key_differentiators: List[str], threshold: float) -> bool:
        """
        Strategy for entities that should be merged liberally (like Company, Person).
        """
        # Check name similarity for the primary identifier
        primary_field = key_differentiators[0] if key_differentiators else "name"
        
        name1 = normalize_text(str(entity1.get(primary_field, "")))
        name2 = normalize_text(str(entity2.get(primary_field, "")))
        
        if not name1 or not name2:
            return False
        
        name_similarity = similarity_score(name1, name2)
        if name_similarity >= threshold:
            return True
        
        # For companies, also check aliases
        if "alias" in entity1 or "alias" in entity2:
            aliases1 = entity1.get("alias", [])
            aliases2 = entity2.get("alias", [])
            
            if isinstance(aliases1, list) and isinstance(aliases2, list):
                for alias1 in aliases1:
                    for alias2 in aliases2:
                        if similarity_score(normalize_text(str(alias1)), 
                                          normalize_text(str(alias2))) >= threshold:
                            return True
            
            # Check if main name matches any alias
            for alias1 in aliases1 if isinstance(aliases1, list) else []:
                if similarity_score(name2, normalize_text(str(alias1))) >= threshold:
                    return True
            
            for alias2 in aliases2 if isinstance(aliases2, list) else []:
                if similarity_score(name1, normalize_text(str(alias2))) >= threshold:
                    return True
        
        return False

    def _should_merge_moderate(self, entity1: Dict[str, Any], entity2: Dict[str, Any],
                             key_differentiators: List[str], threshold: float) -> bool:
        """
        Strategy for entities needing moderate similarity checks.
        """
        return are_entities_similar(entity1, entity2, threshold=threshold,
                                  check_numerical_values_for_similarity=False,
                                  check_temporal_for_similarity=True)

    def _should_merge_strict(self, entity1: Dict[str, Any], entity2: Dict[str, Any],
                           key_differentiators: List[str], threshold: float) -> bool:
        """
        Strategy for entities requiring strict matching.
        """
        return are_entities_similar(entity1, entity2, threshold=threshold,
                                  check_numerical_values_for_similarity=True,
                                  check_temporal_for_similarity=True)

    def _should_merge_temporal_context(self, entity1: Dict[str, Any], entity2: Dict[str, Any],
                                     key_differentiators: List[str], threshold: float) -> bool:
        """
        Strategy for historical events that may be mentioned multiple times.
        """
        # Check event name and type similarity
        name1 = normalize_text(str(entity1.get("eventName", "")))
        name2 = normalize_text(str(entity2.get("eventName", "")))
        
        if not name1 or not name2:
            return False
        
        name_similarity = similarity_score(name1, name2)
        if name_similarity < threshold:
            return False
        
        # Event type should match
        type1 = entity1.get("eventType", "")
        type2 = entity2.get("eventType", "")
        if type1 and type2 and type1 != type2:
            return False
        
        # Date should match if both provided
        date1 = entity1.get("dateOrYear")
        date2 = entity2.get("dateOrYear")
        if date1 and date2:
            if normalize_text(str(date1)) != normalize_text(str(date2)):
                return False
        
        return True

    def merge_incrementally(self, current_document_kg: Dict[str, List[Dict[str, Any]]],
                          new_page_kg: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Enhanced incremental merging using ontology-aware strategies.
        """
        if not new_page_kg or (not new_page_kg.get("entities") and not new_page_kg.get("relationships")):
            return current_document_kg

        if not isinstance(current_document_kg, dict):
            current_document_kg = {"entities": [], "relationships": []}

        entities_main = current_document_kg.get('entities', [])
        relationships_main = current_document_kg.get('relationships', [])
        entities_page = new_page_kg.get('entities', [])
        relationships_page = new_page_kg.get('relationships', [])

        # Validate entities
        valid_entities_page = []
        for entity in entities_page:
            if isinstance(entity, dict) and 'id' in entity and 'type' in entity:
                valid_entities_page.append(entity)
            else:
                print(f"Warning: Skipping invalid entity: {str(entity)[:100]}")

        merged_entities_map: Dict[str, Dict[str, Any]] = {
            entity['id']: entity.copy() for entity in entities_main 
            if isinstance(entity, dict) and 'id' in entity
        }
        
        id_page_to_merged_id_map: Dict[str, str] = {}
        existing_ids = set(merged_entities_map.keys())

        # Track merge statistics by type
        merge_stats = {}

        for entity_pg in valid_entities_page:
            page_entity_id = entity_pg['id']
            entity_type = self._get_entity_type_unprefixed(entity_pg)
            
            if entity_type not in merge_stats:
                merge_stats[entity_type] = {"merged": 0, "preserved": 0}
            
            # Find potential match using ontology-aware logic
            matching_entity = None
            for candidate in merged_entities_map.values():
                if self._should_merge_entities_ontology_aware(entity_pg, candidate):
                    matching_entity = candidate
                    break

            if matching_entity:
                merged_id = matching_entity['id']
                id_page_to_merged_id_map[page_entity_id] = merged_id
                merged_entities_map[merged_id] = merge_entity_attributes(
                    merged_entities_map[merged_id], entity_pg
                )
                merge_stats[entity_type]["merged"] += 1
                print(f"  Merged {entity_type}: '{entity_pg.get('name', page_entity_id)}'")
            else:
                # Preserve as separate entity
                unique_id = self._generate_unique_id(page_entity_id, existing_ids, entity_pg)
                new_entity_copy = entity_pg.copy()
                new_entity_copy['id'] = unique_id
                new_entity_copy['_source_page'] = new_page_kg.get('page_number', 'unknown')
                
                merged_entities_map[unique_id] = new_entity_copy
                existing_ids.add(unique_id)
                id_page_to_merged_id_map[page_entity_id] = unique_id
                merge_stats[entity_type]["preserved"] += 1
                print(f"  Preserved {entity_type}: '{entity_pg.get('name', page_entity_id)}'")

        # Print merge statistics
        print("  Merge statistics by entity type:")
        for entity_type, stats in merge_stats.items():
            total = stats["merged"] + stats["preserved"]
            merge_rate = (stats["merged"] / total * 100) if total > 0 else 0
            print(f"    {entity_type}: {stats['merged']} merged, {stats['preserved']} preserved ({merge_rate:.1f}% merge rate)")

        final_merged_entities = list(merged_entities_map.values())
        
        # Merge relationships
        final_merged_relationships = self._merge_relationships(
            relationships_main, relationships_page, id_page_to_merged_id_map,
            set(entity['id'] for entity in final_merged_entities)
        )

        return {
            'entities': final_merged_entities,
            'relationships': final_merged_relationships
        }

    def _generate_unique_id(self, base_id: str, existing_ids: Set[str], entity: Dict[str, Any]) -> str:
        """Generate a unique ID incorporating entity type for better traceability."""
        entity_type = self._get_entity_type_unprefixed(entity)
        
        if base_id not in existing_ids:
            return base_id
        
        counter = 1
        while True:
            new_id = f"{base_id}_{entity_type}_{counter}"
            if new_id not in existing_ids:
                return new_id
            counter += 1

    def _merge_relationships(self, relationships_main: List[Dict[str, Any]],
                           relationships_page: List[Dict[str, Any]], 
                           id_page_to_merged_id_map: Dict[str, str],
                           valid_entity_ids: Set[str]) -> List[Dict[str, Any]]:
        """Enhanced relationship merging with validation."""
        final_merged_relationships: List[Dict[str, Any]] = []
        merged_relationship_fingerprints: Set[Tuple[str, str, str]] = set()

        # Add existing relationships
        for rel_main in relationships_main:
            if not (isinstance(rel_main, dict) and all(k in rel_main for k in ['source', 'target', 'type'])):
                continue
                
            if rel_main['source'] in valid_entity_ids and rel_main['target'] in valid_entity_ids:
                fingerprint = (rel_main['source'], rel_main['target'], rel_main['type'])
                if fingerprint not in merged_relationship_fingerprints:
                    final_merged_relationships.append(rel_main.copy())
                    merged_relationship_fingerprints.add(fingerprint)

        # Add new relationships
        added_count = 0
        for rel_page in relationships_page:
            if not (isinstance(rel_page, dict) and all(k in rel_page for k in ['source', 'target', 'type'])):
                continue
            
            source_id_new = id_page_to_merged_id_map.get(rel_page['source'])
            target_id_new = id_page_to_merged_id_map.get(rel_page['target'])
            
            if source_id_new and target_id_new and source_id_new in valid_entity_ids and target_id_new in valid_entity_ids:
                fingerprint = (source_id_new, target_id_new, rel_page['type'])
                if fingerprint not in merged_relationship_fingerprints:
                    new_rel_copy = rel_page.copy()
                    new_rel_copy['source'] = source_id_new
                    new_rel_copy['target'] = target_id_new
                    final_merged_relationships.append(new_rel_copy)
                    merged_relationship_fingerprints.add(fingerprint)
                    added_count += 1

        print(f"  Added {added_count} new relationships")
        return final_merged_relationships

    def merge_all_page_kgs(self, page_kgs: List[Dict[str, List[Dict[str, Any]]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Enhanced merging with ontology awareness."""
        if not page_kgs:
            return {'entities': [], 'relationships': []}

        print(f"Starting ontology-aware merge of {len(page_kgs)} page KGs...")
        
        merged_document_kg: Dict[str, List[Dict[str, Any]]] = {'entities': [], 'relationships': []}

        for i, page_kg in enumerate(page_kgs):
            if not isinstance(page_kg, dict):
                continue
            
            page_number = page_kg.get('page_number', i + 1)
            print(f"\nMerging page {page_number} KG...")
            
            page_kg_copy = {
                'entities': [e.copy() for e in page_kg.get('entities', []) if isinstance(e, dict)],
                'relationships': [r.copy() for r in page_kg.get('relationships', []) if isinstance(r, dict)],
                'page_number': page_number
            }
            
            merged_document_kg = self.merge_incrementally(merged_document_kg, page_kg_copy)
        
        final_entity_count = len(merged_document_kg.get('entities', []))
        final_rel_count = len(merged_document_kg.get('relationships', []))
        print(f"\nOntology-aware merge completed: {final_entity_count} entities, {final_rel_count} relationships")
        
        return merged_document_kg