from typing import List, Dict, Any, Set, Tuple, Optional

# Import the common KG utility functions
from .common_kg_utils import (
    find_matching_entity_pekg,
    get_entity_primary_name,
    merge_entity_attributes
)

class PageLevelMerger:
    """
    Page-level merger specifically designed for PEKG ontology.
    Uses entity-type specific strategies without code duplication.
    """

    def __init__(self, ontology, similarity_threshold: float = 0.75):
        """
        Initialize the PEKG-aware page merger.
        
        Args:
            ontology: The PEKG ontology instance
            similarity_threshold: Default threshold (entity-specific overrides apply)
        """
        self.ontology = ontology
        self.similarity_threshold = similarity_threshold
        
        # Define which entity types should be preserved vs merged
        self.entity_strategies = {
            # Context entities - preserve separately (high threshold)
            "transactioncontext": {"strategy": "preserve", "threshold": 0.98},
            
            # Metrics - strict matching required  
            "financialmetric": {"strategy": "strict", "threshold": 0.95},
            "operationalkpi": {"strategy": "strict", "threshold": 0.90},
            "headcount": {"strategy": "strict", "threshold": 0.90},
            
            # Reference entities - merge liberally
            "company": {"strategy": "liberal", "threshold": 0.70},
            "person": {"strategy": "liberal", "threshold": 0.75},
            "technology": {"strategy": "liberal", "threshold": 0.75},
            "location": {"strategy": "liberal", "threshold": 0.75},
            "governmentbody": {"strategy": "liberal", "threshold": 0.80},
            "usecaseorindustry": {"strategy": "liberal", "threshold": 0.75},
            
            # Business entities - moderate matching
            "advisor": {"strategy": "moderate", "threshold": 0.80},
            "shareholder": {"strategy": "moderate", "threshold": 0.85},
            "productorservice": {"strategy": "moderate", "threshold": 0.80},
            "historicalevent": {"strategy": "moderate", "threshold": 0.85},
            "marketcontext": {"strategy": "moderate", "threshold": 0.80},
            "position": {"strategy": "strict", "threshold": 0.90},
        }

    def _get_entity_type_unprefixed(self, entity: Dict[str, Any]) -> str:
        """Extract unprefixed entity type."""
        entity_type = entity.get('type', '')
        if not entity_type:
            return ''
        if ':' in entity_type:
            return entity_type.split(':')[-1].lower()
        return entity_type.lower()

    def _are_nearly_identical(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """Check if entities are nearly identical (for preserve strategy)."""
        # Check if all key attributes are identical
        entity_type = self._get_entity_type_unprefixed(entity1)
        
        if entity_type == "transactioncontext":
            # All key fields must match exactly
            key_fields = ["contextName", "typeSought", "status"]
            for field in key_fields:
                val1 = entity1.get(field)
                val2 = entity2.get(field)
                if val1 != val2:
                    return False
            return True
        
        # For other preserve types, fall back to standard similarity
        return True

    def merge_incrementally(self, current_document_kg: Dict[str, List[Dict[str, Any]]],
                          new_page_kg: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Merge a new page's KG into the current document KG using PEKG-aware strategies.
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

        # Create merged entities map
        merged_entities_map: Dict[str, Dict[str, Any]] = {
            entity['id']: entity.copy() for entity in entities_main 
            if isinstance(entity, dict) and 'id' in entity
        }
        
        id_page_to_merged_id_map: Dict[str, str] = {}
        existing_ids = set(merged_entities_map.keys())

        # Track statistics by entity type and strategy
        merge_stats = {}

        for entity_pg in valid_entities_page:
            page_entity_id = entity_pg['id']
            entity_type = self._get_entity_type_unprefixed(entity_pg)
            strategy_info = self.entity_strategies.get(entity_type, {"strategy": "moderate", "threshold" : 0.75})
            strategy = strategy_info["strategy"]
            
            if entity_type not in merge_stats:
                merge_stats[entity_type] = {"merged": 0, "preserved": 0, "strategy": strategy}
            
            # Find potential match using the dedicated function
            strategy_info = self.entity_strategies.get(entity_type, {"strategy": "moderate", "threshold": 0.75})
            threshold = strategy_info["threshold"]
            
            matching_entity = find_matching_entity_pekg(
                entity_pg,          
                list(merged_entities_map.values()), 
                threshold=threshold
            )

            if matching_entity:
                # Merge entities
                merged_id = matching_entity['id']
                id_page_to_merged_id_map[page_entity_id] = merged_id
                merged_entities_map[merged_id] = merge_entity_attributes(
                    merged_entities_map[merged_id], entity_pg
                )
                merge_stats[entity_type]["merged"] += 1
                
            else:
                # Preserve as separate entity
                unique_id = self._generate_unique_id(page_entity_id, existing_ids, entity_type)
                new_entity_copy = entity_pg.copy()
                new_entity_copy['id'] = unique_id
                new_entity_copy['_source_page'] = new_page_kg.get('page_number', 'unknown')
                
                merged_entities_map[unique_id] = new_entity_copy
                existing_ids.add(unique_id)
                id_page_to_merged_id_map[page_entity_id] = unique_id
                merge_stats[entity_type]["preserved"] += 1

        # Print detailed merge statistics
        self._print_merge_statistics(merge_stats)

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

    def _generate_unique_id(self, base_id: str, existing_ids: Set[str], entity_type: str) -> str:
        """Generate a unique ID incorporating entity type for better traceability."""
        if base_id not in existing_ids:
            return base_id
        
        counter = 1
        while True:
            new_id = f"{base_id}_{entity_type}_{counter}"
            if new_id not in existing_ids:
                return new_id
            counter += 1

    def _print_merge_statistics(self, merge_stats: Dict[str, Dict[str, Any]]) -> None:
        """Print detailed merge statistics by entity type and strategy."""
        print("  PEKG Entity Merge Statistics:")
        
        strategy_groups = {"preserve": [], "strict": [], "moderate": [], "liberal": []}
        
        for entity_type, stats in merge_stats.items():
            strategy = stats["strategy"]
            total = stats["merged"] + stats["preserved"]
            merge_rate = (stats["merged"] / total * 100) if total > 0 else 0
            
            entry = f"    {entity_type}: {stats['merged']} merged, {stats['preserved']} preserved ({merge_rate:.1f}% merge rate)"
            strategy_groups[strategy].append(entry)
        
        for strategy, entries in strategy_groups.items():
            if entries:
                print(f"  {strategy.upper()} strategy:")
                for entry in entries:
                    print(entry)

    def _merge_relationships(self, relationships_main: List[Dict[str, Any]],
                           relationships_page: List[Dict[str, Any]], 
                           id_page_to_merged_id_map: Dict[str, str],
                           valid_entity_ids: Set[str]) -> List[Dict[str, Any]]:
        """Merge relationships with validation."""
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
        """Merge all page KGs using PEKG-aware strategies."""
        if not page_kgs:
            return {'entities': [], 'relationships': []}

        print(f"Starting PEKG-aware merge of {len(page_kgs)} page KGs...")
        
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
        print(f"\nPEKG-aware merge completed: {final_entity_count} entities, {final_rel_count} relationships")
        
        return merged_document_kg