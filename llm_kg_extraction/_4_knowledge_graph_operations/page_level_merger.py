from typing import List, Dict, Any, Set, Tuple, Optional

from .common_kg_utils import (
    find_matching_entity,
    merge_entity_attributes,
    # _get_company_id_for_metric, # This is used by find_matching_entity/are_entities_similar
    # normalize_entity_ids, # Not typically used during incremental page merge, but after all merges
    # clean_knowledge_graph # Can be called periodically by the constructor class
)

class PageLevelMerger:
    """
    Merges knowledge graphs generated from individual pages of the same document.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initializes the PageLevelMerger.

        Args:
            similarity_threshold (float): The threshold for considering entities similar
                                          enough to merge. Passed to find_matching_entity.
        """
        self.similarity_threshold = similarity_threshold

    def merge_incrementally(self,
                            current_document_kg: Dict[str, List[Dict[str, Any]]],
                            new_page_kg: Dict[str, List[Dict[str, Any]]]
                           ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Merges a new page's knowledge graph into the current accumulated document knowledge graph.
        This is an adaptation of the original merge_knowledge_graphs function.

        Args:
            current_document_kg (Dict): The knowledge graph accumulated so far from previous pages.
                                        Expected format: {"entities": [...], "relationships": [...]}.
            new_page_kg (Dict): The knowledge graph extracted from the new page.
                                Expected format: {"entities": [...], "relationships": [...]}.

        Returns:
            Dict: The updated document knowledge graph.
        """
        if not new_page_kg or (not new_page_kg.get("entities") and not new_page_kg.get("relationships")):
            return current_document_kg # No new information to merge

        # Ensure current_document_kg is well-formed
        if not isinstance(current_document_kg, dict):
            current_document_kg = {"entities": [], "relationships": []}

        entities_main = current_document_kg.get('entities', [])
        relationships_main = current_document_kg.get('relationships', [])
        
        entities_page = new_page_kg.get('entities', [])
        relationships_page = new_page_kg.get('relationships', [])

        # Create a mutable copy of main entities for merging
        # The keys are existing IDs, values are the entity dicts.
        # Ensure entities are copied to avoid modifying original dicts if they come from elsewhere.
        merged_entities_map: Dict[str, Dict[str, Any]] = {
            entity['id']: entity.copy() for entity in entities_main if isinstance(entity, dict) and 'id' in entity
        }
        
        # Map to track IDs from the new_page_kg to their corresponding IDs in the merged_entities_map
        id_page_to_merged_id_map: Dict[str, str] = {}

        for entity_pg in entities_page:
            if not isinstance(entity_pg, dict) or 'id' not in entity_pg:
                print(f"Warning: Entity in new_page_kg missing ID or malformed, skipping: {str(entity_pg)[:100]}")
                continue
            
            page_entity_id = entity_pg['id']
            
            # Find a potential match in the already merged entities
            # find_matching_entity now uses are_entities_similar, which has contextual checks
            # for metrics (using _get_company_id_for_metric from common_kg_utils implicitly).
            matching_entity_in_merged = find_matching_entity(
                entity_pg,
                list(merged_entities_map.values()), # Search within current state of merged entities
                threshold=self.similarity_threshold
            )

            if matching_entity_in_merged:
                merged_id = matching_entity_in_merged['id']
                id_page_to_merged_id_map[page_entity_id] = merged_id
                # Merge attributes of entity_pg into the existing entity in merged_entities_map
                merged_entities_map[merged_id] = merge_entity_attributes(
                    merged_entities_map[merged_id], entity_pg
                )
            else:
                # Add entity_pg as a new entity. Ensure its ID is unique within the merged set.
                # This attempts to preserve page-local IDs (e.g., "e1", "e2" from a page)
                # but makes them unique in the context of the whole document by suffixing if needed.
                # This is important if pages independently generate simple IDs.
                current_new_id = page_entity_id
                id_counter = 0
                while current_new_id in merged_entities_map:
                    id_counter += 1
                    # Suffix with _dup or _pg<actual_page_num> if available for better tracing
                    # For simplicity, using _dup suffix.
                    new_suffixed_id = f"{page_entity_id}_dup{id_counter}"
                    # This warning is helpful for debugging ID generation/clashes
                    # print(f"Warning: ID clash for new entity '{page_entity_id}'. Remapped to '{new_suffixed_id}' during page merge.")
                    current_new_id = new_suffixed_id
                
                new_entity_pg_copy = entity_pg.copy()
                new_entity_pg_copy['id'] = current_new_id # Assign the guaranteed unique ID
                merged_entities_map[current_new_id] = new_entity_pg_copy
                id_page_to_merged_id_map[page_entity_id] = current_new_id

        final_merged_entities = list(merged_entities_map.values())
        
        # Merge relationships
        # Start with existing relationships from current_document_kg, ensuring they are valid
        final_merged_relationships: List[Dict[str, Any]] = []
        merged_relationship_fingerprints: Set[Tuple[str, str, str]] = set()
        all_final_entity_ids = {e['id'] for e in final_merged_entities if isinstance(e, dict) and 'id' in e}


        for rel_main in relationships_main:
            if not (isinstance(rel_main, dict) and 'source' in rel_main and 'target' in rel_main and 'type' in rel_main):
                continue
            # IDs from relationships_main should already be valid if current_document_kg is clean
            if rel_main['source'] in all_final_entity_ids and rel_main['target'] in all_final_entity_ids:
                fingerprint = (rel_main['source'], rel_main['target'], rel_main['type'])
                if fingerprint not in merged_relationship_fingerprints:
                    final_merged_relationships.append(rel_main.copy())
                    merged_relationship_fingerprints.add(fingerprint)

        # Add relationships from new_page_kg, remapping source/target IDs
        for rel_page in relationships_page:
            if not (isinstance(rel_page, dict) and 'source' in rel_page and 'target' in rel_page and 'type' in rel_page):
                continue
            
            original_source_page = rel_page['source']
            original_target_page = rel_page['target']
            rel_type = rel_page['type']
            
            # Get the new, document-level IDs for source and target from the map
            source_id_new = id_page_to_merged_id_map.get(original_source_page)
            target_id_new = id_page_to_merged_id_map.get(original_target_page)
            
            if source_id_new and target_id_new:
                 # Ensure these remapped IDs are actually in the final entity list (they should be by construction)
                if source_id_new in all_final_entity_ids and target_id_new in all_final_entity_ids:
                    fingerprint = (source_id_new, target_id_new, rel_type)
                    if fingerprint not in merged_relationship_fingerprints:
                        new_rel_copy = rel_page.copy()
                        new_rel_copy['source'] = source_id_new
                        new_rel_copy['target'] = target_id_new
                        final_merged_relationships.append(new_rel_copy)
                        merged_relationship_fingerprints.add(fingerprint)
                # else:
                    # print(f"Debug (PageMerge): Dropping rel from page KG. Remapped S:{source_id_new} or T:{target_id_new} not in final entities. Orig S:{original_source_page}, T:{original_target_page}")
            # else:
                # print(f"Debug (PageMerge): Dropping rel from page KG. S:{original_source_page} or T:{original_target_page} not found in id_page_to_merged_id_map.")

        return {
            'entities': final_merged_entities,
            'relationships': final_merged_relationships
        }

    def merge_all_page_kgs(self,
                           page_kgs: List[Dict[str, List[Dict[str, Any]]]]
                          ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Merges a list of page-level knowledge graphs into a single document knowledge graph.
        This is an adaptation of the original merge_multiple_knowledge_graphs function.

        Args:
            page_kgs (List[Dict]): A list of knowledge graphs, where each graph
                                   is from a page of the document. Expected format for each
                                   item: {"entities": [...], "relationships": [...]}.

        Returns:
            Dict: The merged document knowledge graph.
        """
        if not page_kgs:
            return {'entities': [], 'relationships': []}

        # Initialize with the first page's KG, ensuring it's a copy
        # If first KG is empty or malformed, start with an empty graph
        first_kg = page_kgs[0]
        if isinstance(first_kg, dict):
            # Deep copy entities and relationships to prevent modification of original graph objects
            merged_document_kg: Dict[str, List[Dict[str, Any]]] = {
                'entities': [e.copy() for e in first_kg.get('entities', []) if isinstance(e, dict)],
                'relationships': [r.copy() for r in first_kg.get('relationships', []) if isinstance(r, dict)]
            }
        else: # Should not happen if PageLLMProcessor returns correctly
            print("Warning: First page KG is not a dictionary, starting with an empty graph for merging.")
            merged_document_kg = {'entities': [], 'relationships': []}


        # Iteratively merge subsequent page KGs
        for i in range(1, len(page_kgs)):
            current_page_kg = page_kgs[i]
            if isinstance(current_page_kg, dict):
                # Ensure entities/relationships of the graph to merge are copied
                page_kg_to_merge_copy = {
                    'entities': [e.copy() for e in current_page_kg.get('entities', []) if isinstance(e, dict)],
                    'relationships': [r.copy() for r in current_page_kg.get('relationships', []) if isinstance(r, dict)]
                }
                merged_document_kg = self.merge_incrementally(merged_document_kg, page_kg_to_merge_copy)
            else:
                print(f"Warning: Item at index {i} in page_kgs list is not a dictionary, skipping its merge.")
        
        return merged_document_kg