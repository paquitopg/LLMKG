from typing import List, Dict, Any, Set, Tuple, Optional, Union

from .common_kg_utils import (
    find_matching_entity, # Still used for finding matches based on single-value attributes from input KGs
    normalize_entity_ids,
    clean_knowledge_graph,
    # merge_entity_attributes is NOT directly used for attribute values anymore;
    # we'll implement a new provenance-aware merging logic.
    # However, it might be used for non-provenance-tracked fields like 'type'.
    merge_entity_attributes as base_merge_entity_attributes 
)

class InterDocumentMerger:
    """
    Merges multiple document-level knowledge graphs into a single, comprehensive
    knowledge graph, including provenance information for entities, relationships,
    and attributes.
    """

    def __init__(self,
                 similarity_threshold: float = 0.75,
                 default_source_id: str = "aggregated"):
        """
        Initializes the InterDocumentMerger.

        Args:
            similarity_threshold (float): Threshold for considering entities similar.
            default_source_id (str): A default source ID to use if an existing attribute
                                     is converted to the provenance format.
        """
        self.similarity_threshold = similarity_threshold
        self.default_source_id = default_source_id

    def _add_provenance_value(self,
                              current_attribute_values: List[Dict[str, Any]],
                              new_value: Any,
                              new_source_doc_id: str
                             ) -> List[Dict[str, Any]]:
        """Helper to add a new value with its source to a list of provenance-tracked values, avoiding duplicates from the same source."""
        # Check if this exact value from this exact source already exists
        for entry in current_attribute_values:
            if entry.get("value") == new_value and entry.get("source_doc_id") == new_source_doc_id:
                return current_attribute_values # Already exists, no change

        current_attribute_values.append({"value": new_value, "source_doc_id": new_source_doc_id})
        return current_attribute_values

    def _merge_attributes_with_provenance(self,
                                         project_entity: Dict[str, Any],
                                         doc_entity_attributes: Dict[str, Any],
                                         doc_id: str
                                        ) -> Dict[str, Any]:
        """
        Merges attributes from doc_entity_attributes into project_entity,
        storing multiple values with provenance if they differ or come from new sources.
        Core attributes like 'id' and 'type' are handled specially.
        """
        merged_attrs = project_entity.copy()

        for attr_key, new_attr_value in doc_entity_attributes.items():
            if attr_key in ["id", "type", "_source_document_ids"]: # Handle these separately
                continue

            current_attr_value_or_list = merged_attrs.get(attr_key)

            if current_attr_value_or_list is None:
                # Attribute does not exist in project_entity, add it with provenance
                merged_attrs[attr_key] = [{"value": new_attr_value, "source_doc_id": doc_id}]
            else:
                if isinstance(current_attr_value_or_list, list) and \
                   all(isinstance(item, dict) and "value" in item and "source_doc_id" in item for item in current_attr_value_or_list):
                    # Already in provenance format, add new value if different or new source
                    merged_attrs[attr_key] = self._add_provenance_value(
                        current_attr_value_or_list, new_attr_value, doc_id
                    )
                else:
                    # Was a single value, convert to provenance format
                    # Use default_source_id or try to infer if project_entity has _source_document_ids
                    existing_source_id = self.default_source_id
                    if merged_attrs.get("_source_document_ids"):
                        # If it's the first time converting, this might be tricky.
                        # For simplicity, if it has only one source so far, use that.
                        # Otherwise, use default_source_id.
                        if len(merged_attrs["_source_document_ids"]) == 1:
                            existing_source_id = merged_attrs["_source_document_ids"][0]
                    
                    provenance_list = [{"value": current_attr_value_or_list, "source_doc_id": existing_source_id}]
                    merged_attrs[attr_key] = self._add_provenance_value(
                        provenance_list, new_attr_value, doc_id
                    )
        return merged_attrs

    def _merge_single_document_kg_into_project_kg(
            self,
            current_project_kg: Dict[str, List[Dict[str, Any]]],
            doc_kg_wrapper: Dict[str, Any] # Now expects {"document_id": ..., "entities": ..., "relationships": ...}
           ) -> Dict[str, List[Dict[str, Any]]]:
        
        doc_id = doc_kg_wrapper.get("document_id", self.default_source_id)
        document_kg_to_add = {
            "entities": doc_kg_wrapper.get("entities", []),
            "relationships": doc_kg_wrapper.get("relationships", [])
        }

        if not document_kg_to_add["entities"] and not document_kg_to_add["relationships"]:
            return current_project_kg

        project_entities = current_project_kg.get('entities', [])
        project_relationships = current_project_kg.get('relationships', [])
        
        merged_entities_map: Dict[str, Dict[str, Any]] = {
            entity['id']: entity.copy() for entity in project_entities 
            if isinstance(entity, dict) and 'id' in entity
        }
        id_doc_to_project_id_map: Dict[str, str] = {}

        for entity_doc in document_kg_to_add.get('entities', []):
            if not (isinstance(entity_doc, dict) and 'id' in entity_doc and 'type' in entity_doc):
                print(f"Warning (InterDocMerge): Entity from doc '{doc_id}' missing ID/type, skipping: {str(entity_doc)[:100]}")
                continue
            
            doc_entity_id = entity_doc['id']
            
            matching_entity_in_project = find_matching_entity(
                entity_doc, # find_matching_entity expects single-value attributes for comparison
                list(merged_entities_map.values()),
                threshold=self.similarity_threshold
            )

            if matching_entity_in_project:
                project_id = matching_entity_in_project['id']
                id_doc_to_project_id_map[doc_entity_id] = project_id
                
                # Merge attributes with provenance
                # Attributes from entity_doc are still single-valued here
                updated_attrs = self._merge_attributes_with_provenance(
                    merged_entities_map[project_id], entity_doc, doc_id
                )
                merged_entities_map[project_id].update(updated_attrs) # Update existing entity

                # Update type using base merge (prefer non-empty)
                # This assumes 'type' should remain a single value.
                # Base merge might prefer longer or non-empty string for 'type'.
                merged_type_dict = base_merge_entity_attributes(
                    {'type': merged_entities_map[project_id].get('type')}, 
                    {'type': entity_doc.get('type')}
                )
                merged_entities_map[project_id]['type'] = merged_type_dict.get('type')


                # Add doc_id to _source_document_ids list (unique)
                if "_source_document_ids" not in merged_entities_map[project_id]:
                    merged_entities_map[project_id]["_source_document_ids"] = []
                if doc_id not in merged_entities_map[project_id]["_source_document_ids"]:
                    merged_entities_map[project_id]["_source_document_ids"].append(doc_id)
            else:
                # Add entity_doc as new, initialize attributes with provenance
                current_new_id_in_project = doc_entity_id
                id_counter = 0
                while current_new_id_in_project in merged_entities_map:
                    id_counter += 1
                    new_suffixed_id = f"{doc_entity_id}_projdup{id_counter}"
                    current_new_id_in_project = new_suffixed_id
                
                new_project_entity = {'id': current_new_id_in_project, 'type': entity_doc['type']}
                # Initialize attributes with provenance
                new_project_entity = self._merge_attributes_with_provenance(
                    new_project_entity, entity_doc, doc_id
                )
                new_project_entity["_source_document_ids"] = [doc_id]
                
                merged_entities_map[current_new_id_in_project] = new_project_entity
                id_doc_to_project_id_map[doc_entity_id] = current_new_id_in_project
        
        final_project_entities = list(merged_entities_map.values())
        final_project_relationships: List[Dict[str, Any]] = []
        project_relationship_fingerprints: Set[Tuple[str, str, str]] = set()
        all_current_project_entity_ids = {e['id'] for e in final_project_entities}

        # Add existing relationships, ensuring they get _source_document_ids if not present
        for rel_proj in project_relationships:
            if not (isinstance(rel_proj, dict) and all(k in rel_proj for k in ['source', 'target', 'type'])):
                continue
            if rel_proj['source'] in all_current_project_entity_ids and \
               rel_proj['target'] in all_current_project_entity_ids:
                fingerprint = (rel_proj['source'], rel_proj['target'], rel_proj['type'])
                if fingerprint not in project_relationship_fingerprints:
                    rel_copy = rel_proj.copy()
                    if "_source_document_ids" not in rel_copy: # Add if merging from older format
                        rel_copy["_source_document_ids"] = [self.default_source_id] 
                    final_project_relationships.append(rel_copy)
                    project_relationship_fingerprints.add(fingerprint)

        # Add new relationships with provenance
        for rel_doc in document_kg_to_add.get('relationships', []):
            if not (isinstance(rel_doc, dict) and all(k in rel_doc for k in ['source', 'target', 'type'])):
                continue
            
            original_source_doc = rel_doc['source']
            original_target_doc = rel_doc['target']
            rel_type = rel_doc['type']
            
            source_id_in_project = id_doc_to_project_id_map.get(original_source_doc)
            target_id_in_project = id_doc_to_project_id_map.get(original_target_doc)
            
            if source_id_in_project and target_id_in_project:
                if source_id_in_project in all_current_project_entity_ids and \
                   target_id_in_project in all_current_project_entity_ids:
                    fingerprint = (source_id_in_project, target_id_in_project, rel_type)
                    
                    # Check if this exact relationship already exists from any source
                    existing_rel_index = -1
                    for idx, prj_rel in enumerate(final_project_relationships):
                        if prj_rel['source'] == source_id_in_project and \
                           prj_rel['target'] == target_id_in_project and \
                           prj_rel['type'] == rel_type:
                            existing_rel_index = idx
                            break
                    
                    if existing_rel_index != -1:
                        # Relationship already exists, just add this doc_id to its provenance
                        if doc_id not in final_project_relationships[existing_rel_index]["_source_document_ids"]:
                            final_project_relationships[existing_rel_index]["_source_document_ids"].append(doc_id)
                    else:
                        # New relationship for the project KG
                        new_rel_copy = rel_doc.copy()
                        new_rel_copy['source'] = source_id_in_project
                        new_rel_copy['target'] = target_id_in_project
                        new_rel_copy["_source_document_ids"] = [doc_id]
                        # Add other attributes from rel_doc with provenance if they exist and need it
                        final_project_relationships.append(new_rel_copy)
                        project_relationship_fingerprints.add(fingerprint) # Add new fingerprint
        
        return {
            'entities': final_project_entities,
            'relationships': final_project_relationships
        }

    def merge_project_kgs(self,
                         document_kgs_with_ids: List[Dict[str, Any]] # Expects list of {"document_id": ..., "entities": ..., "relationships": ...}
                        ) -> Dict[str, List[Dict[str, Any]]]:
        if not document_kgs_with_ids:
            return {'entities': [], 'relationships': []}

        final_project_kg: Dict[str, List[Dict[str, Any]]] = {'entities': [], 'relationships': []}

        for i, doc_kg_wrapper in enumerate(document_kgs_with_ids):
            doc_id_for_log = doc_kg_wrapper.get("document_id", f"unknown_doc_{i+1}")
            print(f"Inter-Document Merge: Processing document KG '{doc_id_for_log}' ({i+1}/{len(document_kgs_with_ids)})...")
            
            if isinstance(doc_kg_wrapper, dict) and "entities" in doc_kg_wrapper and "relationships" in doc_kg_wrapper:
                # Pass the entire wrapper which includes 'document_id'
                final_project_kg = self._merge_single_document_kg_into_project_kg(
                    final_project_kg, doc_kg_wrapper 
                )
            else:
                print(f"Warning (InterDocMerge): Item for document '{doc_id_for_log}' is not a well-formed KG wrapper, skipping.")
        
        print("Inter-document merging completed. Performing final cleanup and normalization...")
        if final_project_kg.get("entities") or final_project_kg.get("relationships"):
            final_project_kg = clean_knowledge_graph(final_project_kg) # clean_knowledge_graph should handle new attr structure
            final_project_kg = normalize_entity_ids(final_project_kg) # normalize_entity_ids also
        
        return final_project_kg