import json
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter

class PageMergerDiagnostic:
    """
    Diagnostic tool to analyze what happens during page-level KG merging.
    Helps identify why entities are being lost or merged unexpectedly.
    """
    
    def __init__(self):
        self.merge_log = []
        self.entity_fate_log = {}  # Track what happens to each entity
        self.similarity_scores = []
        
    def analyze_merger_behavior(self, page_kgs: List[Dict[str, Any]], 
                              final_kg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive analysis of what happened during merging.
        
        Args:
            page_kgs: List of individual page knowledge graphs
            final_kg: The merged knowledge graph result
            
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {
            "summary": {},
            "entity_type_analysis": {},
            "merge_patterns": {},
            "lost_entities": [],
            "unexpected_merges": [],
            "recommendations": []
        }
        
        # Collect all original entities
        all_original_entities = []
        page_entity_map = {}  # entity_id -> page_number
        
        for page_idx, page_kg in enumerate(page_kgs):
            page_num = page_kg.get('page_number', page_idx + 1)
            for entity in page_kg.get('entities', []):
                if isinstance(entity, dict) and 'id' in entity:
                    all_original_entities.append(entity)
                    page_entity_map[entity['id']] = page_num
        
        final_entities = final_kg.get('entities', [])
        
        # Basic statistics
        original_count = len(all_original_entities)
        final_count = len(final_entities)
        loss_percentage = ((original_count - final_count) / original_count * 100) if original_count > 0 else 0
        
        analysis["summary"] = {
            "original_entity_count": original_count,
            "final_entity_count": final_count,
            "entities_lost": original_count - final_count,
            "loss_percentage": round(loss_percentage, 1),
            "pages_processed": len(page_kgs)
        }
        
        # Analyze by entity type
        analysis["entity_type_analysis"] = self._analyze_by_entity_type(
            all_original_entities, final_entities
        )
        
        # Find potential merging issues
        analysis["merge_patterns"] = self._analyze_merge_patterns(
            all_original_entities, final_entities
        )
        
        # Identify specific lost entities
        analysis["lost_entities"] = self._identify_lost_entities(
            all_original_entities, final_entities, page_entity_map
        )
        
        # Find unexpected merges
        analysis["unexpected_merges"] = self._find_unexpected_merges(
            all_original_entities, final_entities
        )
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_by_entity_type(self, original_entities: List[Dict[str, Any]], 
                               final_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entity loss/preservation by type."""
        original_by_type = defaultdict(list)
        final_by_type = defaultdict(list)
        
        for entity in original_entities:
            entity_type = self._get_entity_type(entity)
            original_by_type[entity_type].append(entity)
        
        for entity in final_entities:
            entity_type = self._get_entity_type(entity)
            final_by_type[entity_type].append(entity)
        
        type_analysis = {}
        for entity_type in original_by_type:
            original_count = len(original_by_type[entity_type])
            final_count = len(final_by_type.get(entity_type, []))
            loss_count = original_count - final_count
            loss_percentage = (loss_count / original_count * 100) if original_count > 0 else 0
            
            type_analysis[entity_type] = {
                "original_count": original_count,
                "final_count": final_count,
                "lost_count": loss_count,
                "loss_percentage": round(loss_percentage, 1),
                "sample_entities": [
                    entity.get('name', entity.get('id', 'unnamed'))
                    for entity in original_by_type[entity_type][:3]
                ]
            }
        
        return type_analysis
    
    def _analyze_merge_patterns(self, original_entities: List[Dict[str, Any]], 
                               final_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in how entities are being merged."""
        patterns = {
            "high_similarity_merges": [],
            "low_similarity_merges": [],
            "cross_page_merges": [],
            "within_page_duplicates": []
        }
        
        # Group original entities by name similarity
        name_groups = defaultdict(list)
        for entity in original_entities:
            entity_name = self._get_entity_name(entity).lower().strip()
            if entity_name:
                name_groups[entity_name].append(entity)
        
        # Analyze groups with multiple entities
        for name, entities in name_groups.items():
            if len(entities) > 1:
                # Check if these got merged into one final entity
                entity_types = set(self._get_entity_type(e) for e in entities)
                if len(entity_types) == 1:  # Same type
                    # Find corresponding final entities
                    matching_finals = [
                        f for f in final_entities 
                        if self._get_entity_name(f).lower().strip() == name
                        and self._get_entity_type(f) == list(entity_types)[0]
                    ]
                    
                    if len(matching_finals) < len(entities):
                        merge_info = {
                            "name": name,
                            "type": list(entity_types)[0],
                            "original_count": len(entities),
                            "final_count": len(matching_finals),
                            "entities": [
                                {
                                    "id": e.get('id'),
                                    "page": e.get('_source_page', 'unknown'),
                                    "details": self._extract_key_details(e)
                                }
                                for e in entities
                            ]
                        }
                        
                        # Categorize based on similarity
                        if self._entities_highly_similar(entities):
                            patterns["high_similarity_merges"].append(merge_info)
                        else:
                            patterns["low_similarity_merges"].append(merge_info)
        
        return patterns
    
    def _identify_lost_entities(self, original_entities: List[Dict[str, Any]], 
                              final_entities: List[Dict[str, Any]], 
                              page_entity_map: Dict[str, int]) -> List[Dict[str, Any]]:
        """Identify specific entities that were lost during merging."""
        lost_entities = []
        
        final_names_and_types = set()
        for entity in final_entities:
            name = self._get_entity_name(entity).lower().strip()
            entity_type = self._get_entity_type(entity)
            final_names_and_types.add((name, entity_type))
        
        for entity in original_entities:
            name = self._get_entity_name(entity).lower().strip()
            entity_type = self._get_entity_type(entity)
            
            if (name, entity_type) not in final_names_and_types:
                lost_entities.append({
                    "id": entity.get('id'),
                    "name": name,
                    "type": entity_type,
                    "page": page_entity_map.get(entity.get('id'), 'unknown'),
                    "details": self._extract_key_details(entity),
                    "likely_reason": self._guess_loss_reason(entity, final_entities)
                })
        
        return lost_entities
    
    def _find_unexpected_merges(self, original_entities: List[Dict[str, Any]], 
                               final_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find cases where entities were merged but probably shouldn't have been."""
        unexpected_merges = []
        
        # Group original entities by their names
        name_groups = defaultdict(list)
        for entity in original_entities:
            name = self._get_entity_name(entity).lower().strip()
            if name:
                name_groups[name].append(entity)
        
        for name, entities in name_groups.items():
            if len(entities) > 1:
                # Check if these entities have significant differences
                differences = self._find_entity_differences(entities)
                if differences:
                    # Check if they got merged (fewer in final than original)
                    same_type = entities[0].get('type', '')
                    matching_finals = [
                        f for f in final_entities 
                        if self._get_entity_name(f).lower().strip() == name
                        and f.get('type', '') == same_type
                    ]
                    
                    if len(matching_finals) < len(entities):
                        unexpected_merges.append({
                            "name": name,
                            "type": same_type,
                            "original_count": len(entities),
                            "final_count": len(matching_finals),
                            "differences_found": differences,
                            "entities": [
                                {
                                    "id": e.get('id'),
                                    "page": e.get('_source_page', 'unknown'),
                                    "key_details": self._extract_key_details(e)
                                }
                                for e in entities
                            ]
                        })
        
        return unexpected_merges
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on the analysis."""
        recommendations = []
        
        loss_percentage = analysis["summary"]["loss_percentage"]
        
        if loss_percentage > 50:
            recommendations.append(
                "CRITICAL: Over 50% entity loss suggests overly aggressive merging. "
                "Consider increasing similarity thresholds or implementing entity-type specific strategies."
            )
        
        # Check for context entity losses
        type_analysis = analysis["entity_type_analysis"]
        for entity_type, stats in type_analysis.items():
            if "context" in entity_type.lower() and stats["loss_percentage"] > 20:
                recommendations.append(
                    f"Context entities ({entity_type}) are being lost. "
                    "These should rarely be merged. Implement preserve_all strategy for context entities."
                )
        
        # Check for unexpected merges
        if analysis["unexpected_merges"]:
            recommendations.append(
                "Found entities with different details being merged inappropriately. "
                "Consider adding stricter validation for entities with different numerical values, "
                "temporal information, or other distinguishing attributes."
            )
        
        # Check for high loss in specific types
        high_loss_types = [
            entity_type for entity_type, stats in type_analysis.items() 
            if stats["loss_percentage"] > 70
        ]
        
        if high_loss_types:
            recommendations.append(
                f"High loss rates in: {', '.join(high_loss_types)}. "
                "These entity types may need custom preservation logic."
            )
        
        return recommendations
    
    def _get_entity_type(self, entity: Dict[str, Any]) -> str:
        """Extract the entity type, handling prefixes."""
        entity_type = entity.get('type', 'unknown')
        if ':' in entity_type:
            return entity_type.split(':')[-1].lower()
        return entity_type.lower()
    
    def _get_entity_name(self, entity: Dict[str, Any]) -> str:
        """Extract the most relevant name from an entity."""
        name_fields = ['name', 'metricName', 'kpiName', 'productName', 'fullName', 'contextName']
        for field in name_fields:
            if entity.get(field):
                return str(entity[field])
        return entity.get('id', 'unnamed')
    
    def _extract_key_details(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key identifying details from an entity."""
        details = {}
        
        # Numerical values
        numerical_fields = ['metricValue', 'percentageValue', 'amount', 'headcountValue']
        for field in numerical_fields:
            if field in entity and entity[field] is not None:
                details[field] = entity[field]
        
        # Temporal information
        temporal_fields = ['fiscalPeriod', 'dateOrYear', 'kpiDateOrPeriod']
        for field in temporal_fields:
            if field in entity and entity[field] is not None:
                details[field] = entity[field]
        
        # Other identifying info
        other_fields = ['metricUnit', 'currency', 'location', 'contextType']
        for field in other_fields:
            if field in entity and entity[field] is not None:
                details[field] = entity[field]
        
        return details
    
    def _entities_highly_similar(self, entities: List[Dict[str, Any]]) -> bool:
        """Check if a group of entities are highly similar."""
        if len(entities) < 2:
            return True
        
        # Compare first entity with all others
        first_entity = entities[0]
        for other_entity in entities[1:]:
            # Check key details
            details1 = self._extract_key_details(first_entity)
            details2 = self._extract_key_details(other_entity)
            
            # If they have different key details, they're not highly similar
            for key in details1:
                if key in details2 and details1[key] != details2[key]:
                    return False
        
        return True
    
    def _find_entity_differences(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Find significant differences between entities with the same name."""
        if len(entities) < 2:
            return []
        
        differences = []
        
        # Check for different numerical values
        for field in ['metricValue', 'percentageValue', 'amount']:
            values = set()
            for entity in entities:
                if field in entity and entity[field] is not None:
                    values.add(entity[field])
            
            if len(values) > 1:
                differences.append(f"Different {field} values: {list(values)}")
        
        # Check for different temporal information
        for field in ['fiscalPeriod', 'dateOrYear']:
            values = set()
            for entity in entities:
                if field in entity and entity[field] is not None:
                    values.add(str(entity[field]))
            
            if len(values) > 1:
                differences.append(f"Different {field} values: {list(values)}")
        
        # Check for different pages
        pages = set()
        for entity in entities:
            page = entity.get('_source_page')
            if page:
                pages.add(page)
        
        if len(pages) > 1:
            differences.append(f"Found on different pages: {list(pages)}")
        
        return differences
    
    def _guess_loss_reason(self, lost_entity: Dict[str, Any], 
                          final_entities: List[Dict[str, Any]]) -> str:
        """Guess why an entity was lost during merging."""
        entity_name = self._get_entity_name(lost_entity).lower().strip()
        entity_type = self._get_entity_type(lost_entity)
        
        # Check if there's a similar entity in final
        for final_entity in final_entities:
            final_name = self._get_entity_name(final_entity).lower().strip()
            final_type = self._get_entity_type(final_entity)
            
            if final_type == entity_type:
                if final_name == entity_name:
                    return "Merged with identical name/type entity"
                elif entity_name in final_name or final_name in entity_name:
                    return "Merged with similar name entity"
        
        # Check if it's a context entity
        if "context" in entity_type:
            return "Context entity inappropriately merged or filtered"
        
        return "Unknown - possibly filtered during validation"

# Usage function
def diagnose_merger_issues(page_kgs: List[Dict[str, Any]], 
                          final_kg: Dict[str, Any]) -> None:
    """
    Run complete diagnostic analysis and print results.
    
    Args:
        page_kgs: List of individual page knowledge graphs
        final_kg: The final merged knowledge graph
    """
    diagnostic = PageMergerDiagnostic()
    analysis = diagnostic.analyze_merger_behavior(page_kgs, final_kg)
    
    print("=" * 60)
    print("PAGE MERGER DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # Summary
    summary = analysis["summary"]
    print(f"\nSUMMARY:")
    print(f"  Original entities: {summary['original_entity_count']}")
    print(f"  Final entities: {summary['final_entity_count']}")
    print(f"  Entities lost: {summary['entities_lost']}")
    print(f"  Loss percentage: {summary['loss_percentage']}%")
    
    # Entity type analysis
    print(f"\nENTITY TYPE ANALYSIS:")
    type_analysis = analysis["entity_type_analysis"]
    for entity_type, stats in sorted(type_analysis.items(), 
                                   key=lambda x: x[1]['loss_percentage'], 
                                   reverse=True):
        print(f"  {entity_type}:")
        print(f"    Original: {stats['original_count']}, Final: {stats['final_count']}")
        print(f"    Lost: {stats['lost_count']} ({stats['loss_percentage']}%)")
        print(f"    Examples: {', '.join(stats['sample_entities'])}")
    
    # Lost entities
    lost_entities = analysis["lost_entities"]
    if lost_entities:
        print(f"\nLOST ENTITIES (showing first 10):")
        for entity in lost_entities[:10]:
            print(f"  - {entity['name']} ({entity['type']}) from page {entity['page']}")
            print(f"    Reason: {entity['likely_reason']}")
            if entity['details']:
                print(f"    Details: {entity['details']}")
    
    # Unexpected merges
    unexpected = analysis["unexpected_merges"]
    if unexpected:
        print(f"\nUNEXPECTED MERGES:")
        for merge in unexpected[:5]:
            print(f"  - {merge['name']} ({merge['type']})")
            print(f"    {merge['original_count']} entities merged into {merge['final_count']}")
            print(f"    Differences: {merge['differences_found']}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    for i, rec in enumerate(analysis["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    print("=" * 60)