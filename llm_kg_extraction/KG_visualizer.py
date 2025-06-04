import json # Ensure json is imported
from typing import List, Dict, Any, Optional # Ensure these are imported
# from pyvis.network import Network # This is imported inside the method

class KnowledgeGraphVisualizer:
    """
    A class to visualize a financial knowledge graph.
    The interactive visualization is enhanced to handle provenance information correctly for labels.
    """

    def visualize(self, kg_data: dict):
        import networkx as nx
        import matplotlib.pyplot as plt
        # ... (Simplified static visualization - this will NOT show provenance details well)
        G = nx.DiGraph()
        node_labels = {}
        for e in kg_data.get("entities", []):
            primary_name = e.get("name", e.get("id", "Unknown"))
            # For static view, if name is a provenance list, take the first value
            if isinstance(primary_name, list) and primary_name and isinstance(primary_name[0], dict) and "value" in primary_name[0]:
                label_val = primary_name[0]["value"]
                if isinstance(label_val, list) and label_val: # Handle nested list in value e.g. alias
                    node_labels[e["id"]] = str(label_val[0])[:30]
                else:
                    node_labels[e["id"]] = str(label_val)[:30]
            else:
                node_labels[e["id"]] = str(primary_name)[:30]
            G.add_node(e["id"], label=node_labels[e["id"]])
        for r in kg_data.get("relationships", []):
            G.add_edge(r["source"], r["target"], label=str(r.get("type", "")).split(":")[-1])
        pos = nx.spring_layout(G, k=0.8, iterations=30)
        plt.figure(figsize=(12,12))
        nx.draw(G, pos, labels=node_labels, with_labels=True, node_size=500, font_size=8)
        try:
            edge_labels_dict = nx.get_edge_attributes(G,'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict, font_size=7)
        except Exception:
            pass # In case of issues with edge labels in very dense graphs for static
        plt.title("KG Overview (Static)")
        plt.show()


def export_interactive_html(self, kg_data: dict, output_path: str):
        from pyvis.network import Network # Keep this import here
        net = Network(height="800px", width="100%", directed=True, notebook=False, cdn_resources='remote')
        
        type_to_color = {
            "pekg:Company": "#1f77b4", "pekg:LegalEntity": "#aec7e8", "pekg:Client": "#9edae5",
            "pekg:GovernmentBody": "#10ac1a", "pekg:Person": "#2ca02c", "pekg:Position": "#98df8a",
            "pekg:Shareholder": "#d62728", "pekg:FinancialValue": "#ffbf7f", 
            "pekg:FinancialMetric": "#ff7f0e", "pekg:OperationalKPI": "#ffbb78",
            "pekg:Headcount": "#fdd0a2", "pekg:RevenueStream": "#ffd700",
            "pekg:ProductOrService": "#9467bd", "pekg:Technology": "#8c564b",
            "pekg:MarketContext": "#e377c2", "pekg:MarketMetric": "#f7b6d2",
            "pekg:UseCaseOrIndustry": "#ce6dbd", "pekg:TransactionContext": "#d62728",
            "pekg:HistoricalEvent": "#e7969c", "pekg:Historicalevent": "#e7969c", 
            "pekg:Advisor": "#7f7f7f", "pekg:Location": "#c7c7c7",
            "default": "#cccccc"
        }
        
        id_to_clean_label_map = {}
        for e_idx, e in enumerate(kg_data.get("entities", [])): # e is an entity dictionary
            entity_id = e.get("id")
            if not entity_id:
                print(f"Warning: Entity at index {e_idx} is missing an ID. Skipping for label map.")
                continue

            clean_label_str = None
            # Priority fields for entity label
            label_fields_priority = [
                "name", "fullName", "metricName", "kpiName", "productName", 
                "titleName", "shareholderName", "segmentName", "contextName", 
                "eventName", "locationName", "alias",
                "metricValue", "headcountValue", "kpiValueString", "valueString"
            ]

            for field_name in label_fields_priority:
                attribute_package = e.get(field_name)
                candidate_label = None

                if isinstance(attribute_package, list) and attribute_package:
                    # It's a list, assume provenance structure [ {"value": V, "source_doc_id": S}, ... ]
                    first_provenance_entry = attribute_package[0]
                    if isinstance(first_provenance_entry, dict) and "value" in first_provenance_entry:
                        actual_value = first_provenance_entry.get("value")
                        
                        if isinstance(actual_value, list) and actual_value:
                            # The 'value' itself is a list (e.g., for 'alias': "value": ["SSA", "SSAI"])
                            candidate_label = str(actual_value[0]) # Take the first item of the inner list
                        elif actual_value is not None:
                            # The 'value' is a primitive (string, number, bool)
                            candidate_label = str(actual_value)
                elif attribute_package is not None: 
                    # Attribute is not a list, treat as a direct simple value
                    # This handles KGs that haven't been through inter-document merge OR
                    # attributes that are not provenance-tracked.
                    candidate_label = str(attribute_package)
                
                if candidate_label is not None and candidate_label.strip():
                    clean_label_str = candidate_label
                    break # Found a suitable label
            
            if not (clean_label_str and clean_label_str.strip()):
                clean_label_str = entity_id # Fallback to ID
            
            # --- Append fiscalPeriod to FinancialMetric type labels ---
            # First, determine the clean entity type string
            raw_entity_type_pkg = e.get("type")
            entity_type_for_logic = None
            if isinstance(raw_entity_type_pkg, list) and raw_entity_type_pkg and \
               isinstance(raw_entity_type_pkg[0], dict) and "value" in raw_entity_type_pkg[0]:
                 entity_type_for_logic = str(raw_entity_type_pkg[0].get("value"))
            elif isinstance(raw_entity_type_pkg, str):
                 entity_type_for_logic = raw_entity_type_pkg
            
            if entity_type_for_logic == "pekg:FinancialMetric":
                fiscal_period_pkg = e.get("fiscalPeriod")
                fiscal_period_val_str = None
                if isinstance(fiscal_period_pkg, list) and fiscal_period_pkg and \
                   isinstance(fiscal_period_pkg[0], dict) and "value" in fiscal_period_pkg[0]:
                    fiscal_period_val_str = str(fiscal_period_pkg[0].get("value"))
                elif isinstance(fiscal_period_pkg, (str, int, float)): # Handle direct primitive fiscalPeriod
                    fiscal_period_val_str = str(fiscal_period_pkg)

                if fiscal_period_val_str and fiscal_period_val_str.strip():
                    clean_label_str += f" ({fiscal_period_val_str})"
            
            id_to_clean_label_map[entity_id] = clean_label_str[:50] # Truncate

        # --- Node and Edge Creation Loop ---
        for entity in kg_data.get("entities", []):
            entity_id = entity.get("id")
            raw_entity_type = entity.get("type", "UnknownType")
            
            entity_type_display_str = None # For coloring and tooltip
            if isinstance(raw_entity_type, list) and raw_entity_type and \
               isinstance(raw_entity_type[0], dict) and "value" in raw_entity_type[0]:
                entity_type_display_str = str(raw_entity_type[0].get("value"))
            elif isinstance(raw_entity_type, str):
                 entity_type_display_str = raw_entity_type
            else:
                entity_type_display_str = str(raw_entity_type) if raw_entity_type is not None else "UnknownType"

            if not entity_id:
                print(f"Skipping entity due to missing id (node creation): {entity}")
                continue

            tooltip_parts = [f"<b>ID:</b> {entity_id}", f"<b>Type:</b> {entity_type_display_str}"]
            source_docs = entity.get("_source_document_ids")
            if isinstance(source_docs, list) and source_docs:
                tooltip_parts.append(f"<b>Sources:</b> {', '.join(map(str, source_docs))}")

            for k, v_attr in entity.items():
                if k not in {"id", "type", "_source_document_ids"}: # Already handled or internal
                    if isinstance(v_attr, list) and v_attr and \
                       isinstance(v_attr[0], dict) and "value" in v_attr[0] and "source_doc_id" in v_attr[0]:
                        # This is a provenance-aware attribute list
                        value_strings = []
                        for item_entry in v_attr:
                            # Ensure value is string for tooltip
                            item_val_str = str(item_entry.get('value', 'N/A'))
                            item_src_str = str(item_entry.get('source_doc_id', 'N/A'))
                            value_strings.append(f"{item_val_str} (<i>src: {item_src_str}</i>)")
                        tooltip_parts.append(f"<b>{k}:</b> {'; '.join(value_strings)}")
                    else:
                        # Simple attribute or unknown list structure
                        tooltip_parts.append(f"<b>{k}:</b> {v_attr}")
            tooltip = "<br>".join(tooltip_parts)
            
            color = type_to_color.get(entity_type_display_str, type_to_color["default"])
            display_label = id_to_clean_label_map.get(entity_id, entity_id) # Get the processed clean label

            net.add_node(entity_id, label=display_label, title=tooltip, color=color, shape="dot", size=15)

        # Relationship handling (remains largely the same)
        for rel in kg_data.get("relationships", []):
            # ... (as in your previous correct version)
            source_id = rel.get("source")
            target_id = rel.get("target")
            rel_type_full = str(rel.get("type", "relatedTo")) 

            if not source_id or not target_id or not rel_type_full:
                print(f"Skipping relationship due to missing source, target, or type: {rel}")
                continue
            
            relation_label_short = rel_type_full.split(":")[-1]
            edge_tooltip_parts = [f"<b>Type:</b> {rel_type_full}"]
            rel_source_docs = rel.get("_source_document_ids")
            if isinstance(rel_source_docs, list) and rel_source_docs:
                edge_tooltip_parts.append(f"<b>Sources:</b> {', '.join(map(str, rel_source_docs))}")
            
            for k_rel, v_rel in rel.items():
                if k_rel not in {"source", "target", "type", "_source_document_ids"}:
                    edge_tooltip_parts.append(f"<b>{k_rel}:</b> {v_rel}")
            edge_title = "<br>".join(edge_tooltip_parts)
            net.add_edge(source_id, target_id, label=relation_label_short, title=edge_title)


        net.set_options("""
        var options = {
            "nodes": {
                "font": {"size": 12, "face": "Tahoma"}
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
                "color": {"inherit": "from", "opacity": 0.7},
                "smooth": {"type": "continuous", "roundness": 0.15},
                "font": {"size": 9, "align": "middle"}
            },
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -20000,
                    "centralGravity": 0.1,
                    "springLength": 200,
                    "springConstant": 0.05,
                    "damping": 0.09
                },
                "solver": "barnesHut",
                "minVelocity": 0.75
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "multiselect": true,
                "navigationButtons": true,
                "keyboard": true
            },
            "layout": {
                "hierarchical": false 
            }
        }
        """)
        try:
            net.write_html(output_path)
        except Exception as e:
            print(f"Error writing HTML file for graph: {e}")