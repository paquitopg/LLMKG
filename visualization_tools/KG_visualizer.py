class KnowledgeGraphVisualizer:
    """
    A class to visualize financial knowledge graphs using PyVis and NetworkX.
    Supports both single-document and multi-document knowledge graphs with provenance.
    """

    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.type_to_color = {
            # Core Business & Structure (Blues)
            "pekg:Company": "#1f77b4",      # Muted Blue
            "pekg:LegalEntity": "#aec7e8",  # Light Blue
            "pekg:Client": "#9edae5",       # Pale Cyan/Blue
            "pekg:GovernmentBody": "#10ac1a",# Bright Green

            # People & Roles (Greens)
            "pekg:Person": "#2ca02c",       # Muted Green
            "pekg:Position": "#98df8a",     # Light Green
            "pekg:Shareholder": "#d62728",  # Muted Red
            
            # Financials & Metrics (Oranges/Yellows)
            "pekg:FinancialValue": "#ffbf7f",   
            "pekg:FinancialMetric": "#ff7f0e",  # Orange
            "pekg:OperationalKPI": "#ffbb78",   
            "pekg:Headcount": "#fdd0a2",        
            "pekg:RevenueStream": "#ffd700",    # Gold/Yellow

            # Products, Market, Technology (Purples/Pinks/Browns)
            "pekg:ProductOrService": "#9467bd", # Muted Purple
            "pekg:Technology": "#8c564b",       # Brown
            "pekg:MarketContext": "#e377c2",    # Pink
            "pekg:MarketMetric": "#f7b6d2",     
            "pekg:UseCaseOrIndustry": "#ce6dbd",
            
            # Transactions & Events (Reds/Magentas)
            "pekg:TransactionContext": "#d62728", # Muted Red
            "pekg:HistoricalEvent": "#e7969c",    

            # Supporting & Contextual (Greys/Other)
            "pekg:Advisor": "#7f7f7f",      # Medium Grey
            "pekg:Location": "#c7c7c7",     # Light Grey
            
            "default": "#cccccc" # Default for any unmapped types
        }

    def _extract_attribute_value(self, attribute, prefer_source_doc=None):
        """
        Extract attribute value from either single-document or multi-document format.
        
        Args:
            attribute: Can be a simple value, or a list of {"value": ..., "source_doc_id": ...}
            prefer_source_doc: If specified, prefer value from this source document
            
        Returns:
            Extracted value as string, or None if not found
        """
        if attribute is None:
            return None
        
        # Single document format - simple value
        if not isinstance(attribute, list):
            return str(attribute) if attribute else None
        
        # Multi-document format - list of provenance dictionaries
        if not attribute:  # Empty list
            return None
        
        # If we have a preferred source document, try to find it first
        if prefer_source_doc:
            for item in attribute:
                if isinstance(item, dict) and item.get("source_doc_id") == prefer_source_doc:
                    value = item.get("value")
                    return str(value) if value is not None else None
        
        # Otherwise, take the first available value
        for item in attribute:
            if isinstance(item, dict) and "value" in item:
                value = item.get("value")
                return str(value) if value is not None else None
        
        # Fallback: if it's a list but not in expected format, join values
        return str(attribute[0]) if attribute else None

    def _get_all_source_documents(self, entity):
        """
        Get all source documents for an entity.
        
        Returns:
            List of source document IDs
        """
        # Check for _source_document_ids (inter-document merger format)
        source_docs = entity.get("_source_document_ids", [])
        if source_docs:
            return source_docs
        
        # Check for _source_documents (alternative format)
        source_docs = entity.get("_source_documents", [])
        if source_docs:
            return source_docs
        
        # Check for _source_document (single document format)
        source_doc = entity.get("_source_document")
        if source_doc:
            return [source_doc]
        
        # Fallback: extract from provenance attributes
        source_docs = set()
        for key, value in entity.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "source_doc_id" in item:
                        source_docs.add(item["source_doc_id"])
        
        return list(source_docs)

    def _create_entity_label(self, entity, prefer_source_doc=None):
        """
        Create a meaningful label for an entity, handling both single and multi-document formats.
        """
        entity_type = entity.get("type", "")
        
        # Define priority fields for different entity types
        type_to_name_fields = {
            "pekg:FinancialMetric": ["metricName", "name"],
            "pekg:OperationalKPI": ["kpiName", "name"],
            "pekg:Headcount": ["headcountName", "name"],
            "pekg:TransactionContext": ["contextName", "name"],
            "pekg:Person": ["fullName", "name"],
            "pekg:Position": ["titleName", "name"],
            "pekg:Location": ["locationName", "name"],
            "pekg:Shareholder": ["shareholderName", "name"],
            "pekg:HistoricalEvent": ["eventName", "name"],
            "pekg:MarketContext": ["segmentName", "name"],
            "pekg:ProductOrService": ["productName", "name"],
            "pekg:Company": ["name"],
            "pekg:Advisor": ["name"]
        }
        
        # Get priority fields for this entity type
        name_fields = type_to_name_fields.get(entity_type, ["name"])
        
        # Try each name field in order
        for field in name_fields:
            value = self._extract_attribute_value(entity.get(field), prefer_source_doc)
            if value:
                label = value
                break
        else:
            # Fallback to value fields if no name found
            value_fields = ["valueString", "kpiValueString", "metricValue", "headcountValue"]
            for field in value_fields:
                value = self._extract_attribute_value(entity.get(field), prefer_source_doc)
                if value:
                    label = value
                    break
            else:
                # Final fallback to entity ID
                label = entity.get("id", "Unknown")
        
        # Add period/date info for metrics if available
        if entity_type in ["pekg:FinancialMetric", "pekg:OperationalKPI", "pekg:Headcount"]:
            period_fields = ["DateOrPeriod", "kpiDateOrPeriod", "dateOrYear", "fiscalPeriod"]
            for field in period_fields:
                period = self._extract_attribute_value(entity.get(field), prefer_source_doc)
                if period:
                    label += f" ({period})"
                    break
        
        return str(label)[:50]  # Truncate long labels

    def _create_entity_tooltip(self, entity, is_multi_doc=False):
        """
        Create a detailed tooltip for an entity.
        """
        entity_id = entity.get("id", "Unknown")
        entity_type = entity.get("type", "Unknown")
        
        tooltip_parts = [f"<b>ID:</b> {entity_id}", f"<b>Type:</b> {entity_type}"]
        
        if is_multi_doc:
            # Show source documents
            source_docs = self._get_all_source_documents(entity)
            if source_docs:
                tooltip_parts.append(f"<b>Sources:</b> {', '.join(source_docs)}")
        
        # Add key attributes
        key_attributes = ["name", "fullName", "metricName", "kpiName", "contextName", 
                         "valueString", "metricValue", "description"]
        
        for attr in key_attributes:
            if attr in entity:
                value = self._extract_attribute_value(entity.get(attr))
                if value and len(value) < 100:  # Don't show very long values in tooltip
                    tooltip_parts.append(f"<b>{attr}:</b> {value}")
        
        # For multi-doc, show provenance for key fields
        if is_multi_doc and len(source_docs) > 1:
            tooltip_parts.append("<b>Provenance:</b>")
            for attr in ["name", "metricName", "valueString"]:
                if attr in entity and isinstance(entity[attr], list):
                    for item in entity[attr]:
                        if isinstance(item, dict) and "value" in item and "source_doc_id" in item:
                            tooltip_parts.append(f"  {attr}: {item['value']} (from {item['source_doc_id']})")
        
        return "<br>".join(tooltip_parts)

    def _detect_multi_document_format(self, kg_data):
        """
        Detect if the KG uses multi-document format with provenance.
        """
        entities = kg_data.get("entities", [])
        if not entities:
            return False
        
        # Check if any entity has provenance-style attributes
        for entity in entities[:5]:  # Check first few entities
            for key, value in entity.items():
                if isinstance(value, list) and value:
                    # Check if list contains provenance dictionaries
                    if isinstance(value[0], dict) and "source_doc_id" in value[0]:
                        return True
            
            # Also check for _source_document_ids
            if entity.get("_source_document_ids") or entity.get("_source_documents"):
                return True
        
        return False

    def export_interactive_html(self, kg_data: dict, output_path: str, 
                              prefer_source_doc: str = None, 
                              show_provenance: bool = True):
        """
        Export the knowledge graph to an interactive HTML file using PyVis.
        
        Args:
            kg_data (dict): The knowledge graph data in JSON format.
            output_path (str): The path to save the HTML file.
            prefer_source_doc (str): For multi-doc KGs, prefer values from this document.
            show_provenance (bool): Whether to show provenance information in tooltips.
        """
        from pyvis.network import Network

        # Detect if this is a multi-document KG
        is_multi_doc = self._detect_multi_document_format(kg_data)
        
        if is_multi_doc:
            print(f"Detected multi-document KG format. Creating enhanced visualization...")
        
        net = Network(height="800px", width="100%", directed=True, notebook=False, cdn_resources='remote')

        # Process entities
        entities = kg_data.get("entities", [])
        valid_entity_ids = set()
        
        for entity in entities:
            entity_id = entity.get("id")
            entity_type = entity.get("type")
            
            if not entity_id or not entity_type:
                print(f"Skipping entity due to missing id or type: {entity}")
                continue
            
            valid_entity_ids.add(entity_id)
            
            # Create label and tooltip
            label = self._create_entity_label(entity, prefer_source_doc)
            tooltip = self._create_entity_tooltip(entity, is_multi_doc and show_provenance)
            
            # Get color
            color = self.type_to_color.get(entity_type, self.type_to_color["default"])
            
            # Adjust node size based on number of source documents (for multi-doc KGs)
            node_size = 15
            if is_multi_doc:
                source_docs = self._get_all_source_documents(entity)
                node_size = min(25, 15 + len(source_docs) * 2)  # Larger nodes for entities from multiple docs
            
            net.add_node(
                entity_id,
                label=label,
                title=tooltip,
                color=color,
                shape="dot",
                size=node_size
            )

        # Process relationships
        relationships = kg_data.get("relationships", [])
        valid_relationships = 0
        
        for rel in relationships:
            source_id = rel.get("source")
            target_id = rel.get("target")
            rel_type_full = rel.get("type")

            if not source_id or not target_id or not rel_type_full:
                print(f"Skipping relationship due to missing source, target, or type: {rel}")
                continue
            
            # Only add relationships between valid entities
            if source_id not in valid_entity_ids or target_id not in valid_entity_ids:
                continue
            
            relation_label = rel_type_full.split(":")[-1]  # Show unprefixed relation type
            
            # Create relationship tooltip
            rel_tooltip = f"<b>Type:</b> {rel_type_full}<br><b>From:</b> {source_id}<br><b>To:</b> {target_id}"
            
            # Add source document info for multi-doc KGs
            if is_multi_doc:
                source_docs = rel.get("_source_documents", rel.get("_source_document_ids", []))
                if source_docs:
                    rel_tooltip += f"<br><b>Sources:</b> {', '.join(source_docs)}"
            
            net.add_edge(source_id, target_id, label=relation_label, title=rel_tooltip)
            valid_relationships += 1

        print(f"Added {len(valid_entity_ids)} entities and {valid_relationships} relationships to visualization")

        # Set visualization options
        net.set_options("""
        var options = {
            "nodes": {
                "shape": "dot",
                "font": {"size": 14, "face": "Tahoma"},
                "borderWidth": 2,
                "shadow": true
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
                "color": {"inherit": "from"},
                "smooth": {"type": "continuous", "roundness": 0.2},
                "font": {"size": 10, "align": "middle"},
                "shadow": true
            },
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -30000,
                    "centralGravity": 0.3,
                    "springLength": 250,
                    "springConstant": 0.04,
                    "damping": 0.09
                },
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
            print(f"✅ Interactive graph saved to: {output_path}")
            if is_multi_doc:
                print(f"   Multi-document visualization with provenance information")
        except Exception as e:
            print(f"Error writing HTML file for graph: {e}")

    def export_multi_document_comparison(self, kg_data: dict, output_path: str):
        """
        Create a specialized visualization for multi-document KGs that highlights
        which entities come from which documents.
        """
        from pyvis.network import Network
        import random

        if not self._detect_multi_document_format(kg_data):
            print("This function is designed for multi-document KGs. Using standard export instead.")
            return self.export_interactive_html(kg_data, output_path)

        print("Creating multi-document comparison visualization...")
        
        net = Network(height="900px", width="100%", directed=True, notebook=False, cdn_resources='remote')

        # Get all source documents
        all_source_docs = set()
        for entity in kg_data.get("entities", []):
            source_docs = self._get_all_source_documents(entity)
            all_source_docs.update(source_docs)
        
        # Create color map for source documents
        doc_colors = {}
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]
        for i, doc in enumerate(sorted(all_source_docs)):
            doc_colors[doc] = colors[i % len(colors)]
        
        print(f"Found {len(all_source_docs)} source documents: {', '.join(all_source_docs)}")

        # Process entities with document-aware styling
        valid_entity_ids = set()
        
        for entity in kg_data.get("entities", []):
            entity_id = entity.get("id")
            entity_type = entity.get("type")
            
            if not entity_id or not entity_type:
                continue
            
            valid_entity_ids.add(entity_id)
            source_docs = self._get_all_source_documents(entity)
            
            # Create label
            label = self._create_entity_label(entity)
            
            # Create enhanced tooltip with document info
            tooltip_parts = [
                f"<b>ID:</b> {entity_id}",
                f"<b>Type:</b> {entity_type}",
                f"<b>Source Documents:</b> {', '.join(source_docs)}"
            ]
            
            # Add key attributes with provenance
            for attr in ["name", "metricName", "valueString"]:
                if attr in entity and isinstance(entity[attr], list):
                    tooltip_parts.append(f"<b>{attr} by document:</b>")
                    for item in entity[attr]:
                        if isinstance(item, dict) and "value" in item and "source_doc_id" in item:
                            tooltip_parts.append(f"  • {item['value']} ({item['source_doc_id']})")
            
            tooltip = "<br>".join(tooltip_parts)
            
            # Choose node color based on source documents
            if len(source_docs) == 1:
                # Single document - use document color
                color = doc_colors.get(source_docs[0], "#CCCCCC")
                border_color = color
            else:
                # Multiple documents - use gradient or special color
                color = "#FFD700"  # Gold for multi-document entities
                border_color = "#FF4500"  # Orange border
            
            # Size based on number of documents
            node_size = 15 + len(source_docs) * 3
            
            net.add_node(
                entity_id,
                label=label,
                title=tooltip,
                color={"background": color, "border": border_color},
                shape="dot",
                size=node_size,
                borderWidth=3 if len(source_docs) > 1 else 1
            )

        # Process relationships
        for rel in kg_data.get("relationships", []):
            source_id = rel.get("source")
            target_id = rel.get("target")
            rel_type_full = rel.get("type")

            if (not source_id or not target_id or not rel_type_full or 
                source_id not in valid_entity_ids or target_id not in valid_entity_ids):
                continue
            
            relation_label = rel_type_full.split(":")[-1]
            
            # Relationship tooltip with document sources
            rel_tooltip = f"<b>Type:</b> {rel_type_full}"
            source_docs = rel.get("_source_documents", rel.get("_source_document_ids", []))
            if source_docs:
                rel_tooltip += f"<br><b>Found in:</b> {', '.join(source_docs)}"
            
            net.add_edge(source_id, target_id, label=relation_label, title=rel_tooltip)

        # Add legend as HTML (avoiding Unicode characters that cause encoding issues)
        legend_html = "<div style='position: fixed; top: 10px; right: 10px; background: white; padding: 10px; border: 1px solid #ccc; border-radius: 5px; z-index: 1000;'>"
        legend_html += "<h4>Document Legend</h4>"
        for doc, color in doc_colors.items():
            # Use colored squares instead of bullet characters to avoid encoding issues
            legend_html += f"<div><span style='background-color: {color}; display: inline-block; width: 12px; height: 12px; margin-right: 8px; border: 1px solid #333;'></span>{doc}</div>"
        legend_html += "<div><span style='background-color: #FFD700; display: inline-block; width: 12px; height: 12px; margin-right: 8px; border: 1px solid #333;'></span>Multi-document entity</div>"
        legend_html += "</div>"

        # Enhanced options for comparison view
        net.set_options("""
        var options = {
            "nodes": {
                "shape": "dot",
                "font": {"size": 12, "face": "Tahoma"},
                "shadow": true
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
                "color": {"inherit": "from"},
                "smooth": {"type": "continuous", "roundness": 0.2},
                "font": {"size": 9, "align": "middle"}
            },
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -20000,
                    "centralGravity": 0.1,
                    "springLength": 200,
                    "springConstant": 0.02,
                    "damping": 0.1
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "multiselect": true,
                "navigationButtons": true
            }
        }
        """)

        try:
            # Write the HTML file and add legend
            net.write_html(output_path)
            
            # Add legend to the HTML file
            with open(output_path, 'r') as f:
                html_content = f.read()
            
            # Insert legend before closing body tag
            html_content = html_content.replace('</body>', f'{legend_html}</body>')
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            print(f"✅ Multi-document comparison visualization saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating multi-document visualization: {e}")

    def visualize(self, kg_data: dict):
        """
        Creates a static visualization of the knowledge graph using Matplotlib and NetworkX.
        Updated to handle both single and multi-document formats.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        
        # Detect format
        is_multi_doc = self._detect_multi_document_format(kg_data)
        
        # Create labels
        id_to_label = {}
        for entity in kg_data.get("entities", []):
            entity_id = entity.get("id")
            if entity_id:
                label = self._create_entity_label(entity)
                id_to_label[entity_id] = label[:30]  # Truncate for static view

        # Add nodes and edges
        for entity in kg_data.get("entities", []):
            entity_id = entity.get("id")
            entity_type = entity.get("type", "UnknownType")
            if entity_id:
                G.add_node(entity_id, label=id_to_label[entity_id], type=entity_type)

        for rel in kg_data.get("relationships", []):
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type", "relatedTo")
            if source and target:
                G.add_edge(source, target, label=rel_type.split(':')[-1])

        # Layout and visualization
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        plt.figure(figsize=(16, 12))
        
        # Node colors
        node_colors = []
        for node_id in G.nodes():
            node_type = G.nodes[node_id].get('type', 'default')
            node_colors.append(self.type_to_color.get(node_type, self.type_to_color['default']))

        nx.draw(G, pos, with_labels=True, labels=id_to_label, 
                node_color=node_colors, node_size=2500, font_size=8, arrowsize=15)
        
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=7)
        
        title = "Knowledge Graph (Static Overview)"
        if is_multi_doc:
            title += " - Multi-Document"
        plt.title(title)
        plt.axis('off')
        plt.show()