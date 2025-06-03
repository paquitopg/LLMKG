class KnowledgeGraphVisualizer:
    """
    A class to visualize a financial knowledge graph using PyVis and NetworkX.
    It can create both static and interactive visualizations.
    """

    def visualize(self, kg_data: dict):
        """
        Creates a static visualization of the knowledge graph using Matplotlib and NetworkX.
        This is a basic visualization and might be less effective for large graphs.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        
        # Attempt to create a more meaningful label for each node
        id_to_label = {}
        for e in kg_data.get("entities", []):
            label = e.get("name", # General name
                          e.get("fullName", # For Person
                          e.get("titleName", # For Position
                          e.get("productName", # For ProductOrService
                          e.get("metricName", # For FinancialMetric
                          e.get("kpiName", # For OperationalKPI
                          e.get("shareholderName", # For Shareholder
                          e.get("contextName", # For TransactionContext
                          e.get("eventName", # For HistoricalEvent
                          e.get("locationName", # For Location
                          str(e.get("metricValue", # Fallback for metrics
                                    e.get("headcountValue", # Fallback for headcount
                                    e.get("kpiValueString", # Fallback for KPI
                                    e["id"])))))))))))))) # Default to ID
            id_to_label[e["id"]] = str(label)[:30] # Truncate long labels for static view

        for entity in kg_data.get("entities", []):
            G.add_node(entity["id"], label=id_to_label[entity["id"]], type=entity.get("type", "UnknownType"))

        for rel in kg_data.get("relationships", []):
            G.add_edge(rel["source"], rel["target"], label=rel.get("type", "relatedTo").split(':')[-1])

        pos = nx.spring_layout(G, k=0.5, iterations=50) # Adjust layout parameters
        plt.figure(figsize=(16, 12)) # Increased figure size
        
        # Get node colors based on type
        node_colors = []
        # Define a minimal color map for static view if needed, or use a default
        static_type_to_color = {
            "pekg:Company": "skyblue",
            "pekg:Person": "lightgreen",
            "pekg:FinancialMetric": "lightcoral",
            "default": "lightgrey"
        }
        for node_id in G.nodes():
            node_type = G.nodes[node_id].get('type', 'default')
            node_colors.append(static_type_to_color.get(node_type, static_type_to_color['default']))

        nx.draw(G, pos, with_labels=True, labels=id_to_label, 
                node_color=node_colors, node_size=2500, font_size=8, arrowsize=15)
        
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=7)
        
        plt.title("Financial Knowledge Graph (Static Overview)")
        plt.axis('off')
        plt.show()


    def export_interactive_html(self, kg_data: dict, output_path: str):
        """
        Export the knowledge graph to an interactive HTML file using PyVis.
        Args:
            kg_data (dict): The knowledge graph data in JSON format.
            output_path (str): The path to save the HTML file.
        """
        from pyvis.network import Network

        net = Network(height="800px", width="100%", directed=True, notebook=False, cdn_resources='remote')

        # Updated color map for the streamlined ontology v1.2
        type_to_color = {
            # Core Business & Structure (Blues)
            "pekg:Company": "#1f77b4",      # Muted Blue
            "pekg:LegalEntity": "#aec7e8",  # Light Blue
            "pekg:Client": "#9edae5",       # Pale Cyan/Blue
            "pekg:GovernmentBody": "#10ac1a",# Bright Green (for government entities)

            # People & Roles (Greens)
            "pekg:Person": "#2ca02c",       # Muted Green
            "pekg:Position": "#98df8a",     # Light Green
            "pekg:Shareholder": "#d62728",  # Muted Red (distinct for ownership) - Or a Green if preferred with People
            
            # Financials & Metrics (Oranges/Yellows)
            "pekg:FinancialValue": "#ffbf7f",   # Light Orange (for the component, if visualized)
            "pekg:FinancialMetric": "#ff7f0e",  # Orange
            "pekg:OperationalKPI": "#ffbb78",   # Lighter Orange
            "pekg:Headcount": "#fdd0a2",        # Very Light Orange/Peach
            "pekg:RevenueStream": "#ffd700",    # Gold/Yellow

            # Products, Market, Technology (Purples/Pinks/Browns)
            "pekg:ProductOrService": "#9467bd", # Muted Purple
            "pekg:Technology": "#8c564b",       # Brown
            "pekg:MarketContext": "#e377c2",    # Pink
            "pekg:MarketMetric": "#f7b6d2",     # Lighter Pink
            "pekg:UseCaseOrIndustry": "#ce6dbd",# Medium Purple/Pink
            
            # Transactions & Events (Reds/Magentas)
            "pekg:TransactionContext": "#d62728", # Muted Red (same as Shareholder for impact, or choose different)
            "pekg:HistoricalEvent": "#e7969c",    # Desaturated Red/Pink

            # Supporting & Contextual (Greys/Other)
            "pekg:Advisor": "#7f7f7f",      # Medium Grey
            "pekg:Location": "#c7c7c7",     # Light Grey
            
            "default": "#cccccc" # Default for any unmapped types
        }

        # Attempt to create a more meaningful label for each node for PyVis
        id_to_label = {}
        for e in kg_data.get("entities", []):
            # Prioritize specific name fields based on type, then general 'name', then value fields
            entity_type = e.get("type")
            label_content = e.get("name") # Default to 'name'

            if entity_type == "pekg:FinancialMetric":
                label_content = e.get("metricName", e.get("name"))
                label_content += f" ({e.get('fiscalPeriod', '')})" if e.get("fiscalPeriod") else ""
            elif entity_type == "pekg:Headcount":
                label_content = e.get("headcountName", e.get("name"))
            elif entity_type == "pekg:MarketContext":
                label_content = e.get("segmentName", e.get("name"))
            elif entity_type == "pekg:TransactionContext":
                label_content = e.get("contextName", e.get("name"))
            elif entity_type == "pekg:GovernmentBody":
                label_content = e.get("name", e.get("name"))
            elif entity_type == "pekg:OperationalKPI":
                label_content = e.get("kpiName", e.get("name"))
            elif entity_type == "pekg:ProductOrService":
                label_content = e.get("productName", e.get("name")) # Assuming 'productName' might be used
            elif entity_type == "pekg:Person":
                label_content = e.get("fullName", e.get("name"))
            elif entity_type == "pekg:Position":
                label_content = e.get("titleName", e.get("name"))
            elif entity_type == "pekg:Location":
                label_content = e.get("locationName", e.get("name"))
            elif entity_type == "pekg:Shareholder":
                label_content = e.get("shareholderName", e.get("name"))
            elif entity_type == "pekg:HistoricalEvent":
                label_content = e.get("eventName", e.get("name"))
            elif entity_type == "pekg:Historicalevent":
                label_content = e.get("eventName", e.get("name"))
            # Add more specific fallbacks if needed for other types

            # If still no specific name, try common value fields before ID
            if not label_content:
                label_content = e.get("valueString", 
                                  e.get("kpiValueString",
                                  str(e.get("metricValue",
                                            e.get("headcountValue", 
                                                  e["id"]))))) # Default to ID

            id_to_label[e["id"]] = str(label_content)[:50] # Truncate very long labels for display

        for entity in kg_data.get("entities", []):
            entity_id = entity.get("id")
            entity_type = entity.get("type") # This is the prefixed type like "pekg:Company"
            
            if not entity_id or not entity_type:
                print(f"Skipping entity due to missing id or type: {entity}")
                continue

            tooltip_parts = [f"ID: {entity_id}", f"Type: {entity_type}"]
            for k, v in entity.items():
                if k not in {"id", "type"}:
                    tooltip_parts.append(f"{k}: {v}")
            tooltip = "<br>".join(tooltip_parts)
            
            # Use unprefixed type for color lookup if your map uses that, or full type if map uses full
            # Current map uses full prefixed types.
            color = type_to_color.get(entity_type, type_to_color["default"])
            
            label_for_node = id_to_label.get(entity_id, entity_id) # Use processed label

            net.add_node(
                entity_id,
                label=label_for_node,
                title=tooltip,
                color=color,
                shape="dot", # Default shape, can be customized per type
                size=15 # Default size
            )

        for rel in kg_data.get("relationships", []):
            source_id = rel.get("source")
            target_id = rel.get("target")
            rel_type_full = rel.get("type")

            if not source_id or not target_id or not rel_type_full:
                print(f"Skipping relationship due to missing source, target, or type: {rel}")
                continue
            
            # Ensure source and target nodes exist before adding edge (PyVis might handle this, but good practice)
            # This check might be too slow for very large graphs if done here.
            # PyVis usually just won't draw edges to non-existent nodes.

            relation_label = rel_type_full.split(":")[-1] # Show unprefixed relation type
            net.add_edge(source_id, target_id, label=relation_label, title=rel_type_full)

        net.set_options("""
        var options = {
            "nodes": {
            "shape": "dot",
            "size": 18,
            "font": {"size": 14, "face": "Tahoma"}
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
                "color": {"inherit": "from"},
                "smooth": {"type": "continuous", "roundness": 0.2},
                "font": {"size": 10, "align": "middle"}
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
                "multiselect": true
            },
            "layout": {
                "hierarchical": false 
            }
        }
        """)

        try:
            net.write_html(output_path)
            # print(f"✅ Interactive graph saved to: {output_path}") # Moved to _save_page_graph
        except Exception as e:
            print(f"Error writing HTML file for graph: {e}")