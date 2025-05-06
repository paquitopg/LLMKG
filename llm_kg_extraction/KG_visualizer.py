class KnowledgeGraphVisualizer:
    """
    A class to visualize a financial knowledge graph using PyVis and NetworkX.
    It can create both static and interactive visualizations.
    """

    def visualize(self, kg_data: dict):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        id_to_label = {
            e["id"]: e.get("name") or
                     str(e.get("metricValue")) or
                     str(e.get("headcountValue")) or
                     e["id"]
            for e in kg_data.get("entities", [])
        }

        for entity in kg_data.get("entities", []):
            G.add_node(entity["id"], label=id_to_label[entity["id"]], type=entity["type"])

        for rel in kg_data.get("relationships", []):
            G.add_edge(rel["source"], rel["target"], label=rel["type"])

        pos = nx.spring_layout(G)
        plt.figure(figsize=(14, 10))
        nx.draw(G, pos, with_labels=True, labels=id_to_label, node_color='skyblue', node_size=2000)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Financial Knowledge Graph")
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

        net = Network(height="750px", width="100%", directed=True)

        type_to_color = {
        # Corporate Events
        "pekg:IPOEvent": "#bc80bd",
        "pekg:MergerEvent": "#bc80bd",
        "pekg:AcquisitionEvent": "#bc80bd",
        "pekg:ExitEvent": "#bc80bd",
        "pekg:LeadershipChangeEvent": "#bc80bd",
        "pekg:CorporateEvent": "#bc80bd",
        "pekg:FundingRound": "#bc80bd",

        # Metrics
        "pekg:FinancialMetric": "#4daf4a",
        "pekg:HeadcountMetric": "#4daf4a",
        "pekg:MarketMetric": "#4daf4a",
        "pekg:TaxMetric": "#4daf4a",
        "pekg:KPI": "#4daf4a",

        # Legal Entities
        "pekg:Company": "#1f78b4",
        "pekg:LegalEntity": "#1f78b4",
        "pekg:Investor": "#1f78b4",
        "pekg:Advisor": "#1f78b4",

        # People/Orgs
        "pekg:Person": "#33a02c",
        "pekg:Department": "#33a02c",
        "pekg:Position": "#33a02c",
        "pekg:Committee": "#33a02c",

        # Governance/Policy
        "pekg:GovernmentBody": "#ff7f00",
        "pekg:RegulatoryRequirement": "#ff7f00",
        "pekg:PolicyDocument": "#ff7f00",

        # Assets & IP
        "pekg:Product": "#6a3d9a",
        "pekg:IntellectualProperty": "#6a3d9a",
        "pekg:Contract": "#6a3d9a",
        "pekg:OwnershipStake": "#6a3d9a",

        # Legal Risk
        "pekg:Risk": "#e31a1c",
        "pekg:Litigation": "#e31a1c",

        # Other
        "pekg:UseCase": "#999999",
        "pekg:NewsItem": "#999999",
        "pekg:Location": "#999999",
        "pekg:FinancialInstrument": "#999999",
        "default": "#cccccc" 
        }

        id_to_label = {
            e["id"]: e.get("name") or
                    str(e.get("metricValue")) or
                    str(e.get("headcountValue")) or
                    e["id"]
            for e in kg_data.get("entities", [])
        }

        for entity in kg_data.get("entities", []):
            tooltip = f"Type: {entity['type']}<br>" + "<br>".join(
                f"{k}: {v}" for k, v in entity.items() if k not in {"id", "type"}
            )
            color = type_to_color.get(entity["type"], type_to_color["default"])
            net.add_node(
                entity["id"],
                label=id_to_label[entity["id"]],
                title=tooltip,
                color=color
            )

        for rel in kg_data.get("relationships", []):
            relation = rel["type"].split(":")[-1]
            net.add_edge(rel["source"], rel["target"], label=relation)

        # Improve layout with physics and hierarchical options
        net.set_options("""var options = {
            "nodes": {
            "shape": "dot",
            "size": 18,
            "font": {"size": 14, "face": "Tahoma"}
            },
            "edges": {
            "arrows": {"to": {"enabled": true}},
            "font": {"align": "middle"}
            },
            "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -30000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09
            },
            "minVelocity": 0.75
            },
            "interaction": {
            "hover": true,
            "tooltipDelay": 200
            }
        }""")

        net.write_html(output_path)
        print(f"✅ Graph saved to: {output_path}")