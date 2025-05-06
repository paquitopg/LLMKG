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
            net.add_node(entity["id"], label=id_to_label[entity["id"]], title=tooltip)

        for rel in kg_data.get("relationships", []):
            relation = rel["type"].split(":")[-1]  
            net.add_edge(rel["source"], rel["target"], label=relation)


        net.set_options("""var options = {
            "nodes": {
              "shape": "dot",
              "size": 16,
              "font": {"size": 14}
            },
            "edges": {
              "arrows": {"to": {"enabled": true}},
              "font": {"align": "middle"}
            },
            "interaction": {
              "hover": true,
              "tooltipDelay": 200
            }
          }""")

        net.write_html(output_path)
        print(f"✅ Graph saved to: {output_path}")
