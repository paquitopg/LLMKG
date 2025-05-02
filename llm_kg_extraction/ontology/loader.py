import yaml

class PEKGOntology:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        self.entities = data.get("entities", [])
        self.relations = data.get("relations", [])
        self.attributes = data.get("attributes", {})

    def format_for_prompt(self, include_attrs=True) -> str:
        entity_block = "\n".join(f"- {e}" for e in self.entities)
        relation_block = "\n".join(f"- {r}" for r in self.relations)

        prompt = f"""
        ### ONTOLOGY ###

        Entities:
        {entity_block}

        Relations:
        {relation_block}
        """
        if include_attrs:
            attr_lines = []
            for cls, props in self.attributes.items():
                attr_lines.append(f"- {cls}: {', '.join(props)}")
            prompt += f"\nAttributes:\n" + "\n".join(attr_lines)

        return prompt.strip()
