class VisualElements:
    """
    A class to represent visual elements in a knowledge graph.
    """

    def __init__(self, element_type):
        """
        Initialize the visual element with its type and properties.

        Args:
            element_type (str): The type of the visual element (e.g., 'table', 'chart').
            properties (dict): A dictionary of properties for the visual element.
        """
        self.element_type = element_type
    
    def construct_prompt_for_analysis(self, ontology_desc):
        """
        Construct a prompt for analyzing the visual element.

        Returns:
            str: The constructed prompt.
        """
        if self.element_type == 'table' :
            prompt = f"""
            You are a financial analyst. Extract the data from this financial table.

            First, extract the raw table structure as a JSON object with headers and rows.
            Then, identify entities and relationships according to this ontology:
            
            {ontology_desc}
            
            Format your response as JSON:
            {{
                "table_data": {{
                    "headers": ["Column1", "Column2", ...],
                    "rows": [
                        ["Value1", "Value2", ...],
                        ...
                    ]
                }},
                "entities": [
                    {{"id": "e1", "type": "pekg:Company", "name": "ABC Capital"}},
                    ...
                ],
                "relationships": [
                    {{"source": "e1", "target": "e2", "type": "pekg:receivedInvestment"}},
                    ...
                ]
            }}
            """
