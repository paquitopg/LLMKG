# LLMKG - LLM Knowledge Graph Extraction Framework

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/paquitopg/LLMKG)](https://github.com/paquitopg/LLMKG)

A comprehensive framework for extracting structured knowledge graphs from PDF documents using Large Language Models (LLMs). The system supports both text-based and multimodal extraction, with sophisticated entity merging and ontology-aware processing.

## 🚀 Key Features

- **Multi-Modal Extraction**: Extract knowledge from both text and visual content in PDFs
- **Ontology-Driven**: Uses PEKG (Private Equity Knowledge Graph) ontology with 17+ entity types and rich relationships
- **Intelligent Document Classification**: Automatic document type identification with contextual summarization
- **Advanced Entity Merging**: PEKG-aware page-level and inter-document merging strategies
- **Multi-Document Processing**: Process entire project folders with intelligent cross-document entity resolution
- **Multiple LLM Providers**: Support for Azure OpenAI and Google Vertex AI
- **Interactive Visualizations**: Rich HTML visualizations with provenance tracking
- **Evaluation Framework**: Comprehensive KG quality assessment against ontology compliance

## 📁 Project Structure

```
llm_kg_extraction/
├── _1_document_ingestion/          # PDF processing and document classification
│   ├── pdf_parser.py              # PDF text/image extraction with PyMuPDF
│   └── document_classifier.py     # LLM-based document classification
├── _2_context_understanding/       # Document context preparation
│   └── document_context_preparer.py
├── _3_knowledge_extraction/        # Core KG extraction logic
│   ├── page_llm_processor.py      # Page-level entity/relation extraction
│   └── kg_constructor_single_doc.py # Document-level KG construction
├── _4_knowledge_graph_operations/  # KG merging and post-processing
│   ├── page_level_merger.py       # PEKG-aware entity merging
│   ├── inter_document_merger.py   # Multi-document KG consolidation
│   └── common_kg_utils.py         # Entity similarity and utilities
├── llm_integrations/               # LLM provider wrappers
│   ├── azure_llm.py              # Azure OpenAI integration
│   ├── vertex_llm.py             # Google Vertex AI integration
│   └── base_llm_wrapper.py       # Abstract LLM interface
├── ontology_management/            # Ontology handling
│   ├── ontology_loader.py         # PEKG ontology processing
│   └── pekg_ontology_teasers.yaml # PEKG ontology definition
├── visualization_tools/            # KG visualization
│   └── KG_visualizer.py           # Interactive HTML visualizations
├── core_components/                # Utilities
│   └── document_scanner.py        # PDF file discovery
├── tests/                          # Testing and diagnostics
│   └── merger_diagnostic_tool.py  # Entity merging analysis
├── main_orchestrator.py           # Main pipeline orchestrator
├── KG_evaluator.py                # KG quality evaluation
└── transform_json.py              # KG transformation utilities
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Azure OpenAI or Google Vertex AI access

### Install Dependencies

```bash
# Core dependencies
pip install pypdf pymupdf pillow pyyaml python-dotenv

# Visualization and graph processing
pip install networkx matplotlib pyvis

# LLM integrations
pip install google-generativeai openai

# Additional utilities
pip install pathlib typing-extensions
```

### Environment Setup

Create a `.env` file with your LLM provider credentials:

```env
# For Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_DEPLOYMENT_NAME_GPT4O=your_gpt4o_deployment_name

# For Google Vertex AI
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
GOOGLE_CLOUD_LOCATION=us-central1
```

## 🎯 Quick Start

### Single Document Processing (Vertex AI)

```bash
python main_orchestrator.py \
    /path/to/document.pdf \
    "MyProject" \
    vertexai \
    --main_model_name "gemini-1.5-pro" \
    --classification_model_name "gemini-1.5-flash" \
    --summary_model_name "gemini-1.5-flash" \
    --construction_mode parallel \
    --extraction_mode multimodal \
    --max_workers 4
```

### Multi-Document Processing (Azure OpenAI)

```bash
python main_orchestrator.py \
    /path/to/project/folder \
    "CompanyAnalysis" \
    azure \
    --main_model_name "gpt-4o" \
    --main_azure_model_env_suffix "GPT4O" \
    --classification_model_name "gpt-4o-mini" \
    --classification_azure_model_env_suffix "GPT4OMINI" \
    --summary_model_name "gpt-4o-mini" \
    --summary_azure_model_env_suffix "GPT4OMINI" \
    --construction_mode iterative \
    --extraction_mode text \
    --dump_page_kgs \
    --transform_final_kg
```

## 🧠 PEKG Ontology

The system uses a comprehensive Private Equity Knowledge Graph ontology with:

### Entity Types
- **Companies & Organizations**: Company, GovernmentBody, Advisor
- **People & Roles**: Person, Position, Shareholder
- **Financial Data**: FinancialMetric, OperationalKPI, Headcount
- **Products & Technology**: ProductOrService, Technology
- **Market Context**: MarketContext, UseCaseOrIndustry, Location
- **Transactions**: TransactionContext, HistoricalEvent

### Key Relationships
- `pekg:employs`, `pekg:hasShareholder`, `pekg:reportsFinancialMetric`
- `pekg:offers`, `pekg:operatesIn`, `pekg:advisedBy`
- `pekg:experiencedEvent`, `pekg:hasOfficeIn`

## ⚙️ Configuration Options

### Construction Modes
- **iterative**: Sequential page processing with context building
- **parallel**: Concurrent page processing for speed

### Extraction Modes
- **text**: Text-only extraction from PDFs
- **multimodal**: Combined text and visual content analysis

### Document Types
- `financial_teaser`, `financial_report`, `legal_contract`, `technical_documentation`

## 📊 Output Formats

### Standard KG Format
```json
{
  "entities": [
    {
      "id": "e1",
      "type": "pekg:Company",
      "name": "TechCorp",
      "industry": "Software",
      "foundedYear": 2020
    }
  ],
  "relationships": [
    {
      "source": "e1",
      "target": "e2",
      "type": "pekg:reportsMetric"
    }
  ]
}
```

### Multi-Document Format (with Provenance)
```json
{
  "entities": [
    {
      "id": "e1",
      "type": "pekg:Company",
      "name": [
        {"value": "TechCorp", "source_doc_id": "doc1"},
        {"value": "TechCorp Inc.", "source_doc_id": "doc2"}
      ],
      "_source_document_ids": ["doc1", "doc2"]
    }
  ]
}
```

## 🔧 Entity Merging Strategies

The system implements ontology-aware merging with different strategies per entity type:

- **Preserve Strategy**: Context entities (TransactionContext) - rarely merged
- **Strict Strategy**: Financial metrics - exact temporal/scope matching required
- **Liberal Strategy**: Reference entities (Company, Person) - broader similarity matching
- **Moderate Strategy**: Business entities - balanced approach

## 📈 Evaluation & Diagnostics

### KG Quality Evaluation
```bash
python KG_evaluator.py /path/to/kg.json /path/to/ontology.yaml /path/to/output.json
```

Evaluates:
- Ontology compliance (entity/relation type validity)
- Domain/range constraint adherence
- Graph connectivity and quality metrics
- Overall scoring (0-1 scale)

### Merger Diagnostics
```python
from tests.merger_diagnostic_tool import diagnose_merger_issues
diagnose_merger_issues(page_kgs, final_kg)
```

Provides detailed analysis of:
- Entity loss patterns by type
- Unexpected merges
- Similarity threshold recommendations

## 🎨 Visualization

The system generates interactive HTML visualizations with:

- Entity type-based color coding
- Hover tooltips with detailed entity information
- Relationship labels and directionality
- Multi-document provenance tracking
- Physics-based layout with clustering

## 🛠️ Troubleshooting

### Common Issues

- **Empty KG Output**: Check document classification results and LLM response parsing
- **Entity Over-merging**: Adjust similarity thresholds in merger configuration
- **Memory Issues**: Reduce `max_workers` for large documents
- **Visualization Errors**: Ensure PyVis dependencies are installed correctly

### Debug Mode

Enable detailed logging:
```bash
export KG_DEBUG=1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions and support, please open an issue on the GitHub repository.

---
