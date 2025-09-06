# Solution 1: Local Models RAG System

A complete self-contained RAG implementation using open-source models for academic question answering. This solution achieved scores of 0.322 (private) and 0.323 (public) in the CASML competition, demonstrating effective local model deployment for specialized academic content.

## Features

- **Self-Contained Operation**: No internet required after initial setup
- **TinyLlama Integration**: 1.1B parameter chat model optimized for academic responses
- **Semantic Vector Search**: ChromaDB with 384-dimensional embeddings for precise retrieval
- **Intelligent Citation System**: Automatic page adjustment and section mapping
- **Structured Output**: Pydantic models ensure consistent 200-400 word responses
- **Competition Ready**: Generates clean CSV for automated evaluation

## Technologies Used

**Core Stack**:

- TinyLlama 1.1B Chat model
- LangChain framework
- ChromaDB vector database
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Pydantic + Instructor for structured outputs

**Processing Pipeline**:

- PyPDFLoader for document processing
- RecursiveCharacterTextSplitter for intelligent chunking
- Custom page-to-section mapping system
- Unicode text normalization

## Getting Started

### Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- Google Colab (recommended) or local environment

### Installation

```bash
# Install dependencies
pip install transformers accelerate langchain langchain-community langchain-core
pip install langchain-chroma instructor langchain-huggingface pypdf chromadb
pip install pydantic requests torch
```

### Running on Google Colab (Recommended)

1. **Upload files to Colab**:

   - `Solution 1.ipynb`
   - `helper/page_to_section.json`
   - `../../Sources/book.pdf`
   - `../../Sources/queries.json`

2. **Update file paths in notebook**:

   ```python
   PDF_PATH = Path("/content/book.pdf")
   QUERIES_PATH = Path("/content/queries.json")
   PAGE_TO_SECTION_PATH = Path("/content/page_to_section.json")
   ```

3. **Execute all cells** - dependencies install automatically

### Running Locally

```bash
# Navigate to solution directory
cd "Code Files/Solution 1"

# Start Jupyter notebook
jupyter notebook "Solution 1.ipynb"

# Execute all cells in sequence
```

## Architecture Overview

### Document Processing

```
Psychology Textbook (753 pages)
    ↓ PyPDFLoader
753 Individual Pages
    ↓ RecursiveCharacterTextSplitter (800 chars, 100 overlap)
3,384 Text Chunks with Page Metadata
```

### Vector Database

```
Text Chunks
    ↓ HuggingFace Embeddings (all-MiniLM-L6-v2)
384-Dimensional Vectors
    ↓ ChromaDB
Persistent Vector Store ("textbook_chunks")
```

### Answer Generation

```
User Question
    ↓ Semantic Search (k=3)
Top Relevant Chunks
    ↓ Page Adjustment (-12 offset)
Section Mapping
    ↓ TinyLlama Chat Pipeline
Academic Answer with Citations
```

## Key Implementation Details

### Intelligent Page Mapping

- **Page Offset Correction**: Subtracts 12 to account for PDF front matter
- **Section Lookup**: Maps page numbers to hierarchical sections
- **Citation Format**: Inline references like [p. 79] in academic style

### Structured Output Schema

```python
class QAResponse(BaseModel):
    answer: str = Field(..., description="Answer in 200-400 words")
```

### Context Preparation

- Retrieves top-3 most relevant chunks
- Deduplicates and sorts page references
- Maps pages to textbook sections
- Builds academic-style prompts

## Output Format

Generates `submission1.csv` with competition format:

```csv
ID,context,answer,references
1,"[Retrieved chunks]","[200-400 word answer]","[Page and section data]"
```

## Performance Characteristics

**Strengths**:

- Complete privacy and data control
- No ongoing API costs
- Reproducible results
- Works offline after setup

**Considerations**:

- Hardware-dependent performance
- Limited by TinyLlama's 1.1B parameters
- 2048 token context window

## Competition Results

- **Private Score**: 0.322
- **Public Score**: 0.323
- **Approach**: Fully local implementation
- **Output**: Clean CSV with 50 academic responses

This solution demonstrates effective deployment of local language models for specialized academic content while maintaining complete data privacy and operational independence.
