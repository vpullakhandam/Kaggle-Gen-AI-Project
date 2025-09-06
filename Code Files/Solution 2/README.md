# Solution 2: Gemini API RAG System

A streamlined cloud-based RAG implementation leveraging Google's Gemini 2.5 Flash API for superior language understanding. This solution demonstrates the performance benefits of state-of-the-art language models while maintaining a clean, readable codebase for academic question answering.

## Features

- **Gemini 2.5 Flash Integration**: State-of-the-art language model with advanced reasoning capabilities
- **Structured Output Generation**: Pydantic schema with automatic JSON validation
- **Streamlined Architecture**: Clean, readable code with single-responsibility cells
- **Same Document Processing**: Uses identical PDF processing pipeline as Solution 1 for fair comparison
- **Academic Citation System**: Automatic page adjustment and section mapping with inline citations
- **Error-Resilient Design**: Robust API integration with graceful error handling

## Technologies Used

**Core Stack**:

- Google Gemini 2.5 Flash API
- LangChain framework for document processing
- ChromaDB vector database for semantic search
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Pydantic for structured output validation

**Processing Pipeline**:

- PyPDFLoader for document loading (identical to Solution 1)
- RecursiveCharacterTextSplitter (800 chars, 100 overlap)
- Custom page-to-section mapping system
- JSON-based structured response generation

## Getting Started

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Stable internet connection

### Installation

```bash
# Install core dependencies
pip install google-generativeai python-dotenv
pip install langchain langchain-community langchain-chroma
pip install langchain-huggingface pypdf chromadb pandas pydantic
```

### API Setup

1. **Get Gemini API Key**: Visit [Google AI Studio](https://ai.google.dev/aistudio)
2. **Create `.env` file** in the solution directory:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

### Running the Solution

```bash
# Navigate to solution directory
cd "Code Files/Solution 2"

# Ensure .env file contains your API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Start Jupyter notebook
jupyter notebook sol2.ipynb

# Execute cells 1-13 in sequence
```

## Architecture Overview

### Document Processing 

```
Psychology Textbook (753 pages)
    ↓ PyPDFLoader (Same as Solution 1)
753 Pages with Metadata
    ↓ RecursiveCharacterTextSplitter (800 chars, 100 overlap)
3,384 Text Chunks
    ↓ HuggingFace Embeddings
ChromaDB Vector Store
```

### Question Answering Pipeline 

```
User Question
    ↓ prepare_context() - Semantic search (k=3)
3 Most Relevant Chunks + Page/Section Data
    ↓ format_context_for_prompt() - Context formatting
Structured Prompt with Citations
    ↓ answer_question() - Gemini API call
JSON Response via QAResponse Schema
```

### Structured Output Schema

```python
class QAResponse(BaseModel):
    answer: str = Field(..., description="Answer in 200-400 words with inline citations")
```

## Output Format

Generates `submission2.csv` with:

```csv
ID,context,answer,references
1,"[Retrieved chunks]","[200-400 word answer]","[Page and section data]"
```

## Performance Characteristics

**Advantages over Local Models**:

- Superior language understanding and reasoning
- Consistent cloud performance regardless of hardware
- Advanced context processing capabilities
- Better handling of complex academic concepts
- Structured JSON output with automatic validation

**Considerations**:

- Requires API access and incurs costs
- Internet dependency for operation
- Data sent to external service
- API rate limiting considerations


## Error Handling

The implementation includes robust error handling:

- API response validation
- JSON parsing fallbacks
- Graceful degradation for missing metadata
- Exception handling with informative error messages

## Cost Optimization

- Efficient prompt design minimizes token usage
- Single API call per question (no retries)
- Structured output reduces parsing overhead
- Optimized context length for API limits

This solution showcases the advantages of modern cloud-based language models for academic applications while maintaining clean, readable code that's easy to understand and modify.
