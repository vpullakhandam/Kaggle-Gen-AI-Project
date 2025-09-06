# Psychology RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system built for the CASML Generative AI Hackathon. This project demonstrates two distinct approaches to building intelligent question-answering systems for academic content, specifically designed to answer psychology questions from a 753-page textbook with accurate citations.

## Features

- **Semantic Document Search**: Advanced text chunking and vector similarity search across academic content
- **Precise Citation System**: Automatic page and section referencing with custom mapping
- **Academic Answer Generation**: 200-400 word responses in scholarly format
- **Two Implementation Approaches**: Local models vs cloud-based API comparison
- **Competition Ready**: CSV output format for automated evaluation
- **Google Colab Support**: Cloud-ready notebooks for easy deployment

## Technologies Used

**Solution 1 (Local Models)**:

- TinyLlama 1.1B Chat model
- LangChain framework
- ChromaDB vector database
- HuggingFace embeddings
- Pydantic structured outputs

**Solution 2 (Cloud API)**:

- Not for Competition
- Google Gemini 2.5 Flash API
- Advanced retrieval strategies
- Two-stage candidate filtering
- Same processing pipeline for fair comparison

## Competition Results

**Solution 1 Performance**:

- Private Score: 0.322
- Public Score: 0.323
- Fully local implementation with no API dependencies

## Getting Started

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or local environment
- API key for Solution 2 (Gemini)

### Quick Start

1. **Clone the repository**
2. **Choose your approach**:
   - [Solution 1: Local Models →](Code%20Files/Solution%201/)
   - [Solution 2: Gemini API →](Code%20Files/Solution%202/)
3. **Follow individual setup guides in each solution folder**

### Project Structure

```
├── Code Files/
│   ├── Solution 1/          # Local RAG implementation
│   │   ├── Solution 1.ipynb
│   │   ├── submission1.csv  # Competition results
│   │   └── helper/page_to_section.json
│   └── Solution 2/          # Cloud API implementation
│       ├── Solution 2.ipynb
│       └── helper/page_to_section.json
├── Sources/
│   ├── book.pdf            # Psychology textbook (753 pages)
│   └── queries.json        # 50 test questions
└── README.md
```

## Data Sources

- **Psychology Textbook**: 753-page comprehensive academic textbook
- **Question Dataset**: 50 curated psychology questions across all major topics
- **Section Mapping**: Custom JSON mapping for precise academic citations

## Documentation

Each solution includes comprehensive documentation:

- Detailed setup instructions
- Technical architecture explanations
- Performance analysis and comparisons

## Acknowledgments

- [CASML Generative AI Hackathon](https://www.kaggle.com/competitions/casml-generative-ai-hackathon/overview)
- HuggingFace for open-source models and tools
- Google for Gemini API access
- LangChain community for RAG framework
