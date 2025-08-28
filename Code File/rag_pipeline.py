# 1. Required Imports
import torch
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from FlagEmbedding import FlagReranker
from transformers import AutoModel, AutoTokenizer
import re
import json
from tqdm import tqdm

# 2. Configuration
class Config:
    """Central place for all configuration parameters"""
    CHROMA_PATH = "./chroma"
    DATA_PATH = "./book.pdf"
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100

# 3. Document Processing
def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove boilerplate
    text = re.sub(r'Access for free at openstax\.org\.*', '', text)
    text = re.sub(r'LINK TO LEARNING.*?(?=\n|$)', '', text)
    text = re.sub(r'Watch a brief video.*?(?=\n|$)', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()

def load_and_process_documents():
    """Load PDF and create processed documents"""
    # Load documents
    loader = PyPDFLoader(Config.DATA_PATH)
    documents = loader.load()
    
    # Clean documents
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    
    return chunks

# 4. Vector Store Setup
def setup_vector_store(chunks):
    """Initialize and populate the vector store"""
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': Config.DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Add documents to vector store
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    
    return vectorstore

# 5. Query Processing
class QueryProcessor:
    def __init__(self):
        self.reranker = FlagReranker(
            Config.RERANKER_MODEL,
            use_fp16=True,
            device=Config.DEVICE
        )
    
    def process_query(self, query: str, vectorstore, num_docs: int = 5):
        """Process query and return reranked results"""
        # Get initial results
        results = vectorstore.similarity_search(
            query,
            k=num_docs * 3  # Get more docs for reranking
        )
        
        # Rerank results
        pairs = [[query, doc.page_content] for doc in results]
        scores = self.reranker.compute_score(pairs)
        
        # Sort and return top results
        sorted_results = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:num_docs]

# 6. Main RAG Pipeline
class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.chunks = load_and_process_documents()
        self.vectorstore = setup_vector_store(self.chunks)
        self.query_processor = QueryProcessor()
    
    def answer_query(self, query: str):
        """Process query and generate response"""
        # Get relevant documents
        results = self.query_processor.process_query(
            query,
            self.vectorstore
        )
        
        # Format context
        context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc, score in results
        ])
        
        return {
            'query': query,
            'context': context,
            'results': [
                {
                    'content': doc.page_content,
                    'page': doc.metadata['page'],
                    'score': score
                }
                for doc, score in results
            ]
        }

# 7. Usage Example
def main():
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Example query
    query = "What is psychology?"
    response = rag.answer_query(query)
    
    # Print results
    print(f"\nQuery: {response['query']}")
    print("\nTop Results:")
    for i, result in enumerate(response['results'], 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Page: {result['page']}")
        print(f"   Content: {result['content'][:200]}...")

if __name__ == "__main__":
    main()