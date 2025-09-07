import os
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIngester:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Google Generative AI embeddings (Gemini embbeder)
        # Prefer explicit model "gemini-embedding-001" as requested; fallback kept for robustness
        google_api_key = os.getenv("GOOGLE_API_KEY")
        requested_models = [
            "models/gemini-embedding-001",
            "gemini-embedding-001",
            "models/text-embedding-004"  # fallback if above not available
        ]
        last_err: Optional[Exception] = None
        self.embeddings = None
        for m in requested_models:
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=m,
                    google_api_key=google_api_key
                )
                # Trigger a quick check to ensure the model is valid by embedding a tiny string
                _probe = self.embeddings.embed_query("ok")
                if isinstance(_probe, list) and len(_probe) > 0:
                    self.embedding_dimension = len(_probe)
                    logger.info(f"Using embeddings model: {m} (dim={self.embedding_dimension})")
                    break
            except Exception as e:
                last_err = e
                continue
        if self.embeddings is None:
            raise RuntimeError(f"Failed to initialize Google embeddings; last error: {last_err}")
        
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY") or "pcsk_355dUh_DsZRDBRSD2SWhTGZR1SrDnr2b5hCv18b3jGTD5RfKcMbVnTEcqnfEDf9inrahMx"
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.pinecone = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "quarn-gpt-2")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "default")
        
        # Initialize or get Pinecone index
        self.index = self._initialize_pinecone_index()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def _initialize_pinecone_index(self):
        """Initialize or get existing Pinecone index"""
        try:
            # Check if index exists
            if self.index_name not in [index.name for index in self.pinecone.list_indexes()]:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Create new index
                self.pinecone.create_index(
                    name=self.index_name,
                    # Derive dimension dynamically from the active embedding model
                    dimension=getattr(self, "embedding_dimension", 768),
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                # Wait for index to be ready
                while not self.pinecone.describe_index(self.index_name).status["ready"]:
                    import time
                    time.sleep(1)
                    logger.info("Waiting for index to be ready...")
            
            return self.pinecone.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            raise
    
    def load_documents(self, data_folder: str = "data") -> List[Document]:
        """Load documents from the specified folder"""
        documents = []
        data_path = Path(data_folder)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data folder '{data_folder}' not found")
        
        # Supported file types
        supported_extensions = {'.pdf', '.txt', '.md'}
        
        for file_path in data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    logger.info(f"Processing file: {file_path}")
                    
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                        documents.extend(loader.load())
                    elif file_path.suffix.lower() in {'.txt', '.md'}:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            documents.append(Document(
                                page_content=text,
                                metadata={"source": str(file_path), "file_type": file_path.suffix}
                            ))
                    
                    logger.info(f"Successfully loaded: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
                    continue
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        logger.info("Splitting documents into chunks...")
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Documents split into {len(split_docs)} chunks")
        return split_docs
    
    def create_embeddings(self, documents: List[Document]) -> List[tuple]:
        """Create embeddings for documents"""
        logger.info("Creating embeddings...")
        
        embeddings_data = []
        for i, doc in enumerate(documents):
            try:
                # Create embedding
                embedding = self.embeddings.embed_query(doc.page_content)
                
                # Prepare metadata
                metadata = {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "chunk_id": i
                }
                
                embeddings_data.append((embedding, metadata))
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error creating embedding for document {i}: {e}")
                continue
        
        logger.info(f"Successfully created embeddings for {len(embeddings_data)} documents")
        return embeddings_data
    
    def upload_to_pinecone(self, embeddings_data: List[tuple]):
        """Upload embeddings to Pinecone"""
        logger.info("Uploading embeddings to Pinecone...")
        
        try:
            # Prepare vectors for batch upload
            vectors = []
            for i, (embedding, metadata) in enumerate(embeddings_data):
                vector = {
                    "id": f"doc_{i}_{metadata['chunk_id']}",
                    "values": embedding,
                    "metadata": metadata
                }
                vectors.append(vector)
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=self.namespace)
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully uploaded {len(vectors)} vectors to Pinecone")
            
        except Exception as e:
            logger.error(f"Error uploading to Pinecone: {e}")
            raise
    
    def ingest(self, data_folder: str = "data"):
        """Main ingestion pipeline"""
        try:
            logger.info("Starting document ingestion process...")
            
            # Load documents
            documents = self.load_documents(data_folder)
            if not documents:
                logger.warning("No documents found to process")
                return
            
            # Split documents
            split_docs = self.split_documents(documents)
            
            # Create embeddings
            embeddings_data = self.create_embeddings(split_docs)
            
            # Upload to Pinecone
            self.upload_to_pinecone(embeddings_data)
            
            logger.info("Document ingestion completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            raise

def main():
    """Main function to run the ingestion process"""
    try:
        ingester = DocumentIngester()
        ingester.ingest()
    except Exception as e:
        logger.error(f"Failed to run ingestion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
