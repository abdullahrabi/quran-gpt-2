#!/usr/bin/env python3
"""
Query script to test semantic search using the embeddings stored in Pinecone.
Run this after successfully ingesting documents.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentQuery:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Google Generative AI embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize Pinecone
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "document-embeddings")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "default")
        
        # Get the index
        self.index = self.pinecone.Index(self.index_name)
        
        logger.info(f"Connected to Pinecone index: {self.index_name}")
    
    def search(self, query: str, top_k: int = 5):
        """
        Search for similar documents using semantic similarity
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            logger.info(f"Searching for: '{query}'")
            
            # Create embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True
            )
            
            logger.info(f"Found {len(results.matches)} results")
            return results.matches
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def print_results(self, results, query):
        """Print search results in a formatted way"""
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 60)
        
        if not results:
            print("No results found.")
            return
        
        for i, match in enumerate(results, 1):
            print(f"\n📄 Result {i} (Score: {match.score:.4f})")
            print(f"📁 Source: {match.metadata.get('source', 'Unknown')}")
            print(f"📝 File Type: {match.metadata.get('file_type', 'Unknown')}")
            print(f"🔢 Chunk ID: {match.metadata.get('chunk_id', 'Unknown')}")
            
            # Show a preview of the text
            text = match.metadata.get('text', '')
            if text:
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"📖 Preview: {preview}")
            
            print("-" * 40)
    
    def interactive_search(self):
        """Run interactive search session"""
        print("🚀 Interactive Document Search")
        print("Type 'quit' to exit")
        print("=" * 40)
        
        while True:
            try:
                query = input("\n🔍 Enter your search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    print("⚠️  Please enter a search query.")
                    continue
                
                # Get number of results
                try:
                    top_k = input("📊 Number of results (default 5): ").strip()
                    top_k = int(top_k) if top_k else 5
                except ValueError:
                    top_k = 5
                
                # Perform search
                results = self.search(query, top_k)
                self.print_results(results, query)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    try:
        query_tool = DocumentQuery()
        
        # Check if user wants interactive mode
        if len(os.sys.argv) > 1:
            # Command line search
            query = " ".join(os.sys.argv[1:])
            results = query_tool.search(query)
            query_tool.print_results(results, query)
        else:
            # Interactive mode
            query_tool.interactive_search()
            
    except Exception as e:
        logger.error(f"Failed to initialize query tool: {e}")
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("1. Set up your .env file with API keys")
        print("2. Successfully ingested documents using ingest.py")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
