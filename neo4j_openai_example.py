import os
import asyncio
import dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Load environment variables from .env file
dotenv.load_dotenv('/Users/airray/rayfile/hacktech/backend/.env')

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    # Get API key from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize LightRAG with Neo4j storage
    rag = LightRAG(
        working_dir=WORKING_DIR,
        # Use Neo4JStorage instead of default NetworkXStorage
        graph_storage="Neo4JStorage",
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        # Neo4j configuration parameters
        graph_storage_cls_kwargs={
            "uri": "neo4j://localhost:7687",
            "user": "neo4j",
            "password": "password",
            "database": "neo4j"
        }
    )
    
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def main():
    rag = None
    try:
        # Initialize RAG instance
        rag = await initialize_rag()
        
        # Download sample text if it doesn't exist
        if not os.path.exists("./book.txt"):
            import subprocess
            subprocess.run(["curl", "https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt", "-o", "./book.txt"])
        
        # Read the file
        with open("./book.txt", "r") as f:
            content = f.read()
        
        # Insert sample text - using await with the async version
        await rag.ainsert(content)  # <-- This is the key change!
        print("Content inserted successfully!")

        # Perform hybrid search - await the async method
        mode = "hybrid"
        result = await rag.query(
            "What are the main themes in this story?",
            param=QueryParam(mode=mode)
        )
        print("\nQuery Result:")
        print(result)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())  # Now we're running the async main() function