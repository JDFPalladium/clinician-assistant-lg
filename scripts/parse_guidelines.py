import os
import asyncio
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document

# Load environment variables
load_dotenv("config.env")

# Set up LlamaParse
parser = LlamaParse(
    api_key=os.environ.get("LLAMAPARSE_API_KEY"),
    result_type="markdown",
    extract_charts=True,
    auto_mode=True,
    auto_mode_trigger_on_image_in_page=True,
    auto_mode_trigger_on_table_in_page=True,
    bbox_top=0.05,
    bbox_bottom=0.1,
    verbose=True,
)

# Create output directory if it doesn't exist
os.makedirs("data/processed/lp/indices", exist_ok=True)

async def parse_docs():
    for filename in os.listdir("data/raw/GuidelinesSections"):
        if filename.endswith(".pdf"):
            filepath = f"data/raw/GuidelinesSections/{filename}"
            print(f"Processing: {filepath}")

            try:
                documents = await parser.aload_data(filepath)
            except Exception as e:
                print(f"❌ Failed to parse {filename}: {e}")
                continue

            full_text = "\n\n".join(doc.text for doc in documents)
            combined_doc = Document(text=full_text)

            node_parser = SimpleNodeParser()
            nodes = node_parser.get_nodes_from_documents([combined_doc])

            index = VectorStoreIndex(nodes)

            short_filename = (
                filename.replace("Kenya-ARV-Guidelines-2022-", "")
                .replace(".pdf", "")
            )

            index.storage_context.persist(persist_dir=f"data/processed/lp/indices/{short_filename}")
            print(f"✅ Saved index for {short_filename}")

if __name__ == "__main__":
    asyncio.run(parse_docs())
