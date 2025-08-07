# from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
# from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone import ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from dotenv import load_dotenv
# import os

# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# extracted_data = load_pdf_file(data='Data/')
# text_chunks = text_split(extracted_data)
# embeddings = download_hugging_face_embeddings()

# pc = Pinecone(api_key=PINECONE_API_KEY)

# index_name = "medicalbot"

# pc.create_index(
#     name = index_name,
#     dimension = 384,
#     metric = "cosine",
#     spec = ServerlessSpec(
#         cloud = "aws",
#         region = "us-east-1"
#     )
# )


# docsearch = PineconeVectorStore.from_documents(
#     documents=text_chunks,
#     index_name=index_name,
#     embedding=embeddings,
# )
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import time

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

import os
print("📂 Files in /Data/:", os.listdir("Data/"))


# Step 1: Load and chunk PDF
extracted_data = load_pdf_file(data='Data/')
print(f"📄 Loaded {len(extracted_data)} documents from PDF.")

text_chunks = text_split(extracted_data)
print(f"✂️ Created {len(text_chunks)} text chunks.")


# Step 2: Get embeddings
embeddings = download_hugging_face_embeddings()

# Step 3: Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

# Step 4: Create index only if it doesn't exist
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("✅ Index created.")
    time.sleep(10)  # wait for the index to become ready

# Step 5: Upload to Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print(f"✅ Successfully uploaded {len(text_chunks)} documents to Pinecone.")
