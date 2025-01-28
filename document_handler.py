from langchain_community.document_loaders.parsers.txt import TextParser
from langchain_community.document_loaders.parsers.pdf import PyPDFParser 
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_core.documents.base import Document
from langchain_core.documents.base import Blob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
import os


class ParseDocument:
    """Document Parser class. Supports pdf and txt documents."""
    def __init__(self):
        self.pdf_parser = PyPDFParser()
        self.txt_parser = TextParser()
        
    def __call__(self, document:UploadedFile, client_id: str) -> list[Document]:
        if document.type == "application/pdf":
            blob = Blob.from_data(document.getvalue())
            docs = self.pdf_parser.parse(blob)
            for doc in docs: #Adding client_id metadata 
                doc.metadata['client_id'] = client_id
            return docs
        elif document.type == "text/plain":
            blob = Blob.from_data(document.getvalue())
            docs = self.txt_parser.parse(blob)
            for doc in docs: #Adding client_id metadata 
                doc.metadata['client_id'] = client_id
            return docs
        
        raise Exception(f"document not of the type ['pdf','txt']. Provided document type: {document.type}")
    

class VectorizeDocument:
    """Document Vectorizing class. Uses AstraDB as a vector store.
    """
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                                model_name="gpt-4",
                                chunk_size=100,
                                chunk_overlap=50,
                            )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # checking env variables
        if not os.environ.get("ASTRA_DB_API_ENDPOINT"):
            raise Exception("ASTRA_DB_API_ENDPOINT not specified")
        if not os.environ.get("ASTRA_DB_COLLECTION_NAME"):
            raise Exception("ASTRA_DB_COLLECTION_NAME not specified")
        if not os.environ.get("ASTRA_DB_APPLICATION_TOKEN"):
            raise Exception("ASTRA_DB_APPLICATION_TOKEN not specified")
        
        self.vector_store = AstraDBVectorStore(
                            embedding = self.embeddings,
                            api_endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT"),
                            collection_name=os.environ.get("ASTRA_DB_COLLECTION_NAME"),
                            token=os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
                        )
    
    def __call__(self,documents):
        self.vector_store.add_documents(documents=documents)   

document_parser = ParseDocument()
vectorize_document = VectorizeDocument()
        