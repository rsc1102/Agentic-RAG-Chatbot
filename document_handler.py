from langchain_community.document_loaders.parsers.txt import TextParser
from langchain_community.document_loaders.parsers.pdf import PyPDFParser 
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_core.documents.base import Blob

class ParseDocument:
    def __init__(self):
        self.pdf_parser = PyPDFParser()
        self.txt_parser = TextParser()
        
    def __call__(self, document:UploadedFile):
        if document.type == "application/pdf":
            blob = Blob.from_data(document.getvalue())
            docs = self.pdf_parser.parse(blob)
            return docs
        elif document.type == "text/plain":
            blob = Blob.from_data(document.getvalue())
            docs = self.txt_parser.parse(blob)
            return docs
        
        raise Exception(f"document not of the type ['pdf','txt']. Provided document type: {document.type}")
    
document_parser = ParseDocument()
        