from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import numpy as np
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import requests
import json
import base64
import re

load_dotenv()

form_recognizer_endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
form_recognizer_key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_key = os.getenv("AZURE_OPENAI_KEY")
openai_api_version = "2023-05-15"
gpt_api_version = "2025-01-01-preview"
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
gpt_deployment = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT")

search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
search_index_name = "documents"

if not all([form_recognizer_endpoint, form_recognizer_key, search_endpoint, search_key, openai_endpoint, openai_key, embedding_deployment, gpt_deployment]):
    missing_vars = []
    if not form_recognizer_endpoint:
        missing_vars.append("AZURE_FORM_RECOGNIZER_ENDPOINT")
    if not form_recognizer_key:
        missing_vars.append("AZURE_FORM_RECOGNIZER_KEY")
    if not search_endpoint:
        missing_vars.append("AZURE_SEARCH_SERVICE_ENDPOINT")
    if not search_key:
        missing_vars.append("AZURE_SEARCH_ADMIN_KEY")
    if not openai_endpoint:
        missing_vars.append("AZURE_OPENAI_ENDPOINT")
    if not openai_key:
        missing_vars.append("AZURE_OPENAI_KEY")
    if not embedding_deployment:
        missing_vars.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if not gpt_deployment:
        missing_vars.append("AZURE_OPENAI_GPT_DEPLOYMENT")
    
    raise ValueError(f"Les variables d'environnement suivantes sont manquantes : {', '.join(missing_vars)}")

document_analysis_client = DocumentAnalysisClient(
    endpoint=form_recognizer_endpoint,
    credential=AzureKeyCredential(form_recognizer_key)
)

openai_client = AzureOpenAI(
    api_key=openai_key,
    api_version=openai_api_version,
    azure_endpoint=openai_endpoint,
    max_retries=3
)

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=search_index_name,
    credential=AzureKeyCredential(search_key)
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    user_id: str
    pdf_id: str

class SearchRequest(BaseModel):
    query: str
    user_id: str
    top_k: Optional[int] = 3

def get_embedding(text: str) -> List[float]:
    try:
        response = openai_client.embeddings.create(
            model=embedding_deployment,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Erreur lors de la génération de l'embedding: {str(e)}")
        raise

def vector_search(vector, user_id, pdf_id, top_k=3):
    try:
        results = search_client.search(
            search_text="*",
            filter=f"user_id eq '{user_id}' and pdf_id eq '{pdf_id}'",
            vector_queries=[{
                "vector": vector,
                "fields": "embedding",
                "kind": "vector",
                "k": top_k
            }],
            select=["id", "content", "document_name", "timestamp"]
        )
        
        return {"value": list(results)}
    except Exception as e:
        print(f"Erreur lors de la recherche vectorielle: {str(e)}")
        raise

def encode_document_id(filename: str, page_number: int) -> str:
    raw_id = f"{filename}-page-{page_number}"
    encoded_id = base64.urlsafe_b64encode(raw_id.encode()).decode()
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '-', encoded_id)
    return safe_id

def split_text(text, max_tokens=30):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    paragraphs = re.split(r'\n{2,}', text)
    
    chunks = []
    for paragraph in paragraphs:
        paragraph = paragraph.replace('\n', ' ').strip()
    
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        current_chunk = ""
        for sentence in sentences:
            if len(sentence.split()) > max_tokens:
                words = sentence.split()
                for i in range(0, len(words), max_tokens):
                    chunk = ' '.join(words[i:i + max_tokens])
                    chunks.append(chunk)
            else:
                if len((current_chunk + " " + sentence).split()) <= max_tokens:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:

                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())
    
    chunks = [chunk for chunk in chunks if len(chunk.split()) >= 5] 
    
    return chunks

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...), user_id: str = Body(...)):
    try:
        
        file_content = await file.read()
        
        if len(file_content) == 0:
            return {"status": "error", "message": "Le fichier est vide"}
            
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-read", file_content
        )
        result = poller.result()

        pdf_id = f"{file.filename}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}"

        chunks = split_text(result.content)

        documents = []
        for chunk_index, chunk in enumerate(chunks):
            print("------------------")
            print(chunk)
            embedding = get_embedding(chunk)

            chunk_id = f"{file.filename}-chunk-{chunk_index}"
            encoded_id = base64.urlsafe_b64encode(chunk_id.encode()).decode()
            safe_id = re.sub(r'[^a-zA-Z0-9_-]', '-', encoded_id)
            
            document = {
                "id": safe_id,
                "user_id": str(user_id),
                "content": str(chunk),
                "embedding": embedding,
                "document_name": str(file.filename),
                "pdf_id": pdf_id,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            }
            documents.append(document)

        if not documents:
            return {"status": "error", "message": "Aucun contenu n'a pu être extrait du document"}

        search_client.upload_documents(documents)

        return {
            "status": "success",
            "message": "Document analysé et indexé avec succès",
            "data": {
                "chunks_processed": len(documents),
                "document_name": file.filename,
                "pdf_id": pdf_id
            }
        }
    
    except Exception as e:
        print(f"Erreur lors de l'analyse du document: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        pdf_id = request.pdf_id
        question_embedding = get_embedding(request.question)
        
        results = vector_search(question_embedding, request.user_id, pdf_id)
        relevant_docs = results.get("value", [])
        print(relevant_docs)

        context = "\n\n".join([
            f"Page document {doc['document_name']}:\n{doc['content']}"
            for doc in relevant_docs
        ])

        messages = [
            {"role": "system", "content": "Vous êtes un assistant qui répond aux questions en se basant sur le contexte fourni. Si la réponse n'est pas dans le contexte, dites-le clairement."},
            {"role": "user", "content": f"Contexte : {context}\n\nQuestion : {request.question}"}
        ]
        
        response = openai_client.chat.completions.create(
            model=gpt_deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return {
            "status": "success",
            "answer": response.choices[0].message.content,
            "sources": [
                {
                    "document_name": doc["document_name"],
                    "content": doc["content"]
                }
                for doc in relevant_docs
            ]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)