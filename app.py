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
import firebase_admin
from firebase_admin import credentials, firestore, storage
from elevenlabs import ElevenLabs

cred = credentials.Certificate("./medisense-8ca26-firebase-adminsdk-fbsvc-edee655a7c.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'medisense-8ca26.firebasestorage.app'
})
db = firestore.client()
bucket = storage.bucket()

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
    vocalMode: Optional[bool] = False

class SearchRequest(BaseModel):
    query: str
    user_id: str
    top_k: Optional[int] = 3

class DocumentResponse(BaseModel):
    id: str
    filename: str
    upload_date: datetime
    size: float
    status: str
    chunks_count: Optional[int] = None
    pdf_id: Optional[str] = None
    storage_url: Optional[str] = None
    error_message: Optional[str] = None

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
        print(results)
        
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
            
        timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
        unique_filename = f"{user_id}/{timestamp}_{file.filename}"
        
        blob = bucket.blob(unique_filename)
        blob.upload_from_string(
            file_content,
            content_type=file.content_type
        )

        blob.make_public()
        public_url = blob.public_url
            
        doc_ref = db.collection('documents').document()
        doc_ref.set({
            'filename': file.filename,
            'upload_date': datetime.utcnow(),
            'size': len(file_content)/1024,
            'user_id': user_id,
            'status': 'processing',
            'storage_url': public_url,
            'storage_path': unique_filename
        })
            
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
            doc_ref.update({'status': 'error', 'error_message': 'Aucun contenu n\'a pu être extrait'})
            return {"status": "error", "message": "Aucun contenu n'a pu être extrait du document"}

        search_client.upload_documents(documents)
        
        doc_ref.update({
            'status': 'completed',
            'chunks_count': len(documents),
            'pdf_id': pdf_id
        })

        return {
            "status": "success",
            "message": "Document analysé et indexé avec succès",
            "data": {
                "chunks_processed": len(documents),
                "document_name": file.filename,
                "pdf_id": pdf_id,
                "storage_url": public_url
            }
        }
    
    except Exception as e:
        if 'doc_ref' in locals():
            doc_ref.update({'status': 'error', 'error_message': str(e)})
        return {"status": "error", "message": str(e)}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        print(request)
        pdf_id = request.pdf_id
        question_embedding = get_embedding(request.question)
        
        results = vector_search(question_embedding, request.user_id, pdf_id)
        relevant_docs = results.get("value", [])
        print(relevant_docs)

        # Mots-clés pour détecter une demande de rendez-vous
        rdv_keywords = ["rdv", "@", "email", "email rendez vous", "rendez vous", "rendez-vous", "prendre rendez-vous", "souhaite prendre un rdv", 
                       "souhaite prendre rendez-vous", "voulez prendre un rdv", "voulez prendre rendez-vous"]

        # Mots-clés pour détecter une demande de vocal par mail
        vocal_mail_keywords = ["vocal", "voix", "audio", "enregistrement"]

        if any(keyword in request.question.lower() for keyword in rdv_keywords):
            is_vocal_mail = any(keyword in request.question.lower() for keyword in vocal_mail_keywords)
            
            if is_vocal_mail:
                all_docs = search_client.search(
                    search_text="*",
                    filter=f"user_id eq '{request.user_id}' and pdf_id eq '{pdf_id}'",
                    select=["content", "document_name"]
                )
                
                full_context = "\n\n".join([
                    f"\n{doc['content']}"
                    for doc in all_docs
                ])
                context = full_context
            else:
                context = "\n\n".join([
                    f"Page document {doc['document_name']}:\n{doc['content']}"
                    for doc in relevant_docs
                ])

            webhook_data = {
                "question": request.question,
                "context": context,
                "user_id": request.user_id,
                "pdf_id": pdf_id,
                "relevant_docs": [
                    {
                        "document_name": doc["document_name"],
                        "content": doc["content"]
                    }
                    for doc in relevant_docs
                ]
            }

            response = requests.post(
                "https://n8n-prod.makeitpost.com/webhook/866df774-73fb-4176-b0ec-a4a8c499aaaf",
                json=webhook_data
            )

            print(response.json())

            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Erreur lors de l'appel au webhook: {response.status_code}"
                }

            n8n_response = response.json()
            if isinstance(n8n_response, list) and len(n8n_response) > 0 and "output" in n8n_response[0]:
                answer = n8n_response[0]["output"]
                response_data = {
                    "status": "success",
                    "answer": answer,
                    "sources": [
                        {
                            "document_name": doc["document_name"],
                            "content": doc["content"]
                        }
                        for doc in relevant_docs
                    ]
                }

                if request.vocalMode:
                    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
                    if elevenlabs_api_key:
                        client = ElevenLabs(api_key=elevenlabs_api_key)
                        audio_generator = client.text_to_speech.convert(
                            voice_id="TGAegA0zNRi8I6nUdq3i",
                            output_format="mp3_44100_128",
                            text=answer,
                            model_id="eleven_multilingual_v2"
                        )
                        audio_bytes = b''.join(audio_generator)
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        response_data["audio"] = audio_base64

                return response_data
            else:
                return {
                    "status": "error",
                    "message": "Format de réponse n8n invalide"
                }
        else:
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

            answer = response.choices[0].message.content
            response_data = {
                "status": "success",
                "answer": answer,
                "sources": [
                    {
                        "document_name": doc["document_name"],
                        "content": doc["content"]
                    }
                    for doc in relevant_docs
                ]
            }

            if request.vocalMode:
                elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
                if elevenlabs_api_key:
                    client = ElevenLabs(api_key=elevenlabs_api_key)
                    audio_generator = client.text_to_speech.convert(
                        voice_id="TGAegA0zNRi8I6nUdq3i",
                        output_format="mp3_44100_128",
                        text=answer,
                        model_id="eleven_multilingual_v2"
                    )
                    audio_bytes = b''.join(audio_generator)
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    response_data["audio"] = audio_base64

            return response_data

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/documents/{user_id}")
async def get_user_documents(user_id: str):
    try:
        docs = db.collection('documents').where('user_id', '==', user_id).stream()
        
        documents = []
        for doc in docs:
            doc_data = doc.to_dict()
            documents.append(DocumentResponse(
                id=doc.id,
                filename=doc_data.get('filename'),
                upload_date=doc_data.get('upload_date'),
                size=doc_data.get('size'),
                status=doc_data.get('status'),
                chunks_count=doc_data.get('chunks_count'),
                pdf_id=doc_data.get('pdf_id'),
                error_message=doc_data.get('error_message'),
                storage_url=doc_data.get('storage_url')
            ))
        
        return {
            "status": "success",
            "documents": documents
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)