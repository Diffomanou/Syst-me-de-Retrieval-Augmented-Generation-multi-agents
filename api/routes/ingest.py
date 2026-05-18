import logging
import uuid
import io
from fastapi import APIRouter, HTTPException, UploadFile, File
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pour lire les PDF, installez : pip install pypdf
try:
    import pypdf
except ImportError:
    pypdf = None

from api.schemas import IngestRequest, IngestResponse
from database import neo4j_client, vector_store
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)

CONCEPT_EXTRACTION_PROMPT = """Extrais les 5 concepts principaux de ce texte.
Réponds UNIQUEMENT avec les concepts séparés par des virgules.
Exemple : intelligence artificielle, réseaux de neurones, apprentissage supervisé"""

def extract_concepts(text: str) -> list[str]:
    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.groq_api_key,
    )
    response = llm.invoke([
        SystemMessage(content=CONCEPT_EXTRACTION_PROMPT),
        HumanMessage(content=text[:2000]),
    ])
    return [c.strip() for c in response.content.split(",") if c.strip()]

# --- FONCTION COMMUNE D'INGESTION ---
async def run_ingestion_pipeline(title: str, content: str, source: str) -> IngestResponse:
    """La logique centrale que tes deux routes vont utiliser."""
    chunks = text_splitter.split_text(content)
    concepts = extract_concepts(content)
    
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"title": title, "source": source, "chunk_index": i} for i, _ in enumerate(chunks)]
    
    vector_store.add_documents(texts=chunks, metadatas=metadatas, ids=chunk_ids)

    for i, (chunk_id, chunk) in enumerate(zip(chunk_ids, chunks)):
        neo4j_client.add_document(
            doc_id=chunk_id,
            title=title,
            source=source,
            chunk_index=i,
            concepts=concepts,
        )
    
    return IngestResponse(
        success=True,
        chunk_ids=chunk_ids,
        chunks_created=len(chunks),
        message=f"Document '{title}' ingéré avec succès."
    )

# --- ROUTE 1 : TEXTE BRUT (Celle que tu avais) ---
@router.post("", response_model=IngestResponse, summary="Ingérer du texte brut")
async def ingest_endpoint(request: IngestRequest) -> IngestResponse:
    try:
        return await run_ingestion_pipeline(request.title, request.content, request.source)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# --- ROUTE 2 : UPLOAD DE FICHIER (La nouvelle !) ---
@router.post("/file", response_model=IngestResponse, summary="Uploader un fichier (PDF ou TXT)")
async def ingest_file_endpoint(file: UploadFile = File(...)) -> IngestResponse:
    try:
        content = ""
        # Lecture du fichier selon l'extension
        if file.filename.endswith(".txt"):
            content = (await file.read()).decode("utf-8")
            
        elif file.filename.endswith(".pdf"):
            if not pypdf:
                raise HTTPException(status_code=500, detail="Librairie 'pypdf' non installée pour lire les PDF.")
            pdf_reader = pypdf.PdfReader(io.BytesIO(await file.read()))
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        else:
            raise HTTPException(status_code=400, detail="Seuls les fichiers .txt et .pdf sont acceptés.")

        return await run_ingestion_pipeline(file.filename, content, "file_upload")

    except Exception as exc:
        logger.error("Erreur d'ingestion de fichier : %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))