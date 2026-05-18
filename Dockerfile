# ── Image de base ─────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Métadonnées ───────────────────────────────────────────────────────────
LABEL maintainer="Manuella Diffo Tahboh <manuellatahboh@gmail.com>"
LABEL description="Multi-Agent RAG System — LangGraph + Neo4j + FastAPI"

# ── Variables d'environnement ─────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── Répertoire de travail ─────────────────────────────────────────────────
WORKDIR /app

# ── Installation des dépendances système ──────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Installation des dépendances Python ──────────────────────────────────
# Copier requirements en premier pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copie du code source ──────────────────────────────────────────────────
COPY . .

# ── Création du répertoire de persistance ChromaDB ───────────────────────
RUN mkdir -p /app/chroma_db

# ── Port exposé ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Commande de démarrage ─────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
