# Utiliser une image Python officielle comme base
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Variables d'environnement pour optimiser Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt en premier pour optimiser le cache Docker
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Installer les dépendances supplémentaires pour FastAPI et le déploiement
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    python-multipart==0.0.6 \
    aiohttp==3.9.0 \
    rich==13.7.0 \
    psutil==5.9.6 \
    joblib==1.3.2

# Créer les répertoires nécessaires
RUN mkdir -p /app/models /app/data /app/logs

# Copier tout le code source
COPY . .

# Créer un utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Exposer le port de l'API
EXPOSE 8000

# Variables d'environnement pour l'application
ENV MODEL_PATH=/app/models \
    DATA_PATH=/app/data \
    PORT=8000 \
    HOST=0.0.0.0

# Healthcheck pour vérifier que l'API est fonctionnelle
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande par défaut pour démarrer l'application
CMD ["python", "run_deployment.py", "deploy", "--port", "8000"]