FROM nvcr.io/nvidia/tritonserver:25.12-py3

# Installation de ffmpeg et dépendances
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Installation de faster-whisper
RUN pip install --no-cache-dir \
    faster-whisper>=0.10.0 \
    numpy

# Copie du model repository
COPY model_repository /models

# Port Triton
EXPOSE 8000 8001 8002

# Commande de démarrage
CMD ["tritonserver", "--model-repository=/models", "--http-port=8000", "--grpc-port=8001", "--metrics-port=8002"]
