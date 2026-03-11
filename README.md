# Whisper Transcription API

API de transcription audio asynchrone utilisant Whisper via Triton Inference Server, FastAPI, Celery et Redis.

## Architecture

```
whisper-api/
├── api/                    # Service API (léger, ~200MB)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py         # Endpoints FastAPI
│       ├── celery_client.py
│       ├── schemas.py
│       └── config.py
├── worker/                 # Service Worker (léger, ~150MB)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── celery_app.py
│       ├── tasks.py        # Appelle Triton
│       └── config.py
├── triton/                 # Serveur d'inférence (~3GB)
│   ├── Dockerfile
│   └── model_repository/
│       └── whisper/
│           ├── config.pbtxt
│           └── 1/
│               └── model.py
└── docker-compose.yml
```

```
                                                     ┌─────────────┐
┌────────┐     ┌───────┐     ┌──────────┐           │  Worker 1   │
│ Client │────▶│  API  │────▶│  Redis   │──────────▶│  Worker 2   │
│        │◀────│ :8000 │◀────│  Queue   │           │  Worker N   │
└────────┘     └───────┘     └──────────┘           └──────┬──────┘
                                                           │
                                                           ▼
                                                    ┌─────────────┐
                                                    │   Triton    │
                                                    │   :8001     │
                                                    │  (Whisper)  │
                                                    └─────────────┘
```

## Avantages de Triton

| Aspect | Sans Triton | Avec Triton |
|--------|-------------|-------------|
| Mémoire par worker | ~2GB (modèle chargé) | ~150MB (client HTTP) |
| Scaling workers | Limité par RAM | Illimité |
| Gestion du modèle | Par worker | Centralisée |
| Batching | Manuel | Automatique |
| Monitoring | Basique | Métriques Prometheus |

## Prérequis

- Docker & Docker Compose

## Installation

```bash
# Cloner le repo
git clone <repo-url>
cd whisper-api

# Lancer la stack
docker-compose up -d

# Vérifier (attendre ~60s pour Triton)
curl http://localhost:8000/health
curl http://localhost:8001/v2/health/ready
```

## Utilisation

### Soumettre une transcription

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "language=fr"
```

Réponse :
```json
{
  "job_id": "abc123-def456",
  "status": "pending"
}
```

### Récupérer le résultat

```bash
curl "http://localhost:8000/transcribe/abc123-def456"
```

Réponse :
```json
{
  "job_id": "abc123-def456",
  "status": "completed",
  "result": {
    "text": "Bonjour, ceci est une transcription.",
    "segments": [
      {"start": 0.0, "end": 2.5, "text": "Bonjour,"},
      {"start": 2.5, "end": 5.0, "text": "ceci est une transcription."}
    ],
    "language": "fr",
    "language_probability": 0.98,
    "duration": 5.0
  }
}
```

## Scaling

```bash
# Scale les workers (clients légers)
docker-compose up -d --scale worker=10

# Scale Triton (si plusieurs instances nécessaires)
docker-compose up -d --scale triton=2
```

## Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/transcribe` | Soumettre un fichier audio |
| GET | `/transcribe/{job_id}` | Récupérer le statut/résultat |
| DELETE | `/transcribe/{job_id}` | Annuler une transcription |
| GET | `/health` | Health check API |
| GET | `/docs` | Documentation Swagger |

### Endpoints Triton

| Port | Endpoint | Description |
|------|----------|-------------|
| 8001 | `/v2/health/ready` | Health check |
| 8001 | `/v2/models/whisper` | Info modèle |
| 8002 | gRPC | Inférence gRPC |
| 8003 | `/metrics` | Métriques Prometheus |

## Formats audio supportés

- WAV (.wav)
- MP3 (.mp3)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)
- WebM (.webm)

## Configuration

### Variables d'environnement API

| Variable | Description | Défaut |
|----------|-------------|--------|
| `REDIS_URL` | URL Redis | `redis://localhost:6379/0` |
| `UPLOAD_DIR` | Dossier uploads | `/tmp/whisper_uploads` |

### Variables d'environnement Worker

| Variable | Description | Défaut |
|----------|-------------|--------|
| `REDIS_URL` | URL Redis | `redis://localhost:6379/0` |
| `TRITON_URL` | URL Triton | `triton:8000` |

### Configuration Triton (config.pbtxt)

```protobuf
parameters: {
  key: "WHISPER_MODEL"
  value: { string_value: "tiny" }  # tiny, base, small, medium, large-v3
}

parameters: {
  key: "WHISPER_DEVICE"
  value: { string_value: "cpu" }  # cpu ou cuda
}

parameters: {
  key: "WHISPER_COMPUTE_TYPE"
  value: { string_value: "int8" }  # int8, float16, float32
}

instance_group [
  {
    count: 2  # Nombre d'instances du modèle
    kind: KIND_CPU
  }
]
```

## Modèles Whisper

| Modèle | Paramètres | RAM | Vitesse |
|--------|------------|-----|---------|
| tiny | 39M | ~1 GB | Très rapide |
| base | 74M | ~1 GB | Rapide |
| small | 244M | ~2 GB | Moyen |
| medium | 769M | ~5 GB | Lent |
| large-v3 | 1550M | ~10 GB | Très lent |

## Monitoring

- **Flower** (Celery) : http://localhost:5555
- **Triton Metrics** : http://localhost:8003/metrics

## GPU (optionnel)

Pour utiliser un GPU, modifie `triton/model_repository/whisper/config.pbtxt` :

```protobuf
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

parameters: {
  key: "WHISPER_DEVICE"
  value: { string_value: "cuda" }
}

parameters: {
  key: "WHISPER_COMPUTE_TYPE"
  value: { string_value: "float16" }
}
```

Et dans `docker-compose.yml` :

```yaml
triton:
  build: ./triton
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Logs

```bash
# Tous les logs
docker-compose logs -f

# Triton uniquement
docker-compose logs -f triton

# Workers
docker-compose logs -f worker
```

## License

MIT
