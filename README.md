# 🎙️ SubSync — Whisper Video Subtitling Service

Service de **sous-titrage vidéo et audio asynchrone** basé sur Whisper, déployé via Docker Compose. Les fichiers sont uploadés via une API REST, traités par des workers Celery qui interrogent un serveur Triton, et les résultats sont diffusés en temps réel via **Server-Sent Events (SSE)**.

---

## Architecture

```
whisper-front  ──HTTP──►  api  ──publish──►  redis  ──consume──►  worker  ──http──►  triton
                           │                                          │
                           │◄──────── webhook /internal/webhook ◄────┘
                           │
                           └──────────────────── minio ◄─────────────┘
                                        (stockage vidéo / audio S3)

flower          →  monitoring Celery
prometheus      →  métriques Triton
grafana         →  dashboards
```

| Service | Rôle | Port |
|---|---|---|
| `whisper-front` | Interface web Vite | 5173 |
| `api` | API REST FastAPI v4.0 | 8000 |
| `redis` | Broker de messages Celery | 6379 |
| `minio` | Stockage objet S3 (vidéos / audio / sorties) | 9000 · 9001 |
| `worker` | Worker Celery (traitement async) | — |
| `triton` | Inférence Whisper (Triton Server) | 8001 |
| `flower` | Dashboard monitoring Celery | 5555 |
| `prometheus` | Collecte métriques Triton | 9090 |
| `grafana` | Visualisation dashboards | 3000 |

---

## Prérequis

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) ≥ 2.20
- GPU NVIDIA avec drivers ≥ 525 (pour Triton)
- `nvidia-container-toolkit` installé et configuré

---

## Installation & démarrage

### 1. Cloner le dépôt

```bash
git clone https://github.com/your-org/subsync.git
cd subsync
```

### 2. Configurer les variables d'environnement

```bash
cp .env.example .env
```

Éditer `.env` :

```env
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
```

### 3. Lancer la stack

```bash
docker compose up -d
```

Le conteneur `minio-init` crée automatiquement le bucket `whisper-audio` avec une règle d'expiration à **1 jour**, puis s'arrête (`exited 0`). Triton peut prendre jusqu'à **60 secondes** pour être prêt.

### 4. Vérifier que les services sont up

```bash
docker compose ps
```

```bash
# API
curl http://localhost:8000/health

# Triton
curl http://localhost:8001/v2/health/ready
```

---

## Structure du projet

```
.
├── front/                  # Application Vite
│   └── Dockerfile
├── api/                    # API FastAPI
│   ├── app/
│   │   ├── main.py         # Routes, SSE, webhook
│   │   ├── celery_client.py
│   │   ├── schemas.py
│   │   └── config.py
│   ├── requirements.txt
│   └── Dockerfile
├── worker/                 # Worker Celery
│   ├── app/
│   │   ├── tasks.py        # Pipeline de transcription
│   │   ├── celery_app.py
│   │   └── config.py
│   ├── requirements.txt
│   └── Dockerfile
├── triton/                 # Modèle Whisper + config Triton
│   ├── model_repository/
│   │   └── whisper/
│   │       ├── config.pbtxt
│   │       └── 1/
│   │           └── model.py
│   └── Dockerfile
├── prometheus.yml
├── docker-compose.yml
└── .env.example
```

---

## API — Référence

Documentation interactive disponible à **http://localhost:8000/docs**.

### Formats acceptés

| Type | Extensions |
|---|---|
| Vidéo | `.mp4` `.mov` `.mkv` `.webm` `.avi` |
| Audio | `.mp3` `.wav` `.ogg` `.flac` `.m4a` `.aac` |

### Modes de sortie

| Mode | Description | Vidéo | Audio |
|---|---|---|---|
| `embed` | Vidéo avec sous-titres soft intégrés (sans ré-encodage) | ✅ | ❌ |
| `srt` | Fichier `.srt` avec timestamps | ✅ | ✅ |
| `text` | Fichier `.txt` texte brut sans timestamps | ✅ | ✅ |

---

### `POST /subtitles` — Soumettre un fichier

```bash
curl -X POST "http://localhost:8000/subtitles?mode=srt&language=fr" \
  -F "file=@video.mp4"
```

Paramètres query :

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `mode` | string | `embed` | Mode de sortie : `embed`, `srt`, `text` |
| `language` | string | — | Code ISO facultatif (`fr`, `en`…). Détection auto si absent. |

Réponse :

```json
{
  "job_id": "3f2a1b4c-...",
  "status": "pending"
}
```

---

### `GET /subtitles/{job_id}/stream` — Résultat temps réel (SSE)

Connexion SSE bloquée jusqu'à la fin du traitement (timeout **30 min**). Approche recommandée.

```bash
curl -N http://localhost:8000/subtitles/3f2a1b4c-.../stream
```

Séquence d'événements :

```
event: connected
data: 3f2a1b4c-...

event: done
data: https://localhost:9000/whisper-audio/video/...?X-Amz-...
```

En cas d'erreur ou timeout :

```
event: error
data: <message>
```

L'URL reçue dans `done` est une **presigned URL S3 valide 1 heure**.

---

### `GET /subtitles/{job_id}` — Statut (polling)

Alternative au SSE pour les clients sans support des connexions longues.

```bash
curl http://localhost:8000/subtitles/3f2a1b4c-...
```

Réponse une fois terminé :

```json
{
  "job_id": "3f2a1b4c-...",
  "status": "completed",
  "result": {
    "text": "Bonjour, voici la transcription...",
    "mode": "srt",
    "download_url": "https://localhost:9000/..."
  }
}
```

Valeurs possibles de `status` : `pending` · `started` · `completed` · `failed`.

---

### `DELETE /subtitles/{job_id}` — Annuler un job

```bash
curl -X DELETE http://localhost:8000/subtitles/3f2a1b4c-...
```

```json
{ "job_id": "3f2a1b4c-...", "status": "cancelled" }
```

Révoque la tâche Celery (`terminate=True`) et libère la queue SSE associée.

---

### `GET /health` — Santé de l'API

```bash
curl http://localhost:8000/health
# {"api": "ok", "redis": "ok"}
```

---

### Endpoints Triton

| Port | Endpoint | Description |
|---|---|---|
| 8001 | `GET /v2/health/ready` | Health check |
| 8001 | `GET /v2/models/whisper` | Infos modèle |
| 8002 | gRPC | Inférence gRPC |
| 8003 | `GET /metrics` | Métriques Prometheus |

---

## Flux de traitement

1. **Upload** — Le front envoie le fichier à `POST /subtitles`. L'API le stocke dans MinIO sous `video/pending/<uuid>.<ext>` ou `audio/pending/<uuid>.<ext>` et publie une tâche Celery dans Redis (clé S3 + langue + mode + callback URL).
2. **Traitement** — Le worker consomme la tâche, télécharge le fichier depuis MinIO, extrait l'audio (ffmpeg pour les vidéos, décodage direct pour l'audio), découpe en chunks de 30s avec 1s d'overlap, envoie chaque chunk à Triton (Whisper), fusionne les segments avec timestamps recalés.
3. **Sortie** — Selon le `mode`, le worker génère la vidéo sous-titrée, le `.srt` ou le `.txt`, l'uploade sur MinIO, puis supprime le fichier source.
4. **Webhook** — Le worker appelle `POST /internal/webhook` avec le `job_id` et la clé S3 du fichier de sortie.
5. **Notification SSE** — L'API reçoit le webhook, dépose le résultat dans l'`asyncio.Queue` du job, ce qui débloque instantanément le client SSE avec la presigned URL.

---

## Configuration Triton

### Modèles disponibles

| Modèle | Paramètres | RAM | Vitesse |
|---|---|---|---|
| `tiny` | 39M | ~1 GB | Très rapide |
| `base` | 74M | ~1 GB | Rapide |
| `small` | 244M | ~2 GB | Moyen |
| `medium` | 769M | ~5 GB | Lent |
| `large-v3` | 1550M | ~10 GB | Très lent |

### `config.pbtxt` (CPU)

```protobuf
parameters: {
  key: "WHISPER_MODEL"
  value: { string_value: "base" }
}
parameters: {
  key: "WHISPER_DEVICE"
  value: { string_value: "cpu" }
}
parameters: {
  key: "WHISPER_COMPUTE_TYPE"
  value: { string_value: "int8" }
}
instance_group [{ count: 2, kind: KIND_CPU }]
```

### GPU (optionnel)

Dans `config.pbtxt` :

```protobuf
instance_group [{ count: 1, kind: KIND_GPU }]
parameters: {
  key: "WHISPER_DEVICE"
  value: { string_value: "cuda" }
}
parameters: {
  key: "WHISPER_COMPUTE_TYPE"
  value: { string_value: "float16" }
}
```

Dans `docker-compose.yml`, ajouter sous `triton` :

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## Variables d'environnement

### API

| Variable | Défaut | Description |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | URL Redis |
| `S3_ENDPOINT_URL` | `http://minio:9000` | URL interne MinIO |
| `S3_BUCKET` | `whisper-audio` | Nom du bucket |
| `S3_ACCESS_KEY` | `minioadmin` | Clé d'accès MinIO |
| `S3_SECRET_KEY` | `minioadmin` | Clé secrète MinIO |

### Worker

| Variable | Défaut | Description |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | URL Redis |
| `TRITON_URL` | `triton:8000` | URL interne Triton |
| `S3_ENDPOINT_URL` | `http://minio:9000` | URL interne MinIO |
| `S3_BUCKET` | `whisper-audio` | Nom du bucket |
| `S3_ACCESS_KEY` | `minioadmin` | Clé d'accès MinIO |
| `S3_SECRET_KEY` | `minioadmin` | Clé secrète MinIO |

---

## Interfaces de monitoring

| Interface | URL | Identifiants |
|---|---|---|
| Swagger UI | http://localhost:8000/docs | — |
| MinIO Console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| Flower (Celery) | http://localhost:5555 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin` / `admin` |

---

## Scaling

```bash
# Ajouter des workers (sans toucher à l'API ni à Triton)
docker compose up -d --scale worker=4

# Plusieurs instances Triton si nécessaire
docker compose up -d --scale triton=2
```

> **Note** : le bus SSE est in-memory par processus API. En cas de scaling horizontal de l'API (plusieurs replicas derrière un load balancer), remplacer les `asyncio.Queue` par un pub/sub Redis pour garantir que le webhook atteint le bon replica.

---

## Commandes utiles

```bash
# Rebuild un service après modification
docker compose up -d --build api
docker compose up -d --build worker

# Logs en direct
docker compose logs -f worker
docker compose logs -f api
docker compose logs -f triton

# Arrêt propre
docker compose down

# Arrêt + suppression des volumes (efface Redis et MinIO)
docker compose down -v
```

---

## Licence

MIT