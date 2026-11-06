# HumayDrone Cloud Backend & Inference Service

This directory contains the central cloud backend and deep learning inference service for the **HumayDrone** platform, adhering to the capstone research specification:
*"Drone-Assisted Early Detection of Rice Discoloration with Mobile Mapping System"*.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │    Mobile Drone Controller App       │
                      │  (DJI MSDK V5 + Edge YOLOv8-Tiny)    │
                      └──────────────────┬───────────────────┘
                                         │  HTTP Multipart POST
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    Django REST API Ingestion Layer   │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                        Cloud Inference Pipeline                        │
  │  1. YOLOv8 (Full)  ──► High-Resolution Anomaly Detection               │
  │  2. U-Net          ──► Pixel Segmentation & Severity Area Scoring (%)  │
  │  3. ResNet-50      ──► Multi-Class Secondary Verification              │
  │  4. Advisory Rules ──► Dynamic Chemical/Cultural Recommendations       │
  └──────────────────────────────────────┬─────────────────────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │      PostgreSQL / SQLite Database    │
                      │  (Captures, Inference, GeoJSON Maps) │
                      └──────────────────────────────────────┘
```

---

## Quickstart

### 1. Requirements
- Python 3.10+
- Django 5+ or 6+
- djangorestframework
- django-cors-headers
- Pillow

### 2. Run Database Migrations
```bash
python manage.py migrate
```

### 3. Create Superuser (Admin Dashboard)
```bash
python manage.py createsuperuser
```

### 4. Start Redis and Celery Worker (for Background ML Inference)
In a separate terminal:
```bash
# Start Celery Worker
celery -A config worker --loglevel=info -P solo
```

### 5. Start Development Server
```bash
python manage.py runserver 0.0.0.0:8000
```

---

## Key API Endpoints

### 1. Flight Sessions
- `GET /api/flights/`: List all flight missions.
- `POST /api/flights/`: Create a new flight session.

### 2. Image Ingestion & Cloud Inference
- `POST /api/images/upload/` (multipart/form-data):
  - Parameters:
    - `flight_id` (int, required)
    - `latitude` (float)
    - `longitude` (float)
    - `altitude` (float, default: 3.0)
    - `file` (image binary, required)
    - `edge_detections` (optional JSON string with edge bounding boxes and classes)
  - Returns `202 Accepted` immediately with non-blocking task tracking:
    ```json
    {
      "message": "Image accepted for processing",
      "image_id": 14,
      "task_id": "c1f7259f-...",
      "status": "PENDING"
    }
    ```

- `GET /api/images/<image_id>/`:
  - Poll real-time status (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`) and inspect multi-model classification, severity score, and recommended treatment.

### 3. Analytics & Discoloration Map
- `GET /api/flights/<flight_id>/report/`:
  - Returns disease breakdown counts, average canopy discoloration severity, actionable treatment recommendations, and a ready-to-render **GeoJSON `FeatureCollection`** of discolored coordinates for Mapbox/GIS mapping.

---

## Running Automated Tests
```bash
python manage.py test core
```
