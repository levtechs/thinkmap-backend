print("importing dependencies...")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import uvicorn
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
import logging

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("thinkmap-backend")

# ---------- App ----------
app = FastAPI(title="ThinkMap Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("loading models...")

# ---------- Models / Globals ----------
# NOTE: all-MiniLM-L6-v2 has 384-dim embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
pca = PCA(n_components=3)


# ---------- Schemas ----------
class NoteRequest(BaseModel):
    id: int
    text: str


class PointResponse(BaseModel):
    id: int
    x: float
    y: float
    z: float


class AddNoteRequest(BaseModel):
    note: NoteRequest
    prev_ids: List[int]
    prev_embeds: List[List[float]]
    prev_points: List[List[float]]


class AddNoteResponse(BaseModel):
    points: List[PointResponse]
    embeddings: List[List[float]]


# ---------- Startup / Health ----------
@app.on_event("startup")
def _warmup():
    """
    Warm up the model so the first real request doesn't block the server longer than necessary.
    """
    try:
        model.encode(["warmup"])
        log.info("Model warmup complete.")
    except Exception as e:
        log.warning(f"Model warmup failed (continuing anyway): {e}")


@app.get("/")
def read_root():
    return {"message": "Hello from ThinkMap"}


@app.get("/_health")
def health():
    return {"ok": True}


# ---------- Core compute (runs in a thread pool) ----------
def process_add_note(request: AddNoteRequest) -> AddNoteResponse:
    log.info("Processing /add_note request...")
    note = request.note
    log.info(f"note: id={note.id} text_len={len(note.text)}")

    # Prior data
    embeddings = np.array(request.prev_embeds) if request.prev_embeds else np.zeros((0, 384))
    points = np.array(request.prev_points) if request.prev_points else np.zeros((0, 3))

    # New embedding
    new_embedding = model.encode([note.text])  # shape (1, 384)
    all_embeddings = np.vstack([embeddings, new_embedding]) if embeddings.size else new_embedding

    log.info(f"total embeddings: {all_embeddings.shape[0]}")

    # Projection + (optional) alignment
    if all_embeddings.shape[0] < 3:
        log.info("Only one or two embeddings, using first 3 dims and normalizing to unit sphere.")
        truncated = all_embeddings[:, :3]
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        normalized = np.divide(truncated, norms, out=np.zeros_like(truncated), where=norms != 0)
        aligned = normalized
        all_ids = request.prev_ids + [note.id]
    else:
        log.info("3+ embeddings, performing PCA to 3D.")
        new_projected = pca.fit_transform(all_embeddings)

        if points.shape[0] < 2:
            log.info("Not enough previous points to align, using PCA-projected points directly.")
            aligned = new_projected
            all_ids = request.prev_ids + [note.id]
        else:
            log.info("Aligning projected points to previous points via Orthogonal Procrustes.")
            X = points                  # previous 3D points (n-1, 3) expected
            Y = new_projected[:-1]      # new projection for previous items (n-1, 3)

            # Center both sets
            Xc = X - X.mean(axis=0)
            Yc = Y - Y.mean(axis=0)

            # Find rotation R so Yc @ R ≈ Xc
            R, _ = orthogonal_procrustes(Yc, Xc)
            Y_aligned = Yc @ R + X.mean(axis=0)

            # Align last point (the newly added note)
            last_point = new_projected[-1]
            last_point_aligned = (last_point - Y.mean(axis=0)) @ R + X.mean(axis=0)

            aligned = np.vstack([Y_aligned, last_point_aligned])
            all_ids = request.prev_ids + [note.id]

    log.info(f"Aligned points shape: {aligned.shape}")

    # Normalize to unit sphere
    norms = np.linalg.norm(aligned, axis=1, keepdims=True)
    aligned = np.divide(aligned, norms, out=np.zeros_like(aligned), where=norms != 0)

    embeddings_list = all_embeddings.tolist()
    points_list = [
        PointResponse(id=id_, x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
        for id_, pt in zip(all_ids, aligned)
    ]

    log.info(f"Returning # of points: {len(points_list)} | # of embeddings: {len(embeddings_list)}")
    return AddNoteResponse(points=points_list, embeddings=embeddings_list)


# ---------- Routes (async -> thread pool) ----------
@app.post("/add_note", response_model=AddNoteResponse)
async def add_note(request: AddNoteRequest):
    # Offload heavy CPU-bound work to a worker thread so the event loop stays responsive.
    return await run_in_threadpool(process_add_note, request)


# Optional: handle accidental double-slash path that the frontend might hit on retries.
@app.post("//add_note", response_model=AddNoteResponse)
async def add_note_double_slash(request: AddNoteRequest):
    return await add_note(request)


# ---------- Main ----------
if __name__ == "__main__":
    print("starting server...")
    port = int(os.environ.get("PORT", 8080))
    # Note: multiple workers load multiple copies of the model; keep workers=1 unless you have RAM to spare.
    uvicorn.run(app, host="0.0.0.0", port=port)
