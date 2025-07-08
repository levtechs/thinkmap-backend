from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import uvicorn
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

app = FastAPI()

# Allow more origins for production deployment
allowed_origins = [
    "http://localhost:3001",
    "https://localhost:3001",
    # Add your production frontend URLs here
    "https://*.run.app",  # Cloud Run default domains
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('all-MiniLM-L6-v2')
pca = PCA(n_components=3)

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

@app.post("/add_note", response_model=AddNoteResponse)
def add_note(request: AddNoteRequest):
    print("got request:")
    note = request.note
    print("note:", note)
    embeddings = np.array(request.prev_embeds) if request.prev_embeds else np.zeros((0, 384))
    points = np.array(request.prev_points) if request.prev_points else np.zeros((0, 3))

    new_embedding = model.encode([note.text])
    all_embeddings = np.vstack([embeddings, new_embedding]) if embeddings.size else new_embedding

    print("total embeddings:", all_embeddings.shape[0])

    if all_embeddings.shape[0] < 3:
        print("Only one or 2 embeddings, using first 3 dims")

        truncated = all_embeddings[:, :3]
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        normalized = np.divide(truncated, norms, out=np.zeros_like(truncated), where=norms != 0)

        aligned = normalized
        all_ids = request.prev_ids + [note.id]
    else:
        print("2 or more embeddings, performing PCA")
        new_projected = pca.fit_transform(all_embeddings)

        if points.shape[0] < 2:
            print("Not enough previous points to align, using new projected points")
            aligned = new_projected
            all_ids = request.prev_ids + [note.id]
        else:
            print("Aligning new projected points with previous points")
            X = points
            Y = new_projected[:-1]

            Xc = X - X.mean(axis=0)
            Yc = Y - Y.mean(axis=0)

            R, _ = orthogonal_procrustes(Yc, Xc)
            Y_aligned = Yc @ R + X.mean(axis=0)

            last_point = new_projected[-1]
            last_point_aligned = (last_point - Y.mean(axis=0)) @ R + X.mean(axis=0)

            aligned = np.vstack([Y_aligned, last_point_aligned])
            all_ids = request.prev_ids + [note.id]

    print("Aligned points shape, sending to frontend:", aligned.shape)

    # Normalize points to lie on unit sphere
    norms = np.linalg.norm(aligned, axis=1, keepdims=True)
    aligned = np.divide(aligned, norms, out=np.zeros_like(aligned), where=norms != 0)

    embeddings_list = all_embeddings.tolist()
    points_list = [
        PointResponse(id=id_, x=pt[0], y=pt[1], z=pt[2])
        for id_, pt in zip(all_ids, aligned)
    ]

    print("Returning # of points:", len(points_list))
    print("Returning # of embeddings:", len(embeddings_list))

    return AddNoteResponse(points=points_list, embeddings=embeddings_list)


if __name__ == "__main__":
    # Cloud Run sets the PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)