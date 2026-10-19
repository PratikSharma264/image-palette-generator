import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="AI Image Palette Generator")


@app.post("/analyze")
async def analyze_image(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=400,
            detail="Upload a PNG or JPEG image.",
        )