from collections import Counter
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from sklearn.cluster import KMeans

app = FastAPI(title="AI Image Palette Generator")
INDEX_FILE = Path(__file__).with_name("index.html")


@app.get("/")
async def home():
    return FileResponse(INDEX_FILE)


@app.post("/analyze")
async def analyze_image(file: UploadFile, colors: int = 5):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=400,
            detail="Upload a PNG or JPEG image.",
        )

    try:
        with Image.open(file.file) as image:
            image = image.convert("RGB")
            image.thumbnail((200, 200))
            pixels = np.asarray(image).reshape(-1, 3)
    except OSError as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as an image.",
        ) from error
    
    requested_colors = max(2, min(colors, 10))
    unique_color_count = len({tuple(pixel) for pixel in pixels})
    cluster_count = min(requested_colors, unique_color_count)

    model = KMeans(
        n_clusters=cluster_count,
        n_init="auto",
        random_state=42,
    )
    labels = model.fit_predict(pixels)
    label_counts = Counter(int(label) for label in labels)
    ordered_labels = sorted(
        label_counts,
        key=lambda label: label_counts[label],
        reverse=True,
    )

    palette = []
    for label in ordered_labels:
        center = model.cluster_centers_[label]
        red, green, blue = (int(round(value)) for value in center)
        palette.append(
            {
                "hex": f"#{red:02X}{green:02X}{blue:02X}",
                "rgb": [red, green, blue],
                "percentage": round(label_counts[label] / len(labels) * 100, 1),
            }
        )

    return {"filename": file.filename, "colors": palette}
    