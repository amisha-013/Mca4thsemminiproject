from fastapi import APIRouter, UploadFile, File, Form, HTTPException,Request
import shutil
import os
import numpy as np
import cv2
from services import run_pipeline



router = APIRouter()
@router.get("/")
def root():
    return {"message": "ok"}

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # future: YOLO model call here
    yolo_result = "YOLO will process this image later"

    return {
        "message": "Image uploaded successfully",
        "file": file.filename,
        "yolo": yolo_result
    }


@router.post("/detect")
async def detect(request: Request, file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = request.app.state.model
    mp_face = request.app.state.mp_face
    mp_hands = request.app.state.mp_hands

    result = run_pipeline(img, model, mp_face, mp_hands)

    return result