from fastapi import APIRouter, UploadFile, File , Request
from typing import List
import os
import numpy as np
import cv2
import time
import shutil
from services import run_pipeline, get_image_score
from PIL import Image
import io


router = APIRouter()

UPLOAD_FOLDER = "backend/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.get("/")
def root():
    """
    Server test api
    """
    return {"message": "ok"}


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    """
    File upload test api
    
    """

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

    """
    Object detection test api
    """

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = request.app.state.model
    mp_face = request.app.state.mp_face
    mp_hands = request.app.state.mp_hands

    req_id = str(int(time.time()*1000))
    save_dir = os.path.join(UPLOAD_FOLDER, req_id)
    result = run_pipeline(img, model, mp_face, mp_hands,save_dir)

    return result


@router.post("/score")
async def score_images(
    request: Request,
    original: UploadFile = File(...),
    crops: List[UploadFile] = File(...)
):
    
    """
    Image score with efficientnet test api
    """


    try:
        eff_model = request.app.state.eff_model
        eff_transform = request.app.state.eff_transform
        device = request.app.state.device

        # -------- ORIGINAL IMAGE --------
        original_bytes = await original.read()
        original_image = Image.open(io.BytesIO(original_bytes)).convert("RGB")

        original_score = get_image_score(original_image, eff_model, eff_transform, device)

        # -------- CROPPED IMAGES --------
        crop_scores = []

        for crop in crops:
            crop_bytes = await crop.read()
            crop_image = Image.open(io.BytesIO(crop_bytes)).convert("RGB")

            score = get_image_score(crop_image, eff_model, eff_transform, device)
            crop_scores.append(score)

        # -------- AVERAGE --------
        avg_crop_score = sum(crop_scores) / len(crop_scores) if crop_scores else 0

        # -------- FINAL SCORE --------
        final_score = (0.6 * original_score) + (0.4 * avg_crop_score)

        return {
            "original_score": original_score,
            "crop_scores": crop_scores,
            "average_crop_score": avg_crop_score,
            "final_score": final_score
        }

    except Exception as e:
        return {"error": str(e)}