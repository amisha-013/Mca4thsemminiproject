from fastapi import APIRouter, UploadFile, File, Request, HTTPException
import os
import numpy as np
import cv2
import time
import shutil
from services import run_pipeline, get_image_score,generate_response,response_format
from PIL import Image
import io


router = APIRouter()

UPLOAD_FOLDER = "backend/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/ask")
async def analyze(request: Request,file: UploadFile = File(...)):

    contents = await file.read()
    req_id = str(int(time.time()*1000))
    save_dir = os.path.join(UPLOAD_FOLDER, req_id)
    os.makedirs(save_dir, exist_ok=True)

    """
    Uploading the main images and saving it
    """
    file_path = os.path.join(save_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    """
    Running the yolo model
    """
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = request.app.state.model
    mp_face = request.app.state.mp_face
    mp_hands = request.app.state.mp_hands
    result = run_pipeline(img, model, mp_face, mp_hands,save_dir)


    """
    Generation Effiecient Net score
    """

    eff_model = request.app.state.eff_model
    eff_transform = request.app.state.eff_transform
    device = request.app.state.device

    # -------- ORIGINAL IMAGE --------
    original_image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_score = get_image_score(original_image, eff_model, eff_transform, device)

    # -------- CROPPED IMAGES --------
    crop_scores = []
    for filename in os.listdir(save_dir):
        path = os.path.join(save_dir, filename)

        try:
            crop_image = Image.open(path).convert("RGB")  # 👈 only works if it's an image

            score = get_image_score(crop_image, eff_model, eff_transform, device)
            crop_scores.append(score)

        except Exception:
            continue   # 👈 skip non-image files silently

    
    # -------- AVERAGE --------
    avg_crop_score = sum(crop_scores) / len(crop_scores) if crop_scores else 0

    # -------- FINAL SCORE --------
    final_score = (0.6 * original_score) + (0.4 * avg_crop_score)

    final_score = round(final_score * 10, 1)

    """
    Generation LLM response
    """

    try:
        result = await generate_response(final_score, contents)
        result = response_format(result)
        return {
            "image_name": file.filename,
            "score": final_score,
            **result,
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

