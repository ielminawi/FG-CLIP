from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForCausalLM
import tempfile
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Comparison API")

# Add CORS middleware to allow requests from your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
_model_cache = {}
WEB_DIR = Path(__file__).parent / "web"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_model(model_root="qihoo360/fg-clip2-base"):
    global _model_cache
    
    if "model" not in _model_cache:
        logger.info(f"Loading model from {model_root}...")
        device = get_device()
        model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True).to(device)
        image_processor = AutoImageProcessor.from_pretrained(model_root)
        _model_cache["model"] = model
        _model_cache["image_processor"] = image_processor
        _model_cache["device"] = device
        logger.info(f"Model loaded on device: {device}")
    
    return _model_cache["model"], _model_cache["image_processor"], _model_cache["device"]

def sample_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if length == 0:
        cap.release()
        raise ValueError(f"No frames found in video: {video_path}")

    indices = torch.linspace(0, length - 1, steps=num_frames).long().tolist()

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Failed to extract frames from video: {video_path}")
    
    return frames

def video_to_feature(video_path, model, image_processor, device, num_frames=8):
    frames = sample_frames(video_path, num_frames=num_frames)

    inputs = image_processor(
        images=frames,
        max_num_patches=1024,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = F.normalize(feats, dim=-1)
        video_feat = feats.mean(dim=0, keepdim=True)
        video_feat = F.normalize(video_feat, dim=-1)
    return video_feat

@app.get("/")
async def serve_web():
    if WEB_DIR.exists():
        return FileResponse(WEB_DIR / "index.html")
    return {"status": "ok", "message": "Video Comparison API is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/compare")
async def compare_videos(video1: UploadFile = File(...), video2: UploadFile = File(...)):
    temp_files = []
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp1:
            content1 = await video1.read()
            tmp1.write(content1)
            tmp1_path = tmp1.name
            temp_files.append(tmp1_path)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp2:
            content2 = await video2.read()
            tmp2.write(content2)
            tmp2_path = tmp2.name
            temp_files.append(tmp2_path)
        
        logger.info(f"Processing videos: {tmp1_path}, {tmp2_path}")
        
        model, image_processor, device = load_model()
        
        f1 = video_to_feature(tmp1_path, model, image_processor, device)
        f2 = video_to_feature(tmp2_path, model, image_processor, device)
        
        similarity = (f1 @ f2.T).item()
        
        logger.info(f"Similarity score: {similarity:.4f}")
        
        return {
            "similarity": float(similarity),
            "video1": video1.filename,
            "video2": video2.filename
        }
    
    except Exception as e:
        logger.error(f"Error processing videos: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        for tmp_file in temp_files:
            try:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_file}: {e}")

# Serve additional static assets if needed
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
