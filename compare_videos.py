import cv2
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForCausalLM

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_model(model_root="qihoo360/fg-clip2-base"):
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True).to(device)
    image_processor = AutoImageProcessor.from_pretrained(model_root)
    return model, image_processor, device

def sample_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length == 0:
        cap.release()
        return frames

    # pick evenly spaced indices
    indices = torch.linspace(0, length - 1, steps=num_frames).long().tolist()

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def video_to_feature(video_path, model, image_processor, device, num_frames=8):
    frames = sample_frames(video_path, num_frames=num_frames)
    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path}")

    # batch process frames
    inputs = image_processor(
        images=frames,
        max_num_patches=1024,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        feats = model.get_image_features(**inputs)      # shape [num_frames, dim]
        feats = F.normalize(feats, dim=-1)
        video_feat = feats.mean(dim=0, keepdim=True)    # [1, dim]
        video_feat = F.normalize(video_feat, dim=-1)
    return video_feat

def compare_videos(v1, v2):
    model, image_processor, device = load_model()
    f1 = video_to_feature(v1, model, image_processor, device)
    f2 = video_to_feature(v2, model, image_processor, device)
    sim = (f1 @ f2.T).item()
    return sim

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compare_videos.py path/to/video1.mp4 path/to/video2.mp4")
        exit(1)
    v1, v2 = sys.argv[1], sys.argv[2]
    similarity = compare_videos(v1, v2)
    print(f"Video similarity: {similarity:.4f}")
