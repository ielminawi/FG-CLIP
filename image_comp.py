import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

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

def image_to_feature(path, model, image_processor, device):
    image = Image.open(path).convert("RGB")
    
    # same logic as README, but simplified
    image_input = image_processor(
        images=image,
        max_num_patches=1024,   # you can keep their determine_max_value if you want
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        feat = model.get_image_features(**image_input)
        feat = F.normalize(feat, dim=-1)
    return feat

def compare_images(img1, img2):
    model, image_processor, device = load_model()
    f1 = image_to_feature(img1, model, image_processor, device)
    f2 = image_to_feature(img2, model, image_processor, device)

    # cosine similarity
    sim = (f1 @ f2.T).item()
    return sim

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compare_images.py path/to/img1 path/to/img2")
        exit(1)
    img1, img2 = sys.argv[1], sys.argv[2]
    similarity = compare_images(img1, img2)
    print(f"Similarity: {similarity:.4f}")
