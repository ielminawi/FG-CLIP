## Quick start (local demo)

1. In a terminal, go to the project: `cd FG-CLIP`.
2. Install dependencies (edit to match your env):
   ```bash
   pip install -e .
   pip install fastapi uvicorn opencv-python-headless
   ```
   Make sure PyTorch is installed with the accelerator you need (CPU/CUDA/MPS).
3. Run the API + web page:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
4. Open the site: http://localhost:8000. Pick two short videos and click “Run Comparison”.

## Notes
- The app will download the FG-CLIP model on first use; keep the process running and reuse for faster responses.
- If you host the API elsewhere, update `apiBase` in `web/index.html` to the full URL (e.g., `https://yourdomain.com`).
