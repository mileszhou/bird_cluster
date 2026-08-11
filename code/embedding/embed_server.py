"""DINOv3 embedding server -- runs on the GPU host.

Mirrors the client/server split `bird_label.py` already uses for vLLM: the model
is loaded once here, and the pipeline code elsewhere is a thin HTTP client. Host
and port default to `config.toml` `[servers.embed]`.

    ./run-server-embed                       # or:
    python3 -m code.embedding.embed_server --model facebook/dinov3-vitb16-pretrain-lvd1689m

Endpoints:
    GET  /health -> {"status": "ok", "model": ..., "device": ..., "dim": ...}
    POST /embed  -> {"images": ["<base64 jpeg>", ...]} -> {"embeddings": [[...]], "dim": D}

Embeddings are L2-normalised before being returned, so cosine distance equals
euclidean distance downstream -- which is what the HDBSCAN step assumes.

DINOv3 weights are gated on Hugging Face; the first run needs HF_TOKEN in the
environment with the model licence accepted.
"""

import argparse
import base64
import io
import logging

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModel

from code.lib.config import load_config

DEFAULT_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("embed_server")

app = FastAPI(title="bird_cluster embed server")
STATE: dict = {"model": None, "processor": None, "model_id": None, "device": None,
               "dim": None, "revision": None, "dtype": None, "gpu": None,
               "versions": None}


class EmbedRequest(BaseModel):
    images: list[str]  # base64-encoded JPEG bytes


def load_model(model_id: str, device: str = "auto"):
    """Load the backbone onto `device`.

    `auto` prefers CUDA. Naming a device explicitly matters because the GPU
    being *present* is not the same as it being *free*: a co-tenant inference
    server holding most of the memory makes `.to("cuda")` fail with a plain
    out-of-memory error after the weights have already loaded. ViT-B/16 is small
    enough to run on CPU, so `--device cpu` keeps a demo or a small job going
    without evicting whatever else is resident.

    Recorded in /health and copied into the client's run.json, because the
    device is part of a vector's provenance: CPU and GPU kernels agree to about
    the last decimal place, not exactly. Vectors from the two are close enough to
    compare and not identical, so a single embedding set should stick to one.
    """
    import platform
    import transformers

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"loading {model_id} on {device}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    dim = getattr(model.config, "hidden_size", None)

    # The exact weights, not just the repo name: a model id is a moving target
    # on the Hub, and `main` today is not necessarily `main` next year.
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    STATE.update(
        model=model, processor=processor, model_id=model_id, device=device, dim=dim,
        revision=revision,
        dtype=str(next(model.parameters()).dtype),
        gpu=(torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()),
        versions={"torch": torch.__version__,
                  "transformers": transformers.__version__,
                  "cuda": (torch.version.cuda if device == "cuda" else None),
                  "python": platform.python_version()},
    )
    logger.info(f"loaded {model_id} (dim={dim}, dtype={STATE['dtype']}, "
                f"revision={revision}) on {STATE['gpu']}")


@app.get("/health")
def health():
    """What is serving, and on what.

    The client copies this whole object into its `run.json`, so anything here
    becomes part of a vector set's provenance for free. The versions and the GPU
    are recorded because two runs of the "same model" are only comparable if the
    stack underneath was the same -- kernel selection and library versions move
    results at the last few decimal places, and a study of run-to-run variation
    cannot start from vectors that cannot say what produced them.
    """
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": STATE["model_id"],
            "device": STATE["device"], "dim": STATE["dim"],
            "revision": STATE["revision"], "dtype": STATE["dtype"],
            "gpu": STATE["gpu"], "versions": STATE["versions"]}


@app.post("/embed")
def embed(req: EmbedRequest):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.images:
        return {"embeddings": [], "dim": STATE["dim"]}

    try:
        images = [Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                  for b64 in req.images]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode images: {exc}")

    processor, model, device = STATE["processor"], STATE["model"], STATE["device"]
    with torch.inference_mode():
        batch = processor(images=images, return_tensors="pt").to(device)
        out = model(**batch)
        # Prefer the pooled/CLS representation; fall back to mean over patches.
        if getattr(out, "pooler_output", None) is not None:
            feats = out.pooler_output
        else:
            feats = out.last_hidden_state[:, 0]
        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)

    vectors = feats.cpu().tolist()
    STATE["dim"] = len(vectors[0])
    return {"embeddings": vectors, "dim": STATE["dim"]}


def main():
    servers = load_config().get("servers", {}).get("embed", {})
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address (config.toml's host names the client's target, "
                         "not necessarily a local bind address)")
    ap.add_argument("--port", type=int, default=servers.get("port", 9100))
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                    help="auto prefers CUDA. Use cpu when the GPU is present but "
                         "occupied -- ViT-B/16 runs fine there for small jobs")
    args = ap.parse_args()

    load_model(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
