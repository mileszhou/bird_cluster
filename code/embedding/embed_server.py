"""Embedding server -- runs on the GPU host.

Mirrors the client/server split `bird_label.py` already uses for vLLM: the model
is loaded once here, and the pipeline code elsewhere is a thin HTTP client. Host
and port default to `config.toml` `[servers.embed]`.

    ./server-embed                           # or:
    python3 -m code.embedding.embed_server --model facebook/dinov3-vitb16-pretrain-lvd1689m
    python3 -m code.embedding.embed_server --image-size 1024
    MODEL=hf-hub:imageomics/bioclip-2.5-vith14 ./server-embed

Two backbones, because they are two different instruments and the project wants
both. **DINOv3 is self-supervised**: it never saw a species name, which is what
makes it a referee neither labeller authored. **BioCLIP is supervised** on
Linnaean names over TreeOfLife-200M, so it embeds taxonomy directly -- better at
species by construction, and for exactly that reason not an impartial judge of a
labelling. Where the two disagree is the interesting set, not which one scores
higher.

The backend follows from the model id (`hf-hub:` or `bioclip` -> open_clip,
otherwise transformers) and is recorded in /health, so a vector set says which
loader produced it. `--backend` overrides.

Endpoints:
    GET  /health -> {"status": "ok", "model": ..., "device": ..., "dim": ...,
                     "image_size": "224x224", ...}
    POST /embed  -> {"images": ["<base64 jpeg>", ...]} -> {"embeddings": [[...]], "dim": D}

Embeddings are L2-normalised before being returned, so cosine distance equals
euclidean distance downstream -- which is what the HDBSCAN step assumes.

DINOv3 weights are gated on Hugging Face; the first run needs HF_TOKEN in the
environment with the model licence accepted. BioCLIP is MIT and ungated.
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
BIOCLIP_MODEL = "hf-hub:imageomics/bioclip-2.5-vith14"


def pick_backend(model_id: str) -> str:
    """Which loader a model id needs.

    open_clip models are published as `hf-hub:<repo>` and cannot be reached
    through `AutoModel`; transformers models can. Inferring it from the id keeps
    the backend out of the command line for the two models actually used, which
    is the same reason `--approach` defaults rather than being required.
    """
    if model_id.startswith("hf-hub:") or "clip" in model_id.lower():
        return "open_clip"
    return "hf"


def squash_transform(preprocess, size: int):
    """The model's own preprocess, with the centre crop replaced by a squash.

    open_clip's transform resizes the *shortest edge* and then centre-crops, so a
    3:2 frame loses its outer thirds before the backbone sees it -- and a bird is
    frequently not in the middle third. DINOv3's processor squashes the whole
    frame to a square instead, losing aspect ratio but no content.

    Neither is obviously right: the crop is what BioCLIP was trained on, the
    squash keeps the bird and matches the geometry every existing vector in this
    project was computed under. So it is a recorded flag rather than a silent
    default. Only the resize is replaced; the normalisation the weights expect is
    reused untouched.
    """
    import torchvision.transforms as T

    tail = [t for t in preprocess.transforms
            if not isinstance(t, (T.Resize, T.CenterCrop))]
    return T.Compose([T.Resize((size, size),
                               interpolation=T.InterpolationMode.BICUBIC,
                               antialias=True)] + tail)


def transform_label(preprocess, resize_mode: str) -> str:
    """A short canonical name for what an open_clip transform feeds the model.

    Same job as `size_label` for a transformers processor: this is provenance,
    and the resize mode belongs in it because two runs at "224" that crop and
    squash respectively are not in the same space.
    """
    import torchvision.transforms as T

    size = None
    for t in preprocess.transforms:
        if isinstance(t, T.Resize):
            size = t.size
        elif isinstance(t, T.CenterCrop):
            size = t.size
    if isinstance(size, (list, tuple)):
        h, w = size[0], size[-1]
    else:
        h = w = size
    return f"{h}x{w}/{resize_mode}"


def hub_revision(model_id: str) -> str | None:
    """The commit the weights actually came from, for an `hf-hub:` model id.

    transformers records this on the config; open_clip does not keep it, so ask
    the Hub. Same reason as there: a repo name is a moving target, and a vector
    set that cannot name its weights cannot be reproduced. Best effort -- an
    offline load is still a valid load, it just records nothing.
    """
    repo = model_id.split("hf-hub:", 1)[-1]
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo).sha
    except Exception as exc:  # offline, rate-limited, private -- none are fatal
        logger.warning(f"could not resolve revision for {repo}: {exc}")
        return None


def size_label(processor) -> str:
    """A short canonical name for the resolution the processor feeds the model.

    This is provenance, not decoration. `AutoImageProcessor` carries a resize in
    its own config and applies it silently -- for DINOv3 ViT-B/16 that is
    224x224 -- so a 1024px JPEG reaches the backbone as 196 patches unless the
    size is overridden. Vectors from two resolutions are no more comparable than
    vectors from two backbones, and until this was recorded nothing in a run
    could say which resolution produced it.

    Both shapes the config uses are handled: an explicit height/width, and a
    `shortest_edge` with a separate centre crop.
    """
    def hw(d):
        if "height" in d and "width" in d:
            return f"{d['height']}x{d['width']}"
        if "shortest_edge" in d:
            return f"se{d['shortest_edge']}"
        return ""

    size = dict(getattr(processor, "size", None) or {})
    crop = dict(getattr(processor, "crop_size", None) or {})
    shown, cropped = hw(size), hw(crop)
    if cropped and cropped != shown:
        return f"{shown}->crop{cropped}"
    return shown or "unknown"


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("embed_server")

app = FastAPI(title="bird_cluster embed server")
STATE: dict = {"model": None, "processor": None, "model_id": None, "device": None,
               "dim": None, "revision": None, "dtype": None, "gpu": None,
               "versions": None, "image_size": None, "backend": None,
               "resize_mode": None, "tokenizer": None}


class EmbedRequest(BaseModel):
    images: list[str]  # base64-encoded JPEG bytes


class TextRequest(BaseModel):
    texts: list[str]


def load_model(model_id: str, device: str = "auto", image_size: int | None = None,
               backend: str = "auto", resize_mode: str = "crop"):
    """Load the backbone onto `device`, at `image_size` if one is asked for.

    `image_size` overrides the resize baked into the processor's own config.
    Left alone, DINOv3 ViT-B/16 resizes everything to 224x224 -- so a 1024px
    export reaches the backbone as a 14x14 grid, 196 patches, and the detail
    that separates two warblers is gone before the model sees it. The whole
    first embedding run of this project went that way without recording it.

    One server serves one resolution, deliberately, the same way it serves one
    backbone: the resolution is a property of the vector set, so letting a
    caller vary it per request would let one file hold vectors that cannot be
    compared. To change it, restart the server.

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
    if backend == "auto":
        backend = pick_backend(model_id)
    logger.info(f"loading {model_id} on {device} via {backend}")
    versions = {"torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": (torch.version.cuda if device == "cuda" else None),
                "python": platform.python_version()}

    if backend == "open_clip":
        import open_clip

        # `force_image_size` interpolates the position embeddings, the same trick
        # `--image-size` plays on DINOv3. It is a bigger ask here: BioCLIP is a
        # supervised contrastive model trained at exactly 224, so moving off that
        # resolution is a thing to measure, not to assume helps.
        extra = {"force_image_size": image_size} if image_size else {}
        model, _, preprocess = open_clip.create_model_and_transforms(model_id, **extra)
        model = model.to(device).eval()
        if resize_mode == "squash":
            preprocess = squash_transform(preprocess, image_size or 224)
        processor = preprocess
        STATE["tokenizer"] = open_clip.get_tokenizer(model_id)
        dim = getattr(getattr(model, "visual", None), "output_dim", None)
        image_size_label = transform_label(preprocess, resize_mode)
        versions["open_clip"] = open_clip.__version__
        revision = hub_revision(model_id)
    else:
        override = {}
        if image_size:
            square = {"height": image_size, "width": image_size}
            # crop_size too: a config that resizes the shortest edge and then centre
            # crops would otherwise undo the resize we just asked for.
            override = {"size": square, "crop_size": square}
        processor = AutoImageProcessor.from_pretrained(model_id, **override)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        dim = getattr(model.config, "hidden_size", None)
        image_size_label = size_label(processor)
        # The exact weights, not just the repo name: a model id is a moving target
        # on the Hub, and `main` today is not necessarily `main` next year.
        revision = getattr(getattr(model, "config", None), "_commit_hash", None)
        resize_mode = "squash"  # what AutoImageProcessor does with a square size

    STATE.update(
        model=model, processor=processor, model_id=model_id, device=device, dim=dim,
        revision=revision, image_size=image_size_label, backend=backend,
        resize_mode=resize_mode,
        dtype=str(next(model.parameters()).dtype),
        gpu=(torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()),
        versions=versions,
    )
    logger.info(f"loaded {model_id} (backend={backend}, dim={dim}, "
                f"image_size={STATE['image_size']}, dtype={STATE['dtype']}, "
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
            "image_size": STATE["image_size"],
            "backend": STATE["backend"], "resize_mode": STATE["resize_mode"],
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
        if STATE["backend"] == "open_clip":
            # open_clip's transform takes one PIL image at a time and the image
            # tower has its own entry point -- there is no `**batch` call here.
            batch = torch.stack([processor(img) for img in images]).to(device)
            feats = model.encode_image(batch)
        else:
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


@app.post("/embed_text")
def embed_text(req: TextRequest):
    """Embed label text into the *same* space as /embed, for zero-shot use.

    This is what makes a CLIP backbone more than an embedder: encode a candidate
    taxon name here, cosine it against a stored image vector, and you have a
    classification without a second pass over any pixels. The 27k image vectors
    this project already holds are exactly `encode_image` output, so the whole
    library can be classified against a new checklist for the cost of one matmul.

    Only the open_clip backbone has a text tower. DINOv3 is vision-only -- it was
    never trained against language and has nothing to encode text with -- so this
    refuses rather than inventing a vector, which would silently produce
    plausible nonsense downstream.

    L2-normalised like /embed, and for the same reason. Callers that ensemble
    several prompt templates per class must average the normalised vectors and
    then renormalise; that is the caller's job because the choice of templates
    belongs to the study, not to the server.
    """
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if STATE["backend"] != "open_clip":
        raise HTTPException(
            status_code=501,
            detail=f"{STATE['model_id']} has no text tower ({STATE['backend']} "
                   f"backend); text embedding needs a CLIP-style model")
    if not req.texts:
        return {"embeddings": [], "dim": STATE["dim"]}

    model, device, tokenizer = STATE["model"], STATE["device"], STATE["tokenizer"]
    with torch.inference_mode():
        tokens = tokenizer(req.texts).to(device)
        feats = model.encode_text(tokens)
        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
    return {"embeddings": feats.cpu().tolist(), "dim": feats.shape[1]}


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
    ap.add_argument("--image-size", type=int, default=None,
                    help="square resolution to feed the backbone. Default: whatever "
                         "the processor config says, which for DINOv3 ViT-B/16 is 224 "
                         "and for BioCLIP is 224. One server serves one resolution; "
                         "restart to change it")
    ap.add_argument("--backend", default="auto", choices=("auto", "hf", "open_clip"),
                    help="which loader. auto reads it off the model id: `hf-hub:` or "
                         "a name containing `clip` needs open_clip, everything else "
                         "goes through transformers")
    ap.add_argument("--resize-mode", default="crop", choices=("crop", "squash"),
                    help="open_clip only. crop is what BioCLIP was trained on "
                         "(shortest edge, then centre crop) and drops the outer "
                         "thirds of a 3:2 frame; squash keeps the whole frame at the "
                         "cost of aspect ratio, which is what every DINOv3 vector "
                         "here was computed under")
    args = ap.parse_args()

    load_model(args.model, args.device, args.image_size, args.backend, args.resize_mode)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
