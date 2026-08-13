
# Evolution

This project continues the earlier bird_label work, taking as its input the labelling
results that project produced. Development is on `main`; the `cluster` branch it grew on
was merged into it on 2026-08-12.

The code base under code/ will be removed with the project going. Particularly, those immediately under code/ will all be moved to code/bird_label/ upon the first project review (supportive, can be ignore if it doesn't deem useful).

# Theory
The goal and underlying theory is outlined in the document "./research/Bird Semantic Study Plan.md". 

# Quickstart

## Before you start: a Hugging Face token

You need one, and it takes a few minutes to arrange, so do it first. The
embedding step uses **DINOv3, which is a gated model** — it cannot be downloaded
anonymously.

1. Accept the licence **for your own account**:
   <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
2. Create a token: <https://huggingface.co/settings/tokens> (read access is enough)
3. After cloning, put it in `.env`:

       cp _env .env
       echo "export HF_TOKEN=hf_..." >> .env

Step 1 is the one people skip, and it fails confusingly: with a valid token but
no accepted licence the model *page* still resolves and `model_info()` succeeds,
while the file download returns 403 — so it reads like a network fault rather
than a permission. Only the first download needs the token; after that the model
is cached.

## Then

    git clone <this repo> && cd bird_cluster
    # set up .env as above
    ./run-all

`run-all` is the whole pipeline — label, embed, cluster, plot — on a small
sample dataset. On a fresh clone it creates the virtualenv and fetches the
sample itself. It will not start the two model servers for you, because
starting a GPU process is a decision about a machine rather than a step in a
demo:

    ./server-vllm       # labelling backend, in docker
    ./server-embed      # DINOv3 embedder; --device cpu if the GPU is busy

`--no-setup` reports what is missing without fixing anything. Results land in
`output/`, one subdirectory per stage.

On a DGX Spark this is genuinely all of it: `config.toml` already points at
`localhost` for both servers. Other hosts need the compose file adjusted —
`docker.compose/docker-compose.spark.yaml` pins an ARM/GB10 image — and their
own hosts named in `config.local.toml`.

# Data

The code ships without photographs. Pick one of three; nothing needs editing for
the first two.

**1. The sample (default).** A small tree that exercises the awkward cases —
images with no sidecar, Lightroom virtual copies, an `&` in a folder name.
Clone it into `sample_data/`, which is gitignored so it stays a separate repo:

    git clone https://github.com/mileszhou/sample_data.git sample_data

With nothing else set up, every tool finds it. Labels have been stripped from
its sidecars on purpose: the sample is the pipeline's *input*, so `./run-label`
has something to actually do. Run labelling before embedding — `./run-embed`
reads a labelling run's CSV and will tell you so if it is missing.

**2. The private library.** `data/` is a submodule and needs access:

    git submodule update --init

**3. Your own photos, anywhere.** Copy the template and set one line:

    cp _config.local.toml config.local.toml     # the copy is gitignored
    # data_dir = "/mnt/photos/library"

Whatever you name must contain a `jpg/` directory — that is the only test, and
it is the same test for all three. `$BIRD_DATA_DIR` does the same for a one-off,
and `--data-dir` overrides everything for a single run.

Resolution order: `--data-dir`, `$BIRD_DATA_DIR`, `config.local.toml`, `./data`,
`./sample_data`. **A `data_dir` you name is binding** — if it does not resolve
the run stops rather than falling back, so a typo cannot silently point a run at
a different population. Details in `docs/design/dataset-resolution.md`.

One thing to avoid: do not make `./data` a git repository of your own. A plain
directory there is invisible to git, but a repo is read as the submodule at the
wrong commit and leaves your tree permanently dirty. Keep your own versioned
dataset elsewhere and name it as `data_dir` in `config.local.toml`.

# Servers

Two model servers run as separate processes, and the tools reach them over HTTP:
vLLM for labelling, and the DINOv3 embedder. `config.toml` points at
`localhost` for both, so if you run them on the same machine nothing needs
configuring.

If yours are elsewhere, set them in the same `config.local.toml` — it is
gitignored and layered over `config.toml` recursively, so naming just a host
keeps the port. `--vllm-url` and `--embed-url` override
either for a single run.

    ./server-embed              # the embedder; --device cpu if the GPU is busy

The vLLM server is not started by this project; run it however you normally do,
and `run-label` will probe it for the model it is actually serving.

# Licence

Two works, two licences, because code and photographs want opposite terms.

**The code is MIT** (`LICENSE`) — use it, change it, ship it.

**The sample photographs are not.** They live in their own repository with
their own `LICENSE`, and are provided to run and understand this project, not
to be republished or used as model training data. Asymmetry like this is
ordinary for a dataset shipped alongside code; the restrictive half is a
statement of what the photographers are willing to have happen, and the images
are deliberately small (1024 px) so that little rides on it.

The private `data/` submodule is not published at all.

# Project structure
* first level: architectural components, like code, docks, and so on.
* the subfolders are organized according to the domain subjects, like embedding, cluster, grouping, etc.

Folder structure:
    * $var    means somestring maining *var*; could be var as used in script; This convention will be       broadly used in documentation.

cluster/ (marked as .)
    code/       # program code, mainly in python with others allowed distinguished just by file type
        lib/        # library like code, being called, never manually run
    test/       # any test code, run at this folder
        $domain_subfolders/
        lib/        # library lide code used by test
    project/    # everything about the work in progress -- not a deliverable
        plans/      # design documents, dated
        status/     # numbered session handoffs, so the next session can pick up
        messages/   # correspondence with the user, YYYY-MM-DD.NN who:-topic.md
        reports/    # analysis output. The report is tracked under a fixed
                    # filename so successive runs diff cleanly; the worklist
                    # csvs beside it are gitignored (large, fully regenerated)
            archive/    # numbered snapshots, ./tool-audit --snapshot $label
    docs/       # product documentation: output for whoever uses the result,
                # rather than notes about building it
    research/   # theoretical and research notes, mainly above software implimentation layer
    output/     # volitile output folder. Useful contents shall be copied to restuls/ manually
    restuls/    # the usefule outputs + manual organization
    thirdparty/ # any code/library/resources from thirdparty
    tools/      # mainly used by project developers; tools used by final users shall go code/ or ./

files_at_./
run-$func       # +x, $func is the domain function abbriviation
venv            # +x, establish .venv the first tiem upon git clone
README.md       # this file
CLAUDE.md       # claude memory about the project
requirements.txt    # .venv spec, generated when .venv is stablized
