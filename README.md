
# Evolution

This project (branch cluster) is a continuation of the bird_label project. It uses as the input the results/ organized from the output/ of the bird_label (branche bird_label). 

The code base under code/ will be removed with the project going. Particularly, those immediately under code/ will all be moved to code/bird_label/ upon the first project review (supportive, can be ignore if it doesn't deem useful).

# Theory
The goal and underlying theory is outlined in the document "./research/Bird Semantic Study Plan.md". 

# Data

The code ships without photographs. Pick one of three; nothing needs editing for
the first two.

**1. The sample (default).** A small tree that exercises the awkward cases —
images with no sidecar, Lightroom virtual copies, an `&` in a folder name.
Clone it into `sample_data/`, which is gitignored so it stays a separate repo:

    git clone <SAMPLE-DATA-REPO-URL> sample_data     # TODO: fill in once published

With nothing else set up, every tool finds it. Labels have been stripped from
its sidecars on purpose: the sample is the pipeline's *input*, so `./run-label`
has something to actually do. Run labelling before embedding — `./run-embed`
reads a labelling run's CSV and will tell you so if it is missing.

**2. The private library.** `data/` is a submodule and needs access:

    git submodule update --init

**3. Your own photos, anywhere.** Copy the template and edit your copy:

    cp _datapath .datapath          # .datapath is gitignored; _datapath is not

Put one path in it. Whatever you point at must contain a `jpg/` directory —
that is the only test, and it is the same test for all three. `$BIRD_DATA_DIR`
does the same thing for a one-off, and `--data-dir` overrides everything for a
single run.

Resolution order: `--data-dir`, `$BIRD_DATA_DIR`, `.datapath`, `./data`,
`./sample_data`. **A path in `.datapath` is binding** — if it does not resolve
the run stops rather than falling back, so a typo cannot silently point a run at
a different population. Details and the reasoning in
`docs/design/dataset-resolution.md`.

One thing to avoid: do not make `./data` a git repository of your own. A plain
directory there is invisible to git, but a repo is read as the submodule at the
wrong commit and leaves your tree permanently dirty. Keep your own versioned
dataset elsewhere and name it in `.datapath`.

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
