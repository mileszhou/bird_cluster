
# Evolution

This project (branch cluster) is a continuation of the bird_label project. It uses as the input the results/ organized from the output/ of the bird_label (branche bird_label). 

The code base under code/ will be removed with the project going. Particularly, those immediately under code/ will all be moved to code/bird_label/ upon the first project review (supportive, can be ignore if it doesn't deem useful).

# Theory
The goal and underlying theory is outlined in the document "./research/Bird Semantic Study Plan.md". 

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
            archive/    # numbered snapshots, ./run-audit --snapshot $label
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
