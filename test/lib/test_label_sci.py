"""`label_sci`: the binomial, because it is the checklist's strong key.

19.6% of bird images fell back to a pixel guess for their taxonomy -- not
because the labels were odd, but because `great crested grebe`, `eurasian
hoopoe`, `white wagtail` and `common moorhen` have no entry in the checklist's
*common-name* index while every one of them is present by binomial. Asking the
labeller for a binomial matches on the complete side of the checklist.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PROJECT_ROOT", str(ROOT))

from code.bird_label import CSV_COLUMNS, VLLM_SYSTEM_PROMPT, normalise_binomial  # noqa: E402


def test_the_prompt_asks_for_it():
    assert "label_sci" in VLLM_SYSTEM_PROMPT
    assert "binomial" in VLLM_SYSTEM_PROMPT.lower()


def test_the_column_exists_and_sits_beside_the_other_names():
    assert "label_sci" in CSV_COLUMNS
    assert CSV_COLUMNS.index("label_sci") == CSV_COLUMNS.index("label_cn") + 1


def test_a_binomial_is_normalised_to_one_form():
    assert normalise_binomial("motacilla cinerea") == "Motacilla cinerea"
    assert normalise_binomial("Motacilla cinerea") == "Motacilla cinerea"
    # a trinomial keeps the species; an author citation is dropped
    assert normalise_binomial("Motacilla alba alba") == "Motacilla alba"
    assert normalise_binomial("Motacilla cinerea Tunstall, 1771") == "Motacilla cinerea"


def test_anything_of_the_wrong_shape_is_refused():
    """Shape only -- two Latin words of three letters or more, genus capitalised.

    Existence is checked against the checklist in `tools.map_label_taxa`, and
    that layering is what makes a wrong binomial cheap: one the checklist does
    not carry fails to match and falls back to the common name, exactly as a run
    with no label_sci does. A wrong *common* name has no such backstop.
    """
    for bad in ("Motacilla", "", "灰鹡鸰", "Motacilla Cinerea", "a bird", "of it"):
        assert normalise_binomial(bad) == "", bad


def test_a_shapely_but_nonexistent_binomial_is_harmless():
    """It normalises, then matches nothing and falls through to the name."""
    assert normalise_binomial("Notagenus notaspecies") == "Notagenus notaspecies"
