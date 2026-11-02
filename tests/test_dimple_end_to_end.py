"""One blob, end to end, and the registry stays untouched."""

from phynetpy.Network import Network
from phynetpy.data import GeneTrees
from phynetpy.infer import infer_dimple, registered_cells, score
from phynetpy.criteria import PseudoLikelihood
from phynetpy.models import MSC

from dimple_fixtures import assert_phylogeny


def test_one_blob_infers_merges_and_grafts():
    scaffold = Network.from_newick("(a,b,c,d);")
    start = Network.from_newick("((a,b),(c,d));")
    genes = GeneTrees(trees=[
        Network.from_newick("((a,b),(c,d));"),
        Network.from_newick("((a,c),(b,d));"),
    ])
    result = infer_dimple(
        genes, scaffold, start,
        max_subset_size=12, n_divisions=1, max_reticulations=0,
        seed=0, num_iter=1,
    )
    assert result.method == "DIMPLE"
    assert_phylogeny(result.best, list("abcd"))
    trace = result.raw
    assert len(trace.blobs) == 1
    assert len(trace.blobs[0].subsets) == 1
    subset = trace.blobs[0].subsets[0]
    assert subset.score == score(
        subset.network, genes, model=MSC(), criterion=PseudoLikelihood(),
    )
    assert result.score == trace.score
    assert not any(cell[3] == "DIMPLE" for cell in registered_cells())
