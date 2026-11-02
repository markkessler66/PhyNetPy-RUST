"""Partition rules: forbidden edges, sizes, mega-blobs, and seeded divisions."""

from phynetpy.Network import Network

from phynetpy._dimple import partition_scaffold


def _subsets(job):
    return [subset for division in job.divisions for subset in division]


def test_a_small_blob_is_one_subset_and_is_not_pruned():
    job = partition_scaffold(Network.from_newick("(a,b,c,d);"), n_divisions=2, seed=0)[0]
    assert job.pruned == []
    assert job.divisions == [[frozenset("abcd")]]
    assert job.tob_leaves == frozenset("abcd")
    assert job.is_root


def test_a_root_polytomy_is_one_blob_on_all_of_its_leaves():
    job = partition_scaffold(Network.from_newick("(a,b,c,d);"))[0]
    assert len(partition_scaffold(Network.from_newick("(a,b,c,d);"))) == 1
    assert job.tob_leaves == frozenset("abcd")


def test_adjacent_polytomies_are_one_mega_blob():
    jobs = partition_scaffold(
        Network.from_newick("(OUT,(a,b,(c,d,e)));"),
        outgroup="OUT",
        n_divisions=1,
        seed=0,
    )
    assert len(jobs) == 1
    assert jobs[0].tob_leaves == frozenset("abcde")
    assert all("OUT" not in subset for subset in _subsets(jobs[0]))


def test_cuts_do_not_take_a_polytomys_leaf_children():
    # Three leaves hang directly on the polytomy. Two cuts of size 3 are
    # available deeper in the tree. S = 5, so a size-6 clade does not fit
    # and the prune has to take the deep triples instead of a blob edge.
    scaffold = Network.from_newick(
        "(a,b,c,(((d,(e,f)),(g,(h,i))),(j,(k,l))));"
    )
    job = partition_scaffold(scaffold, max_subset_size=5, n_divisions=1, seed=0)[0]
    pruned_leaves = [prune.leaves for prune in job.pruned]
    assert len(pruned_leaves) >= 2
    for leaves in pruned_leaves:
        assert len(leaves) >= 3
        assert len(leaves) <= 5
        assert not (leaves & set("abc"))
    # The second cut's siblings are the residual at that moment, so they
    # do not include the clade that was already removed.
    assert not (job.pruned[1].anchor_leaves & job.pruned[0].leaves)
    assert job.pruned[0].anchor_leaves.isdisjoint(job.pruned[0].leaves)
    for subset in _subsets(job):
        assert 3 <= len(subset) <= 5


def test_same_seed_repeats_and_extra_k_does_not_invent_duplicates():
    scaffold = Network.from_newick("(a,b,c,d,e,f);")
    first = partition_scaffold(scaffold, max_subset_size=4, n_divisions=3, seed=0)[0]
    again = partition_scaffold(scaffold, max_subset_size=4, n_divisions=3, seed=0)[0]
    other = partition_scaffold(scaffold, max_subset_size=4, n_divisions=1, seed=1)[0]
    assert first.divisions == again.divisions
    assert first.divisions[0] != other.divisions[0]
    many = partition_scaffold(scaffold, max_subset_size=4, n_divisions=100, seed=0)[0]
    keys = [frozenset(division) for division in many.divisions]
    assert len(keys) == len(set(keys))
    assert len(keys) < 100
    for division in many.divisions:
        for subset in division:
            assert 3 <= len(subset) <= 4


def test_the_first_division_is_dimples_smallest_first_packing():
    # Root polytomy with child clades of 4, 2, 2 and 3 leaves and no legal
    # prune (every deeper clade has 2 leaves). S = 6. Smallest-first fills
    # one bin with the two 2-leaf items, then packs the rest first-fit.
    scaffold = Network.from_newick("(((a,b),(c,d)),(e,f),(g,h),((i,j),k));")
    job = partition_scaffold(scaffold, max_subset_size=6, n_divisions=3, seed=0)[0]
    assert job.pruned == []
    assert set(job.divisions[0]) == {frozenset("efgh"), frozenset("ijk"), frozenset("abcd")}
    again = partition_scaffold(scaffold, max_subset_size=6, n_divisions=3, seed=7)[0]
    assert again.divisions[0] == job.divisions[0]
    assert len(job.divisions) == 3
    for division in job.divisions:
        assert set().union(*division) == set("abcdefghijk")
        for subset in division:
            assert 3 <= len(subset) <= 6


def test_a_blob_that_cannot_keep_every_subset_at_three_still_divides():
    # Five leaves on a polytomy with S = 4: every split has a 1- or 2-leaf
    # subset. DIMPLE relaxes the minimum to 1 instead of failing.
    job = partition_scaffold(Network.from_newick("(a,b,c,d,e);"), max_subset_size=4, n_divisions=2, seed=0)[0]
    assert job.divisions
    for division in job.divisions:
        assert set().union(*division) == set("abcde")
        assert all(len(subset) <= 4 for subset in division)
        assert any(len(subset) < 3 for subset in division)
