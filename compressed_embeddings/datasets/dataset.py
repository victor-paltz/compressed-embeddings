import numpy as np
import scipy


from compressed_embeddings.datasets.distribution_constants import PMI_VALUES_HIST


def get_fast_random_pmi(size: int, sparcity=1e-2):
    """
    Generates a random PMI matrix with a given sparcity.
    This method is fast (20s/GB).

    (!) the PMI matrix doesn't have real notions of similarity
    There won't be clusters of co-occurrences like in a real PMI matrix.
    """

    # Define distribution functions
    rng = np.random.default_rng()
    rvs = lambda x: scipy.stats.rv_histogram(PMI_VALUES_HIST).rvs(size=x)

    # Generate a sparse matrix with random PMI values and half sparsity.
    part1 = scipy.sparse.random(size, size, density=sparcity / 2, random_state=rng, dtype=np.float32, data_rvs=rvs)

    # Transform to symetric matrix
    pmi = part1 + part1.transpose()

    return pmi
