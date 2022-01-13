import numpy as np
import scipy
import tensorflow as tf
from compressed_embeddings.datasets.distribution_constants import PMI_VALUES_HIST


def get_fast_random_pmi(size: int, sparcity: float = 1e-4):
    """
    Generates a random PMI matrix with a given sparcity.
    This method is fast (20s/GB).

    (!) the PMI matrix doesn't have real notions of similarity
    There won't be clusters of co-occurrences like in a real PMI matrix.

    Parameters
    ----------
    size: int
        Size of the matrix
    sparcity: float
        Sparcity/density of the matrix
    """

    # Define distribution functions
    rng = np.random.default_rng()
    rvs = lambda x: scipy.stats.rv_histogram(PMI_VALUES_HIST).rvs(size=x)

    # Generate a sparse matrix with random PMI values and half sparsity.
    part1 = scipy.sparse.random(size, size, density=sparcity / 2, random_state=rng, dtype=np.float32, data_rvs=rvs)

    # Transform to symetric matrix
    pmi = part1 + part1.transpose()

    return pmi


def gen_batch_from_dataset(
    features_dataset: np.ndarray,
    values_dataset: np.ndarray,
    batch_size: int,
    p_neg: float = 0,
    voc_count: int = None,
    drop_remainder: bool = True,
):
    """
    Batch generator with negative sampling

    Parameters
    ----------
    features_dataset: np.ndarray
        Array of shape (N, 2) with (row, col) pairs
    values_dataset: np.ndarray
        Array of shape (N,) with PMI values
    batch_size: int
        Size of the batches
    p_neg: float
        Proportion of negative samples
    voc_count: int
        Number of elements in the feature space
    drop_remainder: bool
        If True, the last batch will be dropped if it is smaller than batch_size
    """

    assert 0 <= p_neg <= 1
    assert 1 <= batch_size

    if p_neg:
        assert voc_count is not None

    size = features_dataset.shape[0]

    # x4 slower iteration with perm like that, but ok
    rand_perm = np.random.permutation(size)

    i = 0

    while i < size:

        neg_size = 0
        if p_neg:
            neg_size = np.random.binomial(batch_size, p_neg)
        pos_size = batch_size - neg_size
        if i + pos_size > size:
            pos_size -= i + pos_size - size
            if drop_remainder:
                break

        perm = rand_perm[i : i + pos_size]

        features = features_dataset[perm]
        values = values_dataset[perm]

        if neg_size:
            rand_indices = np.random.randint(0, voc_count, (neg_size, 2))
            features = np.concatenate((features, rand_indices))
            values = np.concatenate((values, np.zeros((neg_size, 1), dtype="float32")))

        yield tf.constant(features, dtype="int32"), tf.constant(values, dtype="int32")

        i += pos_size


def generate_batch_from_sparse_pmi(
    pmi: scipy.sparse.coo_matrix, batch_size: int, p_neg: float = 0, drop_remainder: bool = True
):
    """
    Batch generator with negative sampling

    Parameters
    ----------
    pmi: scipy.sparse.csr_matrix
        Sparse PMI matrix
    batch_size: int
        Size of the batches
    p_neg: float
        Proportion of negative samples
    drop_remainder: bool
        If True, the last batch will be dropped if it is smaller than batch_size
    """

    assert scipy.sparse.isspmatrix(pmi)
    features_dataset = pmi.nonzero()
    values_dataset = pmi.data
    voc_count = pmi.shape[0]
    return gen_batch_from_dataset(features_dataset, values_dataset, batch_size, p_neg, voc_count, drop_remainder)
