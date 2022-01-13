""""Same as tf.layers.Embedding but with a compressed embedding table"""


from typing import Tuple

import tensorflow as tf

from compressed_embeddings.models.utils.hashing_functions import hash_3


class CompressedEmbedding(tf.keras.layers.Layer):
    """
    This class is doing the same as layers.Embedding, but instead of
    Storing a 2D tensor of weights, the embedding table is compressed.
    This can save a lot of memory.
    """

    def __init__(self, nb_embeddings: int, embeddings_dim: int, mem: int, salt: int = None) -> None:
        super().__init__()

        self.nb_embeddings = nb_embeddings
        self.d = embeddings_dim
        self.mem = mem
        self.salt = 0 if salt is None else salt

        w_init = tf.random_normal_initializer()

        self.hash_function = tf.function(hash_3)
        self.mem_pool = tf.Variable(initial_value=w_init(shape=(mem,), dtype="float32"), trainable=True)

    def build(self, input_shape: Tuple) -> None:

        nb_dims = len(input_shape)

        # Precompute an offset
        # It is a trick to have the same vector for each indice with
        # a different memory mapping for each of the dimensions
        self.offset = tf.constant(
            tf.reshape(tf.range(0, self.d, dtype=tf.int32), shape=(1,) * nb_dims + (self.d,)), dtype=tf.int32,
        )

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        From an indice tensor of shape (*x.shape,), this function returns
        the corresponding d-dimensionnal embedding values of shape (*x.shape, d).
        """

        # Get d indices per embeddings, these indices indicate where to retrieve the
        # embedding value from the memory pool.
        # (*x.shape,) -> (*x.shape, d)
        embeddings_indices = self.hash_function(tf.expand_dims(x, -1) * self.d + self.offset, self.mem, self.salt)

        # Retrieve the embedding values from the indices
        # (*x.shape, d) -> (*x.shape, d)
        embeddings = tf.gather(self.mem_pool, embeddings_indices, False)

        return embeddings
