"""
This file contains all the models that are tested in this repository.
"""
import tensorflow as tf
from compressed_embeddings.models.layers.CompressedEmbedding import CompressedEmbedding
from tensorflow.keras import layers


class ModelWithCompressedEmbeddings(tf.keras.Model):
    """
    Simple Glove-like model that generates embeddings in order
    to approximate a PMI matrix.

    The embeddings can be either stored in a compressed table or not.
    """

    def __init__(
        self,
        nb_embeddings: int,
        embeddings_dim: int,
        use_compressed_embeddings: bool,
        use_bias: bool = True,
        use_same_embeddings: bool = False,
        use_same_bias: bool = False,
        memory_dimension: int = None,
        salt: int = 0,
        p_dropout: float = 0.1,
    ):
        """
        Initialize the model.

        Parameters
        ----------
        nb_embeddings: int
            Number of embeddings to generate.
        embeddings_dim: int
            Dimension of the embeddings.
        use_compressed_embeddings: bool
            Whether to use a compressed embedding table or not.
            This is the whole point of that project.
        use_bias: bool
            Whether to use a bias or not.
        use_same_embeddings: bool
            Whether to use the same context and target embeddings or not.
            Using the same embeddings divides the model size by 2.
        use_same_bias: bool
            Whether to use the same bias or not.
        memory_dimension: int
            Dimension of the memory pool. Used only when use_compressed_embeddings is set to True.
            If memory_dimension == nb_embeddings*embeddings_dim, then the memory pool has the same
            size as an uncompressed embedding table.
        salt: int
            Salt used to hash the embedding indices.
        p_dropout: float
            Dropout probability.
        """

        super().__init__()

        no_offset = tf.constant([0, 0])
        offset = tf.constant([0, nb_embeddings])

        nb_stored_embeddings = nb_embeddings if use_same_embeddings else nb_embeddings * 2
        self.embedding_indice_offset = no_offset if use_same_embeddings else offset
        nb_stored_bias = nb_embeddings if use_same_bias else nb_embeddings * 2
        self.bias_indice_offset = no_offset if use_same_bias else offset

        if use_compressed_embeddings:
            assert memory_dimension is not None
            self.embedding_table = CompressedEmbedding(nb_stored_embeddings, embeddings_dim, memory_dimension, salt)
        else:
            self.embedding_table = layers.Embedding(
                nb_stored_embeddings, embeddings_dim, input_length=1, name="target_and_context_embedding"
            )

        if use_bias:
            self.embeddings_bias = layers.Embedding(
                nb_stored_bias, 1, input_length=1, name="target_and_context_embedding_bias"
            )

        self.dropout = tf.keras.layers.Dropout(p_dropout)
        self.use_bias = use_bias

    @tf.function
    def call(self, pair: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Parameters
        ----------
        pair: tf.Tensor of shape (batch_size, 2, embedding_dim)
            A batch of pairs of indices.
        training: bool
            Whether to use dropout or not.

        Returns
        -------
        tf.Tensor of shape (bs, 1)
            The PMI value of the given pair.
            It is the dot product of the corresponding embeddings of the pair of indices
            plus a bias if use_bias is set to True.
        """

        # Retrieve embeddings from memory
        # Shape (bs, 2, embedding_dim)
        embeddings = self.embedding_table(pair + self.indice_offset)

        # Add dropout to avoid overfitting
        # Shape (bs, 2, embedding_dim)
        embeddings = self.dropout(embeddings, training=training)

        # Compute similarity
        # Shape (bs, 1)
        sim = tf.einsum("bd,bd->b", embeddings[:, 0, :], embeddings[:, 1, :])
        sim = tf.expand_dims(sim, axis=1)

        # Add bias
        # Shape (bs, 1)
        if self.use_bias:
            sim += tf.sum(self.embeddings_bias(pair + self.bias_indice_offset), axis=-1)

        return sim


class BasicModel(tf.keras.Model):
    """
    Simple Glove-like model that generates embeddings in order
    to approximate a PMI matrix.

    This model is not using a compressed embedding table.

    This is a particular case of ModelWithCompressedEmbeddings and is presented here
    only for the sake of completeness and clarity.
    """

    def __init__(self, nb_embeddings: int, embeddings_dim: int, p_dropout: float = 0.1, use_bias: bool = True):
        """
        Initialize the model.

        Parameters
        ----------
        nb_embeddings: int
            Number of embeddings to generate.
        embeddings_dim: int
            Dimension of the embeddings.
        p_dropout: float
            Dropout probability.
        use_bias: bool
            Whether to use a bias or not.
        """
        super().__init__()
        self.target_embeddings = layers.Embedding(
            nb_embeddings, embeddings_dim, input_length=1, name="target_embedding"
        )
        self.context_embeddings = layers.Embedding(
            nb_embeddings, embeddings_dim, input_length=1, name="context_embedding"
        )

        if use_bias:
            self.context_embeddings_bias = layers.Embedding(
                nb_embeddings, 1, input_length=1, name="context_embedding_bias"
            )
            self.target_embeddings_bias = layers.Embedding(
                nb_embeddings, 1, input_length=1, name="target_embedding_bias"
            )

        self.dropout = tf.keras.layers.Dropout(p_dropout)
        self.use_bias = use_bias

    @tf.function
    def call(self, pair: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Parameters
        ----------
        pair: tf.Tensor of shape (batch_size, 2, embedding_dim)
            A batch of pairs of indices.
        training: bool
            Whether to use dropout or not.

        Returns
        -------
        tf.Tensor of shape (bs, 1)
            The PMI value of the given pair.
            It is the dot product of the corresponding embeddings of the pair of indices
            plus a bias if use_bias is set to True.
        """

        i, j = pair[:, 0], pair[:, 1]

        if len(i.shape) == 2:
            i = tf.squeeze(i, axis=[1])
        if len(j.shape) == 2:
            j = tf.squeeze(j, axis=[1])

        i_embedding = self.target_embeddings(i)
        j_embedding = self.context_embeddings(j)

        if training:
            i_embedding = self.dropout(i_embedding)
            j_embedding = self.dropout(j_embedding)

        sim = tf.einsum("bd,bd->b", i_embedding, j_embedding)

        sim = tf.expand_dims(sim, axis=1)

        if self.use_bias:
            sim += self.target_embeddings_bias(i) + self.context_embeddings_bias(j)

        return sim
