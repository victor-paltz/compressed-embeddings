"""
A few hash functions
"""
import tensorflow as tf
import numpy as np


def hash_identity(array: np.ndarray, *args, **kwargs) -> np.ndarray:
    return array


def hash_1(array: tf.Tensor, num_bins: int, salt: int = 0) -> tf.Tensor:
    hashing_layer = tf.keras.layers.Hashing(num_bins=num_bins, salt=salt)
    return hashing_layer(array)


def hash_2(array: np.ndarray, num_bins: int, salt: int = 0) -> np.ndarray:
    """
    Hash function based on the one int tha pandas library
    """
    array = np.uint64(array.numpy()) + salt
    array ^= array >> 30
    array *= np.uint64(0xBF58476D1CE4E5B9)
    array ^= array >> 27
    array *= np.uint64(0x94D049BB133111EB)
    array ^= array >> 31
    return np.int32(array) % num_bins


def hash_3(array: tf.Tensor, num_bins: int, salt: int = 0) -> tf.Tensor:
    """
    Tensorflow implementation
    """
    array = tf.cast(array, dtype=tf.uint64) + salt
    array = tf.bitwise.bitwise_xor(array, tf.bitwise.right_shift(array, 30))
    array *= 0xBF58476D1CE4E5B9
    array = tf.bitwise.bitwise_xor(array, tf.bitwise.right_shift(array, 27))
    array *= 0x94D049BB133111EB
    array = tf.bitwise.bitwise_xor(array, tf.bitwise.right_shift(array, 31))
    return tf.cast(array, dtype=tf.int32) % num_bins


def universal_hash(array: tf.Tensor, P, A, B, C, salt: int = 0) -> tf.Tensor:
    """
    WIP: hash function from ROBE-Z paper
    """

    # TODO
    # hashed_idx = ((((((indices.view(-1,1) + val_offset) * helper_E1sR) + helper_Eidx_base * B) + A) % P) % (hashed_weights_size -uma_chunk_size +1) + helper_Eidx_offset)

    raise NotImplementedError("TODO")
