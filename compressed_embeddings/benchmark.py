""" Benchmark on compressed embedding table speed """


from typing import List
from compressed_embeddings.models.models import ModelWithCompressedEmbeddings
from compressed_embeddings.utils.converter import cast_bytes_to_memory_string
from compressed_embeddings.utils.decorators import timing
from keras.metrics import mean_squared_error
from tqdm import tqdm as tq
import pandas as pd
import numpy as np
import tensorflow as tf


def benchmark(batch_sizes: List[int], mem_size: List[int]) -> pd.DataFrame:
    """ Benchmark compressed embedding table speed """

    report = pd.DataFrame()

    nb_embeddings = 100_000
    embeding_dim = 100

    model_names = ["FullEmbTableModel", "CompressedEmbTableModel"]
    optimizers = [
        tf.keras.optimizers.Adagrad(learning_rate=0.05),
        tf.keras.optimizers.SGD(learning_rate=0.01),
        tf.keras.optimizers.Adam(learning_rate=0.001),
    ]

    for optimizer in optimizers:
        for model_str in model_names:
            for bs in tq(batch_sizes):
                for i, mem in enumerate(mem_size):

                    use_compressed_embeddings = model_str == "CompressedEmbTableModel"

                    if not use_compressed_embeddings:
                        mem = None
                        if i != 0:
                            continue

                    model = ModelWithCompressedEmbeddings(
                        nb_embeddings,
                        embeding_dim,
                        use_same_embeddings=True,
                        use_compressed_embeddings=use_compressed_embeddings,
                        use_bias=False,
                        memory_dimension=mem,
                    )

                    loss_fct = mean_squared_error

                    forward_pass = []
                    gradient_computation = []
                    backward_pass = []

                    for _ in range(8):

                        x = tf.constant(np.random.randint(0, nb_embeddings, (bs, 2), dtype="int32"))
                        y = tf.constant(np.random.rand(bs, 1), dtype="float32")

                        with tf.GradientTape() as tape:
                            # model
                            t, y_pred = timing(model)(x, training=True)
                            forward_pass.append(t)
                            loss_value = loss_fct(y, y_pred)
                        # grad
                        t, grads = timing(tape.gradient)(loss_value, model.trainable_weights)

                        gradient_computation.append(t)
                        # apply
                        t, _ = timing(optimizer.apply_gradients)(zip(grads, model.trainable_weights))
                        backward_pass.append(t)

                    forward_pass = np.mean(forward_pass[2:])
                    gradient_computation = np.mean(gradient_computation[2:])
                    backward_pass = np.mean(backward_pass[2:])

                    optimizer_name = str(type(optimizer)).split(".")[-1].split("'")[0]

                    infos = pd.DataFrame(
                        {
                            "Model": model_str,
                            "batch_size": [bs],
                            "memory_size": [mem],
                            "forward_pass_ms": [forward_pass * 1000],
                            "gradient_computation_ms": [gradient_computation * 1000],
                            "backward_pass_ms": [backward_pass * 1000],
                            "model size": [cast_bytes_to_memory_string(4 * model.count_params())],
                            "optimizer": [optimizer_name],
                        }
                    )
                    print(infos)
                    report = report.append(infos)

    return report


# pylint: disable=redefined-outer-name
if __name__ == "__main__":

    batch_sizes = [64, 512, 4096, 32768]
    mem_size = [1, 8, 64, 512] + [32768 * 8 ** i for i in range(5)]

    my_report = benchmark(batch_sizes, mem_size)
    my_report.to_csv("benchmark.csv", index=False)
