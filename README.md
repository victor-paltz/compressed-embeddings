# compressed-embeddings

This repo trains compressed embeddings to factorize a PMI matrix.

## How to run a benchmark

First create a virtual environment and install dependencies
```python
python -m venv .env
source .env/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Then run the following command to benchmark the training speed of compressed embeddings versus not compressed embeddings.	
```python
python benchmark.py
```
The output of this command is a csv file.

Here is an extract of the current output results:

| Model                   | Batch Size | Memory pool size (nb values) | Forward pass (ms) | gradient computation (ms) | backward pass (ms) | total time (ms) | model size | optimizer |
|-------------------------|------------|------------------------------|-------------------|---------------------------|--------------------|-----------------|------------|-----------|
| FullEmbTableModel       | 32768      |                              | 24.61             | 28.51                     | 21.8               | 74.92           | 381.5MB    | Adagrad   |
| CompressedEmbTableModel | 32768      | 262144                       | 94.95             | 27.13                     | 389.54             | 511.62          | 1.0MB      | Adagrad   |
| CompressedEmbTableModel | 32768      | 100000000                    | 119.08            | 26.52                     | 1096.84            | 1242.45         | 381.5MB    | Adagrad   |
| FullEmbTableModel       | 32768      |                              | 23.59             | 27.27                     | 12.73              | 63.59           | 381.5MB    | SGD       |
| CompressedEmbTableModel | 32768      | 262144                       | 94.49             | 27.5                      | 77.02              | 199.01          | 1.0MB      | SGD       |
| CompressedEmbTableModel | 32768      | 100000000                    | 117.99            | 27                        | 104.76             | 249.75          | 381.5MB    | SGD       |
| FullEmbTableModel       | 32768      |                              | 23.86             | 28.76                     | 1900.49            | 1953.11         | 381.5MB    | Adam      |
| CompressedEmbTableModel | 32768      | 262144                       | 95.39             | 26.62                     | 393.24             | 515.25          | 1.0MB      | Adam      |
| CompressedEmbTableModel | 32768      | 100000000                    | 120.13            | 26.72                     | 2873.85            | 3020.7          | 381.5MB    | Adam      |