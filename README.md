# compressed-embeddings

This repo trains compressed embeddings to factorize a PMI matrix.

## How to use

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