# TensorFluid
Implementation of lattice-boltzmann method in Tensorflow, python and C++.

## Installing TensorFluid
### 1. Clone and install virtual environment

```
git clone https://github.com/ma595/TensorFluid
cd TensorFluid
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

It is now time to install the dependencies for our code, for example Tensorflow.
The project has been packaged with a [`pyproject.toml`](pyproject.toml) so can be installed in one go.
From within the root directory in a active virtual environment run:
```
pip install .
```

## Timings summary 
Timings for flow around bluff body, problem size 1680 x 720 

### Tensorflow
(without output)
single GPU: 24s
single Threaded (CPU): 134s

### CPU C++ 
Partially optimised OpenMP implementation:

| OMP Threads | Time (s) |
|    :---:    |   :---:  |
| 1           |   233.1  |
| 2           |   158.6  |
| 4           |   123.2  |
| 8           |   93.16  |
| 16          |   69.54  |
| 32          |   66.22  | 

