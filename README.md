# MEMD for Python

## Introduction
Python implementation of Multivariate Empirical Mode Decomposition (MEMD).

Multivariate Empirical Mode Decomposition (MEMD) is an extension of the traditional Empirical Mode Decomposition (EMD) method, which is used to decompose non-linear and non-stationary signals into simpler oscillatory components known as Intrinsic Mode Functions (IMFs). The key feature of MEMD is its ability to handle multivariate signals, meaning it can simultaneously decompose multiple related signals (or different dimensions of a signal) while ensuring that the decomposition is consistent across all channels.

__Key Benefits of Repository__ :
- Multivariate Decomposition :
  - Decomposes n-dimensional signals, not limited to bivariate or quadrivariate, making it applicable to complex, high-dimensional data.
- Performance Optimization :
  - The implementation is significantly faster than existing models, making it practical for large-scale and real-time data applications.

## Dependencies 
- NumPy
- SciPy  
- sys
- math

```bash
# sys and math are part of Python Standard Libraries
pip install numpy scipy 
```


## General Functions and Usage  
```python
signal = np.random.randn(5, 1000)
# Standard MEMD: Perform regular MEMD (returns matrix of (Channels, IMFs, Data Points))
imfs_memd = memd(signal)
```

## Acknowledgements
Several existing packages and repositories were referenced in the creation of this library. All credit goes to these authors for their contributions to the field.
* https://github.com/laszukdawid/PyEMD/tree/master
* https://github.com/mariogrune/MEMD-Python-/tree/master [1]

## Citations
[1] “Research | Empirical Mode Decomposition (EMD), Multivariate EMD, Matlab code and data sources ∴ Dr. Danilo P. Mandic,” www.commsp.ee.ic.ac.uk. https://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm (accessed Jul. 09, 2024)  [LINK](https://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm)
