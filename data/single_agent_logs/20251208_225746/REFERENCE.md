# Reference Implementation

```python
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT using scipy.fftpack.
        """
        return fftpack.fftn(problem)

```
