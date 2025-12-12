# Reference Implementation

```python
    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        return signal.convolve(a, b, mode=self.mode)

```
