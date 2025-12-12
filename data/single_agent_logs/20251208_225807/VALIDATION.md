# Validation Method

```python
    def is_solution(self, problem: NDArray, solution: NDArray) -> bool:
        """
        Check if the DCT Type I solution is valid and optimal.

        A valid solution must match the reference implementation (scipy.fftpack.dctn)
        within a small tolerance.

        :param problem: Input array.
        :param solution: Computed DCT result.
        :return: True if the solution is valid and optimal, False otherwise.
        """
        tol = 1e-6
        reference = scipy.fftpack.dctn(problem, type=1)
        error = np.linalg.norm(solution - reference) / (np.linalg.norm(reference) + 1e-12)
        if error > tol:
            logging.error(f"DCT solution error {error} exceeds tolerance {tol}.")
            return False
        return True

```
