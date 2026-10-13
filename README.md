# QAOA Portfolio Optimization: Local Run (Track 2, Beginner)

Solving the Track 2 portfolio-selection instance with QAOA on a local Qiskit simulator, and checking the result against brute force. The problem statement is in [`qiskit-fall-fest-2026-IIITD/hackathon/track-2-quantum-optimization/README.md`](qiskit-fall-fest-2026-IIITD/hackathon/track-2-quantum-optimization/README.md).

## Notebooks

| Notebook | What it does |
|---|---|
| [`scripts/optim_bruteforce.ipynb`](scripts/optim_bruteforce.ipynb) | Classical brute force over all valid portfolios |
| [`scripts/optim_beginner.ipynb`](scripts/optim_beginner.ipynb) | QUBO → Pauli operator → QAOA on a local simulator, then a check against brute force |

## Local instance

```python
rng = np.random.default_rng(2026)
n, K, q = 6, 3, 0.5               # 6 stocks, pick exactly 3, risk aversion 0.5
mu = rng.uniform(0.05, 0.25, n)   # expected returns
A = rng.normal(0, 0.1, (n, n))
sigma = A @ A.T                   # covariance
lam = 0.5                         # budget-penalty strength (chosen)
```

## Environment

- Local conda environment `qiskit_env` (Python 3.11)
- `qiskit`, `qiskit-optimization`, `numpy`, `scipy`, `matplotlib`
- Everything ran **locally** on Qiskit's exact statevector primitives (`StatevectorEstimator`, `StatevectorSampler`). The notebook also contains IBM Quantum setup cells, but no hardware was used for these results.

## 1. Brute force (`optim_bruteforce.ipynb`)

- Listed all $\binom{6}{3} = 20$ valid portfolios with `itertools.combinations` and checked that each buys exactly 3 stocks.
- Evaluated $C(x) = q\,x^\top\Sigma x - \mu^\top x + \lambda(\sum x_i - K)^2$ for each one and took the minimum.

**Result:** $x = [0, 1, 1, 0, 0, 1]$, i.e. stocks **{1, 2, 5}**, with **C = −0.50301** (expected return 0.52954).

The same brute-force step is repeated at the top of `optim_beginner.ipynb` (`best_x`), so the QAOA result can be checked in the same notebook.

## 2. Building the Pauli operator (`optim_beginner.ipynb`)

The cost was expanded into QUBO form by hand, using $x_i^2 = x_i$:

| Term | Coefficient |
|---|---|
| Linear, $x_i$ | $q\,\Sigma_{ii} - \mu_i + \lambda(1 - 2K)$ |
| Quadratic, $x_i x_j$ ($i<j$) | $2q\,\Sigma_{ij} + 2\lambda$ |
| Constant | $\lambda K^2 = 4.5$ |

These coefficients were loaded into a `qiskit_optimization.QuadraticProgram` as linear and quadratic dictionaries. `mod.to_ising()` then converted the program to a qubit operator:

- `pauliop`: a `SparsePauliOp` with **21 terms**, 6 single-qubit `Z` terms and 15 two-qubit `ZZ` terms
  - `Z` coefficients ≈ 0.036 – 0.100
  - `ZZ` coefficients ≈ 0.247 – 0.260
- `offset` = **0.370067**, added back to every energy so that ⟨H⟩ + offset = ⟨C⟩

With λ = 0.5, the penalty contributes 2λ = 1.0 to every quadratic coefficient. That is much larger than the risk part (2qΣᵢⱼ ≈ ±0.04), so the energy landscape is dominated by the "pick exactly 3" rule.

## 3. QAOA parameters used

| Setting | Value |
|---|---|
| Circuit | `QAOAAnsatz(cost_operator=pauliop, reps=2)`, so depth p = 2 and 4 angles |
| Transpilation | Once, before optimizing: `generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])` |
| Energy evaluation | `StatevectorEstimator` (exact, noiseless) |
| Optimizer | SciPy `minimize`, method COBYLA, `maxiter=300` |
| Starts | 8 starts, initial angles drawn uniformly from [0, π] with `np.random.default_rng(42)`; kept the lowest-energy run |
| Sampling | `StatevectorSampler(seed=42)`, 4096 shots |
| Answer selection | Kept shots that buy exactly K = 3 stocks, reversed each Qiskit key (little-endian), and picked the one with the **lowest cost** |

**Optimizer runs (⟨C⟩ per start):** 0.1367, 0.0131, **−0.3426**, −0.2020, −0.2164, 0.2757, −0.2573, −0.3115

**Best angles:** β = (1.3269, 1.4573), γ = (0.7703, 2.7702). This is `QAOAAnsatz`'s parameter order: β values first, then γ.

## 4. Results and comparison with brute force

| | Brute force | QAOA (p = 2, local) |
|---|---|---|
| Best portfolio | [0 1 1 0 0 1], stocks {1, 2, 5} | [0 1 1 0 0 1], stocks {1, 2, 5} |
| Cost C | −0.5030 | −0.5030 |
| Matches | | ✅ `matches brute force: True` |

Sample quality, compared with what random guessing over all 64 bitstrings would give:

| Metric | QAOA | Random guessing |
|---|---|---|
| Valid fraction (exactly 3 stocks) | **96.0%** | 31.3% (20/64) |
| P(optimum) | **4.9%** (199 / 4096 shots) | 1.6% (1/64) |
| Expected cost ⟨C⟩ | −0.3426 | |
| Approximation ratio r* | ≈ 0.968 | |

\* r = (C_max − ⟨C⟩) / (C_max − C_min). It was calculated by hand from the numbers above, with C_max = 4.5 ("buy nothing", penalty 9λ) and C_min = −0.5030. It is not an output of the notebook.

QAOA learned the budget rule well: 96% of shots are valid. Among the valid portfolios, however, the counts are almost flat (158–216 per portfolio). The optimum is identified by **evaluating the cost of each sampled valid portfolio**, not by how often it appears.

## Notes from earlier attempts

- **Single random start:** the first run used one unseeded starting point. COBYLA got stuck in a poor local minimum: only 22.6% of shots were valid (worse than random), and the most frequent valid bitstring was {0, 1, 2}, which is rank 10 of 20. Running 8 seeded starts fixed this.
- **Choosing by frequency:** picking the *most frequent* valid bitstring returned the wrong portfolio even when the optimum had been sampled, because counts differ only by shot noise. Choosing by lowest cost fixed this.
- **No transpilation:** without transpiling first, every optimizer step printed SciPy `SparseEfficiencyWarning`s. Transpiling once before the loop removed them.

## Next steps

- Sweep depth p = 1 … 4 and plot r
- Sweep λ and plot the valid fraction and P(optimum)
- Run the best circuit on IBM hardware and compare it with the simulator
