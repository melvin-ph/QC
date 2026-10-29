# Q-Balance Architecture & Theoretical Design

**Q-Balance** is an interactive, browser-native quantum workload balancing simulator. It demonstrates how **Adiabatic Quantum Computing (AQC)** and **Quantum Annealing** principles can solve combinatorial optimization problems—specifically the **NP-hard Number Partitioning Problem**—without requiring quantum hardware, cloud services, or a backend.

---

## 1. Problem Formulation: Server Workload Balancing

Given a workload of $n$ integer task sizes:
$$A = [a_1, a_2, \dots, a_n]$$

The objective is to divide these items into two disjoint subsets assigned to **Server A** and **Server B** such that the load imbalance is minimized:

$$\min_{z \in \{0, 1\}^n} \left| \sum_{i \in \text{Server A}} a_i - \sum_{j \in \text{Server B}} a_j \right|$$

In computer science, this is known as the **Number Partitioning Problem** (a Karp NP-complete problem). The search space grows exponentially as $2^n$.

---

## 2. Quantum Formulation: The Ising Model

To solve this problem via quantum annealing, the discrete cost function is converted into an operator on a Hilbert space $(\mathbb{C}^2)^{\otimes n}$.

### 2.1 Mapping Decisions to Pauli-$Z$ Operators
Each item $i$ maps to a qubit $i$, where the computational basis states define server assignment:
- $|1\rangle \implies$ **Server A** (eigenvalue $z_i = -1$)
- $|0\rangle \implies$ **Server B** (eigenvalue $z_i = +1$)

The total load difference operator is:
$$D = \sum_{i=1}^n a_i Z_i$$

### 2.2 Problem (Cost) Hamiltonian $H_P$
The objective function is quadratic, penalizing non-zero imbalance:
$$H_P = D^2 = \left( \sum_{i=1}^n a_i Z_i \right)^2 = \sum_{i=1}^n a_i^2 I + 2 \sum_{i < j} a_i a_j Z_i Z_j$$

- $H_P$ is **strictly diagonal** in the computational basis.
- The ground-state eigenvalue $E_0$ is the minimum possible squared imbalance $\Delta_{\min}^2$.
- If an exact balanced partition exists, $E_0 = 0$.

### 2.3 Initial (Driver) Hamiltonian $H_0$
$$H_0 = -\sum_{i=1}^n X_i$$

- Each single-qubit transverse field operator $X_i$ acts independently (they commute with each other).
- The ground state of $H_0$ is the equal superposition state:
  $$|\psi_0\rangle = |+\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{z \in \{0,1\}^n} |z\rangle$$

### 2.4 Adiabatic Interpolation
The system is evolved under the time-dependent Hamiltonian parametrized by dimensionless time $s \in [0, 1]$:
$$H(s) = (1 - s) H_0 + s H_P$$

According to the **Adiabatic Theorem of Quantum Mechanics**, if the evolution parameter $s(t)$ changes slowly enough with respect to the minimum instantaneous spectral gap $\Delta E(s) = E_1(s) - E_0(s)$, the system stays in the ground state throughout the evolution, ending in the ground state of $H_P$.

---

## 3. Simulation Engine Architecture

Because the app is 100% frontend-only, the quantum state evolution is computed in JavaScript using typed arrays (`Float64Array`).

```
                +---------------------------------------+
                |     Initial Statevector Preparation   |
                |   |ψ(0)> = |+>^(⊗n) = 1/√(2ⁿ) Σ |z>   |
                +---------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| Time Evolution Loop: step = 0 -> steps - 1 (dt = T / steps)           |
|                                                                       |
| 1. Midpoint schedule: s = (step + 0.5) / steps                        |
|                                                                       |
| 2. Apply Problem Hamiltonian Phase (Diagonal):                       |
|    |z>  --->  exp( -i · s · E(z) · dt ) |z>                           |
|                                                                       |
| 3. Apply Driver Mixing Rotations (Commuting X_i terms):               |
|    For each qubit q: exp( +i · (1 - s) · dt · X_q )                   |
|                                                                       |
| 4. Periodic Telemetry (onFrame callback):                             |
|    - Norm & Probabilities: p(z) = |⟨z|ψ⟩|²                            |
|    - Energy Expectation: ⟨H_P(s)⟩ = Σ p(z) · E(z)                     |
|    - Optimal State Probability: P_opt = Σ_{E(z)=E_min} p(z)           |
+-----------------------------------------------------------------------+
                                    |
                                    v
                +---------------------------------------+
                |        Final State Measurement        |
                |   Candidate bitstring selected from   |
                |   highest probability optimal state   |
                +---------------------------------------+
```

### 3.1 Statevector Representation
- For $n \le 10$ qubits, the statevector contains $N = 2^n$ complex amplitudes.
- A single `Float64Array(2 * N)` stores interleaved real and imaginary parts:
  - $\text{Re}(\psi_z) =$ `state[2 * z]`
  - $\text{Im}(\psi_z) =$ `state[2 * z + 1]`

### 3.2 Trotterized First-Order Product Formula
At each discrete midpoint step:
1. **Diagonal Phase Step ($H_P$):**
   $$\exp(-i \cdot s \cdot H_P \cdot dt)$$
   Executed in $O(2^n)$ operations by direct complex multiplication on each basis amplitude.
2. **Transverse Mixer Step ($H_0$):**
   $$\exp(-i \cdot (1 - s) \cdot H_0 \cdot dt) = \prod_{q=1}^n \exp(+i \cdot (1 - s) \cdot dt \cdot X_q)$$
   Because $\exp(+i \alpha X) = \cos(\alpha) I + i \sin(\alpha) X$, each qubit rotation is applied across all bit pairs that differ only at bit $q$. This takes $O(n \cdot 2^n)$ operations.

---

## 4. System Components & UI Architecture

The React application is structured into clear modular panels:

```
App Shell (src/main.jsx)
 │
 ├── Sidebar Navigation ("overview" | "simulator" | "hamiltonian" | "verification")
 │
 ├── Topbar & Global Stats Grid (Qubits, 2ⁿ Search Space, Best Imbalance, Optimal Count)
 │
 ├── Overview Tab (Dashboard)
 │    ├── Panel 01: Workload Configuration (Input weights, chips, sliders, start/reset)
 │    ├── Panel 02: Adiabatic Evolution (Real-time progress, Qubit orbital stage, energy indicators)
 │    ├── Panel 03: Evolution Analytics (New)
 │    │    ├── Metric Chips (Initial ⟨H_P⟩, Current ⟨H_P⟩, Ground E₀, Optimal Subspace P_opt)
 │    │    ├── EnergyGraph (Custom responsive SVG plotting ⟨H_P(s)⟩ decay and P_opt(s) rise)
 │    │    └── ProbabilityChart (Bar chart of top basis states with OPTIMAL state highlighting)
 │    └── Panel 04: Server Allocation (Server A & Server B load cards, imbalance comparison)
 │
 ├── Simulator Tab (Dedicated step-by-step simulator view with large qubit cards)
 ├── Hamiltonian Tab (Theoretical derivations of H_P, H_0, and H(s))
 └── Verification Tab (Exhaustive classical brute-force table across all 2ⁿ assignments)
```

---

## 5. Classical Verification & Ground Truth

To provide absolute scientific verification without fabricating outcomes:
- An independent classical brute-force function (`bruteForce`) checks every integer mask in $[0, 2^n - 1]$.
- Calculates:
  - Total items assigned to Server A ($a$) and Server B ($b = \text{Total} - a$).
  - Difference $\Delta = |a - b|$.
- Identifies the exact global minimum difference $\Delta_{\min}$ and all degenerate optimal bitstrings (including bit-flip symmetry).
- Example: For workload `[3, 1, 1, 2, 2, 1]`, the total sum is 10, the exact optimal imbalance is $\Delta_{\min} = 0$, and there are 10 distinct optimal partitions.

---

## 6. Directory Structure

```
QC/
├── index.html              # HTML5 entrypoint
├── package.json            # React 18, Vite 6, Lucide React dependencies
├── README.md               # User guide and quick start
├── architecture.md         # Full architectural & theoretical specification
└── src/
    ├── main.jsx            # Quantum simulation core, state management, & UI
    └── style.css           # Custom dark theme styles, animations, & chart layouts
```

---

## 7. Performance Characteristics

- **Computational Complexity per Step:** $\mathcal{O}(n \cdot 2^n)$
- **Browser Memory Footprint:** $< 2 \text{ MB}$ for up to 10 qubits (1024 basis states).
- **Execution Latency:** Browser renders smooth 60fps animations by yielding to the event loop (`setTimeout(0)`) at regular intervals.

