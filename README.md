# Q-Balance — Quantum Workload Balancing Simulator

Frontend-only React + Vite demo. No backend, account, or API key required.

## Run locally
1. Install Node.js 18 or newer.
2. Open a terminal in this folder.
3. Run `npm install`
4. Run `npm run dev`
5. Open the local URL printed by Vite.

## What it includes
- Editable positive-integer workload list (up to 10 items)
- Browser-side statevector simulation using a first-order product formula for adiabatic evolution
- Animated evolution progress and computed state probabilities
- Ising Hamiltonian explanation
- Independent brute-force enumeration and exact optimum
- Responsive dark dashboard

## Scientific notes
The statevector is initialized in the ground state of `H0 = -sum(X_i)`. At each midpoint step, the simulation applies the diagonal phase from `HP = (sum(a_i Z_i))^2` and the commuting single-qubit evolution terms from `H0`. This is a first-order product-formula approximation, not a hardware run. The finite-time final state need not be in the ground state. The brute-force result is the exact reference for the workload instance.

Bit convention in the UI: `1` assigns the item to Server A; `0` assigns it to Server B.
