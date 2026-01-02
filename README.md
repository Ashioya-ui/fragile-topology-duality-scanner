Fragile Topology Duality Scanner

Numerical verification of Momentum-Space Anyon localization in Crystalline Chern Insulators.

Overview

This repository contains the simulation engines designed to test the Position-Momentum Duality hypothesis proposed by Sati and Schreiber (2025).

We perform two distinct tests to classify the "Fragile" Phase:

Duality Test (duality_engine.py): Checks for "Momentum Space Anyons" using the Momentum-Space Entanglement Spectrum (MSES).

Obstruction Test (polarization_test.py): Checks for "Obstructed Atomic Limits" using the Many-Body Resta Polarization.

The Test Cases

1. Entanglement Spectrum (Liquid Order)

Run: python duality_engine.py

Goal: Detect topological edge modes in the momentum-space partition.

Result: Determines if the phase is a "Dual FCI."

2. Resta Polarization (Crystal Order)

Run: python polarization_test.py

Goal: Calculate the center-of-mass shift of the interacting ground state.

Result: Determines if the phase is an "Obstructed Atomic Limit" (electrons locked in empty voids).

$P \approx 0$: Trivial Insulator.

$P \neq 0$: Fragile Topological Crystal.

Installation

pip install -r requirements.txt


References

H. Sati, U. Schreiber, "Fragile Topological Phases and Topological Order of 2D Crystalline Chern Insulators" (2025).