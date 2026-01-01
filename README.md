Fragile Topology Duality Scanner

Numerical verification of Momentum-Space Anyon localization in Crystalline Chern Insulators.

Overview

This repository contains the simulation engine designed to test the Position-Momentum Duality hypothesis proposed by Sati and Schreiber (2025).

The central question: Do "fragile" topological phases host anyonic excitations localized in momentum space?

To answer this, this engine simulates interacting spinless fermions on a Breathing Kagome Lattice and computes the Entanglement Spectrum (ES) in two dual partitions:

Real-Space Cut: Detects standard topological order (Chern Insulators).

Momentum-Space Cut: Detects the proposed dual topological order (Fragile/Crystalline phases).

The Test

We simulate the interacting Hamiltonian:


$$H = \sum_{ij} t_{ij} c^\dagger_i c_j + V \sum_{\langle ij \rangle} n_i n_j$$

Null Hypothesis: The interaction $V$ drives the fragile phase into a trivial Charge Density Wave (CDW), resulting in a featureless spectrum in both cuts.

Duality Hypothesis (Sati/Schreiber): The interaction stabilizes a state where the Li-Haldane counting statistics emerge in the Momentum-Space Entanglement Spectrum, verifying the physical existence of momentum-space anyons.

Installation

pip install -r requirements.txt


Usage

Run the main duality engine:

python duality_engine.py


Configuration

You can adjust the physics parameters in duality_engine.py to test different regimes:

Lx, Ly: Lattice dimensions (Default: 3x2 for rigorous bulk/edge separation).

V: Interaction strength (Default: 2.0).

t1, t2: Hopping parameters (t1 < t2 selects the Obstructed/Fragile phase).

Interpretation of Results

The script generates a dashboard with two plots:

Left (Real Space): Should be gapped (no low-lying levels) for the fragile phase.

Right (Momentum Space): Look for low-lying entanglement levels counting 1, 1, 2, 3... This signature confirms the duality.

References

H. Sati, U. Schreiber, "Fragile Topological Phases and Topological Order of 2D Crystalline Chern Insulators" (2025).