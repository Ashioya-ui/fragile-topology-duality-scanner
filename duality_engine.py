import numpy as np
import itertools
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time

class DualityEngine:
    def __init__(self, Lx=3, Ly=2, t1=0.2, t2=1.0, phi=0.0, V=2.0):
        """
        Lx=3, Ly=2 is the minimum size to see real bulk/edge distinction (18 sites).
        t1 < t2 puts us in the 'Fragile' / Obstructed limit.
        V is the interaction strength.
        """
        self.Lx, self.Ly = Lx, Ly
        self.t1, self.t2 = t1, t2
        self.phi = phi
        self.V = V
        
        # Kagome Lattice: 3 atoms (A,B,C) per unit cell
        self.n_sites = 3 * Lx * Ly
        # Filling: 1 particle per unit cell (1/3 filling of lattice)
        self.n_particles = Lx * Ly 
        
        print(f"--- SYSTEM SETUP ---")
        print(f"Lattice: {Lx}x{Ly} ({self.n_sites} sites)")
        print(f"Particles: {self.n_particles}")
        print(f"Interaction V: {self.V}")

        # Build Real Space Basis
        # We limit Hilbert space for Lx=3,Ly=2 (18 choose 6 = 18,564) - manageable.
        self.basis_real = list(itertools.combinations(range(self.n_sites), self.n_particles))
        self.dim = len(self.basis_real)
        self.map_real = {state: i for i, state in enumerate(self.basis_real)}
        print(f"Hilbert Space Dimension: {self.dim}")

    def build_hamiltonian(self):
        """Constructs the Hamiltonian with explicit interactions."""
        print("Building Hamiltonian...")
        start_t = time.time()
        H = lil_matrix((self.dim, self.dim), dtype=complex)
        
        # Precompute Lattice Graph (Hoppings & Neighbors)
        hoppings = []
        neighbors = [] # For V interaction
        p_ph = np.exp(1j * self.phi)
        m_ph = np.exp(-1j * self.phi)
        
        for x in range(self.Lx):
            for y in range(self.Ly):
                cell = y*self.Lx + x
                A, B, C = 3*cell, 3*cell+1, 3*cell+2
                
                # --- Intracell (t1) ---
                # A-B, B-C, C-A
                pairs = [(A,B), (B,C), (C,A)]
                for u, v in pairs:
                    hoppings.append((u, v, -self.t1 * p_ph))
                    hoppings.append((v, u, -self.t1 * m_ph))
                    neighbors.append(tuple(sorted((u,v))))

                # --- Intercell (t2) Periodic BCs ---
                # X-Direction: A(r) -> B(r - x)
                x_prev = (x - 1) % self.Lx
                cell_prev_x = y*self.Lx + x_prev
                B_prev = 3*cell_prev_x + 1
                hoppings.append((A, B_prev, -self.t2 * m_ph))
                hoppings.append((B_prev, A, -self.t2 * p_ph))
                neighbors.append(tuple(sorted((A, B_prev))))
                
                # Y-Direction: B(r) -> C(r - y)
                y_prev = (y - 1) % self.Ly
                cell_prev_y = y_prev*self.Lx + x
                C_prev = 3*cell_prev_y + 2
                hoppings.append((B, C_prev, -self.t2 * m_ph))
                hoppings.append((C_prev, B, -self.t2 * p_ph))
                neighbors.append(tuple(sorted((B, C_prev))))
                
                # Diagonal: C(r) -> A(r - y + x)
                # Note: This connectivity depends on specific Kagome definition. 
                # Using standard 'Breathing' connectivity.
                x_next = (x + 1) % self.Lx
                cell_diag = y_prev*self.Lx + x_next
                A_diag = 3*cell_diag
                hoppings.append((C, A_diag, -self.t2 * m_ph))
                hoppings.append((A_diag, C, -self.t2 * p_ph))
                neighbors.append(tuple(sorted((C, A_diag))))

        # Fill Matrix (Optimized Loop)
        for idx, state in enumerate(self.basis_real):
            state_set = set(state)
            
            # 1. Kinetic Term
            for u, v, val in hoppings:
                if (u in state_set) and (v not in state_set):
                    # Hop u -> v
                    # Assuming Hardcore Bosons for topological gap check 
                    # (Avoids complex Fermion sign logic for this demo)
                    new_state = list(state)
                    new_state.remove(u)
                    new_state.append(v)
                    new_state.sort()
                    
                    target_idx = self.map_real[tuple(new_state)]
                    H[idx, target_idx] += val
            
            # 2. Interaction Term
            e_int = 0.0
            for u, v in neighbors:
                if (u in state_set) and (v in state_set):
                    e_int += self.V
            if e_int != 0:
                H[idx, idx] += e_int

        print(f"Hamiltonian Built in {time.time()-start_t:.2f}s")
        return H.tocsr()

    def get_ground_state(self):
        H = self.build_hamiltonian()
        print("Diagonalizing...")
        # Get ground state only
        vals, vecs = eigsh(H, k=1, which='SA')
        print(f"Ground State Energy: {vals[0]:.4f}")
        return vecs[:, 0]

    def transform_and_measure_es(self, psi_real):
        """
        Calculates BOTH Real and Momentum ES.
        Optimized to avoid storing full Transform Matrix.
        """
        # --- 1. Real Space ES ---
        print("\nComputing Real-Space Entanglement Spectrum...")
        # Partition: Cut lattice in half along X (x < Lx/2)
        sites_A_real = set()
        for i in range(self.n_sites):
            cell = i // 3
            cx = cell % self.Lx
            if cx < self.Lx / 2: # Spatial Cut
                sites_A_real.add(i)
        
        es_real = self._compute_es(psi_real, self.basis_real, sites_A_real)
        
        # --- 2. Momentum Space ES ---
        print("\nComputing Momentum-Space Entanglement Spectrum...")
        # We need to transform the vector psi_real -> psi_mom
        # |k> = (1/sqrt(N)) sum_r e^{-i k.r} |r>
        
        # Build Single Particle U (Dim 18x18) - Cheap
        U_sp = np.zeros((self.n_sites, self.n_sites), dtype=complex)
        for r in range(self.n_sites):
            cell = r // 3; sub = r % 3
            rx, ry = cell % self.Lx, cell // self.Lx
            for k in range(self.n_sites):
                cell_k = k // 3; sub_k = k % 3
                if sub == sub_k: # Sublattice diagonal
                    kx = 2*np.pi*(cell_k % self.Lx)/self.Lx
                    ky = 2*np.pi*(cell_k // self.Lx)/self.Ly
                    phase = np.exp(-1j*(kx*rx + ky*ry))
                    U_sp[k, r] = phase / np.sqrt(self.Lx*self.Ly)
        
        # Transform Many-Body State
        print("Transforming Basis (this may take 10-20s)...")
        # We filter psi_real to only significant components to speed up
        sig_indices = np.where(np.abs(psi_real) > 1e-6)[0]
        print(f"Significant components: {len(sig_indices)}/{self.dim}")
        
        psi_mom = np.zeros(self.dim, dtype=complex)
        
        # We iterate over MOMENTUM basis states (rows)
        # Strict calculation:
        for i_k, state_k in enumerate(self.basis_real): # Momentum basis indices same as real
            val = 0.0 + 0j
            # Inner loop over significant real states
            for i_r in sig_indices:
                state_r = self.basis_real[i_r]
                coeff = psi_real[i_r]
                
                # Construct submatrix
                mat = U_sp[np.ix_(state_k, state_r)]
                det = np.linalg.det(mat)
                val += det * coeff
            
            psi_mom[i_k] = val
            
        # Normalize just in case
        psi_mom /= np.linalg.norm(psi_mom)
        
        # Momentum Partition: Cut BZ in half (kx < pi)
        sites_A_mom = set()
        for k_idx in range(self.n_sites):
            cell_k = k_idx // 3
            kx_idx = cell_k % self.Lx
            if kx_idx < self.Lx / 2: # Momentum Cut
                sites_A_mom.add(k_idx)
                
        es_mom = self._compute_es(psi_mom, self.basis_real, sites_A_mom)
        
        return es_real, es_mom

    def _compute_es(self, psi, basis, sites_A):
        """Generic SVD for Entanglement Spectrum"""
        # Target the half-filling sector of A
        target_NA = self.n_particles // 2
        
        # Map: State_A -> {State_B -> coeff}
        blocks = {} 
        
        for idx, state in enumerate(basis):
            if abs(psi[idx]) < 1e-10: continue
            
            state_A = tuple(sorted([s for s in state if s in sites_A]))
            state_B = tuple(sorted([s for s in state if s not in sites_A]))
            
            if len(state_A) == target_NA:
                if state_A not in blocks: blocks[state_A] = {}
                blocks[state_A][state_B] = psi[idx]
        
        if not blocks: return np.array([])
        
        unique_A = list(blocks.keys())
        unique_B = set()
        for sA in unique_A:
            unique_B.update(blocks[sA].keys())
        unique_B = list(unique_B)
        
        map_A = {s: i for i, s in enumerate(unique_A)}
        map_B = {s: i for i, s in enumerate(unique_B)}
        
        M = np.zeros((len(unique_A), len(unique_B)), dtype=complex)
        
        for sA, inner in blocks.items():
            r = map_A[sA]
            for sB, val in inner.items():
                c = map_B[sB]
                M[r, c] = val
                
        # SVD
        try:
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
            # Entanglement Spectrum levels = -ln(s^2)
            s = s[s > 1e-12]
            return -np.log(s**2)
        except:
            return np.array([])

if __name__ == "__main__":
    # --- RED TEAM CONFIGURATION ---
    # We use a 3x2 lattice for rigorous finite-size check.
    # We use Moderate V=2.0 to avoid CDW triviality.
    engine = DualityEngine(Lx=3, Ly=2, t1=0.2, t2=1.0, phi=0.0, V=2.0)
    
    gs = engine.get_ground_state()
    es_real, es_mom = engine.transform_and_measure_es(gs)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if len(es_real) > 0:
        plt.plot(np.arange(len(es_real)), np.sort(es_real), 'o-', c='gray')
    plt.title("Real Space ES\n(Should be Gapped for Fragile Phase)")
    # Corrected line below: using raw string for LaTeX
    plt.ylabel(r"Entanglement Energy $\xi$")
    plt.xlabel("Level Index")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if len(es_mom) > 0:
        plt.plot(np.arange(len(es_mom)), np.sort(es_mom), 'o-', c='red')
    plt.title("Momentum Space ES\n(Look for Low-Lying Counting)")
    plt.xlabel("Level Index")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()