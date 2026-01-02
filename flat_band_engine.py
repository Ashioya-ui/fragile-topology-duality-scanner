import numpy as np
import itertools
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time
from numba import jit, prange

# --- ARCHITECT'S NOTE: Numba Speed Core ---
@jit(nopython=True, parallel=True, fastmath=True)
def transform_basis_fast(psi_real, basis_array, U_sp, sig_indices):
    dim = len(basis_array)
    psi_mom = np.zeros(dim, dtype=np.complex128)
    n_sig = len(sig_indices)
    
    for i_k in prange(dim): 
        state_k = basis_array[i_k]
        val = 0.0 + 0j
        for idx_r in range(n_sig):
            i_r = sig_indices[idx_r]
            state_r = basis_array[i_r]
            coeff = psi_real[i_r]
            
            mat = np.zeros((6, 6), dtype=np.complex128)
            for r in range(6):
                row_idx = state_k[r]
                for c in range(6):
                    col_idx = state_r[c]
                    mat[r, c] = U_sp[row_idx, col_idx]
            
            val += np.linalg.det(mat) * coeff
        psi_mom[i_k] = val
    return psi_mom

class ArchitectEngine:
    def __init__(self, Lx=3, Ly=2, t1=0.2, t2=1.0, phi=0.0, V=2.0, flatten=True):
        self.Lx, self.Ly = Lx, Ly
        self.t1, self.t2 = t1, t2
        self.phi = phi
        self.V = V
        self.flatten = flatten # The Secret Weapon
        
        self.n_sites = 3 * Lx * Ly
        self.n_particles = Lx * Ly 
        
        print(f"--- ARCHITECT ENGINE ---")
        print(f"Lattice: {Lx}x{Ly} | V={V}")
        print(f"Strategy: {'FLAT BAND PROJECTION' if flatten else 'STANDARD'}")

        self.basis_real_list = list(itertools.combinations(range(self.n_sites), self.n_particles))
        self.dim = len(self.basis_real_list)
        self.map_real = {state: i for i, state in enumerate(self.basis_real_list)}
        self.basis_array = np.array(self.basis_real_list, dtype=np.int32)

    def build_hamiltonian(self):
        print("Building Optimized Hamiltonian...")
        start_t = time.time()
        
        # 1. Construct Single-Particle Hopping Matrix First
        H_sp = np.zeros((self.n_sites, self.n_sites), dtype=complex)
        p_ph = np.exp(1j * self.phi)
        m_ph = np.exp(-1j * self.phi)
        
        # Define Neighbors for V
        neighbors = []
        
        # --- Hopping Construction (Geometric) ---
        for x in range(self.Lx):
            for y in range(self.Ly):
                cell = y*self.Lx + x
                A,B,C = 3*cell, 3*cell+1, 3*cell+2
                
                # Intracell
                pairs = [(A,B), (B,C), (C,A)]
                for u,v in pairs:
                    H_sp[u,v] = -self.t1*p_ph; H_sp[v,u] = -self.t1*m_ph
                    neighbors.append(tuple(sorted((u,v))))

                # Intercell
                xp = (x-1)%self.Lx; yp = (y-1)%self.Ly
                xn = (x+1)%self.Lx
                
                # X-Dir
                B_prev = 3*(y*self.Lx+xp)+1
                H_sp[A,B_prev] = -self.t2*m_ph; H_sp[B_prev,A] = -self.t2*p_ph
                neighbors.append(tuple(sorted((A,B_prev))))
                
                # Y-Dir
                C_prev = 3*(yp*self.Lx+x)+2
                H_sp[B,C_prev] = -self.t2*m_ph; H_sp[C_prev,B] = -self.t2*p_ph
                neighbors.append(tuple(sorted((B,C_prev))))
                
                # Diagonal
                A_diag = 3*(yp*self.Lx+xn)
                H_sp[C,A_diag] = -self.t2*m_ph; H_sp[A_diag,C] = -self.t2*p_ph
                neighbors.append(tuple(sorted((C,A_diag))))
        
        # --- THE ARCHITECT'S FIX: FLATTENING ---
        if self.flatten:
            print("  > Flattening Bands (Removing Kinetic Dispersion)...")
            # Diagonalize Single Particle H
            evals, evecs = np.linalg.eigh(H_sp)
            # We want to keep the gap but flatten the occupied manifold.
            # Identify the gap (between band 1 and 2 for Kagome 1/3 filling?)
            # Kagome has 3 bands. Occupied is bottom 1/3 (Flat band usually at bottom or top?)
            # Standard Kagome flat band is TOP (E=2t). Fragile physics might be bottom.
            # We simply project: set all occupied energies to 0, all empty to 100.
            # Actually, better: Rescale H_sp so the occupied band has width ~ 0.
            
            # Simple Flattening: H_new = P_occ * 0 + P_unocc * Large
            # But we need H_sp for the MB build. 
            # We reconstruct H_sp from the flattened eigenvalues.
            
            n_occ = self.n_sites // 3 # 1/3 filling
            
            # Check gap
            gap = evals[n_occ] - evals[n_occ-1]
            print(f"  > Original Gap: {gap:.4f}")
            
            # Flatten: Set bottom n_occ eigenvalues to -1, others to +1
            flat_evals = np.ones(self.n_sites)
            flat_evals[:n_occ] = -1.0 
            
            # Rebuild H_sp = U Lambda U^dag
            H_sp = evecs @ np.diag(flat_evals) @ evecs.T.conj()
            print("  > Bands Flattened. Kinetic Energy minimized.")

        # 2. Build Many-Body H
        H = lil_matrix((self.dim, self.dim), dtype=complex)
        
        # Pre-cache non-zero H_sp elements for speed
        rows, cols = np.nonzero(np.abs(H_sp) > 1e-6)
        sp_vals = H_sp[rows, cols]
        sp_map = list(zip(rows, cols, sp_vals))

        # Fill Matrix
        for idx, state in enumerate(self.basis_real_list):
            state_set = set(state)
            
            # Kinetic (Flattened or Standard)
            for u, v, val in sp_map:
                if (u in state_set) and (v not in state_set):
                    new_state = list(state); new_state.remove(u); new_state.append(v); new_state.sort()
                    target = self.map_real[tuple(new_state)]
                    H[idx, target] += val
            
            # Interaction (Unchanged)
            e_int = 0.0
            for u, v in neighbors:
                if (u in state_set) and (v in state_set): e_int += self.V
            if e_int != 0: H[idx, idx] += e_int

        print(f"Hamiltonian Built in {time.time()-start_t:.2f}s")
        return H.tocsr()

    def get_ground_state(self):
        H = self.build_hamiltonian()
        print("Diagonalizing...")
        vals, vecs = eigsh(H, k=1, which='SA')
        return vecs[:, 0]

    def measure_es(self, psi_real):
        # Momentum ES Setup
        print("\nComputing Momentum-Space ES (Dual Scan)...")
        U_sp = np.zeros((self.n_sites, self.n_sites), dtype=complex)
        for r in range(self.n_sites):
            cell_r = r // 3; sub_r = r % 3
            rx, ry = cell_r % self.Lx, cell_r // self.Lx
            for k in range(self.n_sites):
                cell_k = k // 3; sub_k = k % 3
                if sub_r == sub_k:
                    kx = 2*np.pi*(cell_k % self.Lx)/self.Lx
                    ky = 2*np.pi*(cell_k // self.Lx)/self.Ly
                    phase = np.exp(-1j*(kx*rx + ky*ry))
                    U_sp[k, r] = phase / np.sqrt(self.Lx*self.Ly)

        print("  > Transforming Basis...")
        start_t = time.time()
        sig_indices = np.where(np.abs(psi_real) > 1e-6)[0].astype(np.int32)
        psi_mom = transform_basis_fast(psi_real, self.basis_array, U_sp, sig_indices)
        psi_mom /= np.linalg.norm(psi_mom)
        print(f"  > Done ({time.time()-start_t:.2f}s)")

        # Cut
        sites_A_mom = {k for k in range(self.n_sites) if ((k//3)%self.Lx) < self.Lx/2}
        
        # SVD
        target_NA = self.n_particles // 2
        blocks = {}
        for idx in range(len(psi_mom)):
            if abs(psi_mom[idx]) < 1e-10: continue
            state = self.basis_real_list[idx]
            sA = tuple(sorted([s for s in state if s in sites_A_mom]))
            if len(sA) == target_NA:
                sB = tuple(sorted([s for s in state if s not in sites_A_mom]))
                if sA not in blocks: blocks[sA] = {}
                blocks[sA][sB] = psi_mom[idx]
        
        if not blocks: return np.array([])
        
        # Matrix build
        uA = list(blocks.keys()); uB = set()
        for k in uA: uB.update(blocks[k].keys())
        uB = list(uB)
        mapA = {s:i for i,s in enumerate(uA)}; mapB = {s:i for i,s in enumerate(uB)}
        M = np.zeros((len(uA), len(uB)), dtype=complex)
        for sA, inner in blocks.items():
            r = mapA[sA]
            for sB, val in inner.items():
                M[r, mapB[sB]] = val
                
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        return -np.log(s[s>1e-12]**2)

if __name__ == "__main__":
    # Run Architect Engine with Flattening ENABLED
    engine = ArchitectEngine(Lx=3, Ly=2, t1=0.2, t2=1.0, phi=0.0, V=2.0, flatten=True)
    gs = engine.get_ground_state()
    es_mom = engine.measure_es(gs)
    
    plt.figure(figsize=(6, 5))
    if len(es_mom) > 0:
        plt.plot(np.arange(len(es_mom)), np.sort(es_mom), 'o-', c='red')
    plt.title("Momentum Space ES (Flat Band Limit)")
    plt.ylabel(r"$\xi$")
    plt.grid(True)
    plt.show()