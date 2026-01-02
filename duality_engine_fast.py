import numpy as np
import itertools
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time
from numba import jit, prange

# --- JIT COMPILED CORE (The Speed Boost) ---
@jit(nopython=True, parallel=True, fastmath=True)
def transform_basis_fast(psi_real, basis_array, U_sp, sig_indices):
    """
    Compiles the loop to Machine Code. 
    Runs in parallel on all CPU cores.
    """
    dim = len(basis_array)
    psi_mom = np.zeros(dim, dtype=np.complex128)
    n_sig = len(sig_indices)
    
    # Pre-allocate small matrix for determinant calculation to avoid memory churn
    # Note: In parallel loops, we can't share temp buffers easily without thread-local storage.
    # Numba handles allocation reasonably well for small fixed sizes.
    
    for i_k in prange(dim): # Parallel loop over Momentum States
        state_k = basis_array[i_k]
        val = 0.0 + 0j
        
        # Loop only over significant components of Real State
        for idx_r in range(n_sig):
            i_r = sig_indices[idx_r]
            state_r = basis_array[i_r]
            coeff = psi_real[i_r]
            
            # Construct 6x6 matrix manually (Numba doesn't like np.ix_)
            # We assume N=6 particles for 3x2 lattice
            mat = np.zeros((6, 6), dtype=np.complex128)
            for r in range(6):
                row_idx = state_k[r]
                for c in range(6):
                    col_idx = state_r[c]
                    mat[r, c] = U_sp[row_idx, col_idx]
            
            # Calculate Determinant
            det = np.linalg.det(mat)
            val += det * coeff
            
        psi_mom[i_k] = val
        
    return psi_mom

class DualityEngine:
    def __init__(self, Lx=3, Ly=2, t1=0.2, t2=1.0, phi=0.0, V=2.0):
        self.Lx, self.Ly = Lx, Ly
        self.t1, self.t2 = t1, t2
        self.phi = phi
        self.V = V
        
        self.n_sites = 3 * Lx * Ly
        self.n_particles = Lx * Ly 
        
        print(f"--- SYSTEM SETUP ---")
        print(f"Lattice: {Lx}x{Ly} ({self.n_sites} sites)")
        print(f"Particles: {self.n_particles}")
        
        # Basis construction
        self.basis_real_list = list(itertools.combinations(range(self.n_sites), self.n_particles))
        self.dim = len(self.basis_real_list)
        self.map_real = {state: i for i, state in enumerate(self.basis_real_list)}
        
        # Convert Basis to Numpy Array for Numba (Crucial Step)
        self.basis_array = np.array(self.basis_real_list, dtype=np.int32)
        
        print(f"Hilbert Space Dimension: {self.dim}")

    def build_hamiltonian(self):
        print("Building Hamiltonian...")
        start_t = time.time()
        H = lil_matrix((self.dim, self.dim), dtype=complex)
        
        # --- PRECOMPUTE HOPPINGS ---
        hoppings = []
        neighbors = []
        p_ph = np.exp(1j * self.phi)
        m_ph = np.exp(-1j * self.phi)
        
        for x in range(self.Lx):
            for y in range(self.Ly):
                cell = y*self.Lx + x
                A, B, C = 3*cell, 3*cell+1, 3*cell+2
                
                # Intracell
                hoppings.extend([(A,B, -self.t1*p_ph), (B,A, -self.t1*m_ph)])
                hoppings.extend([(B,C, -self.t1*p_ph), (C,B, -self.t1*m_ph)])
                hoppings.extend([(C,A, -self.t1*p_ph), (A,C, -self.t1*m_ph)])
                neighbors.extend([tuple(sorted((A,B))), tuple(sorted((B,C))), tuple(sorted((C,A)))])

                # Intercell (Periodic)
                # X-Dir
                x_prev = (x - 1) % self.Lx; B_prev = 3*(y*self.Lx + x_prev) + 1
                hoppings.extend([(A, B_prev, -self.t2*m_ph), (B_prev, A, -self.t2*p_ph)])
                neighbors.append(tuple(sorted((A, B_prev))))
                
                # Y-Dir
                y_prev = (y - 1) % self.Ly; C_prev = 3*(y_prev*self.Lx + x) + 2
                hoppings.extend([(B, C_prev, -self.t2*m_ph), (C_prev, B, -self.t2*p_ph)])
                neighbors.append(tuple(sorted((B, C_prev))))
                
                # Diagonal
                x_next = (x + 1) % self.Lx; A_diag = 3*(y_prev*self.Lx + x_next)
                hoppings.extend([(C, A_diag, -self.t2*m_ph), (A_diag, C, -self.t2*p_ph)])
                neighbors.append(tuple(sorted((C, A_diag))))

        # --- FILL MATRIX ---
        for idx, state in enumerate(self.basis_real_list):
            state_set = set(state)
            
            # Kinetic
            for u, v, val in hoppings:
                if (u in state_set) and (v not in state_set):
                    new_state = list(state)
                    new_state.remove(u); new_state.append(v)
                    new_state.sort()
                    target = self.map_real[tuple(new_state)]
                    H[idx, target] += val
            
            # Interaction
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
        print(f"Ground State Energy: {vals[0]:.4f}")
        return vecs[:, 0]

    def transform_and_measure_es(self, psi_real):
        # 1. Real Space ES
        print("\nComputing Real-Space Entanglement Spectrum...")
        sites_A_real = {i for i in range(self.n_sites) if (i//3)%self.Lx < self.Lx/2}
        es_real = self._compute_es(psi_real, self.basis_real_list, sites_A_real)
        
        # 2. Momentum Space ES
        print("\nComputing Momentum-Space Entanglement Spectrum...")
        print("  > Building U_sp matrix...")
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

        print("  > Starting JIT-Compiled Basis Transform (This is fast)...")
        start_t = time.time()
        
        # Identify significant components
        sig_indices = np.where(np.abs(psi_real) > 1e-6)[0]
        # Ensure correct types for Numba
        sig_indices = sig_indices.astype(np.int32)
        
        # CALL NUMBA FUNCTION
        psi_mom = transform_basis_fast(psi_real, self.basis_array, U_sp, sig_indices)
        
        # Normalize
        norm = np.linalg.norm(psi_mom)
        if norm > 1e-10: psi_mom /= norm
        
        print(f"  > Transform complete in {time.time()-start_t:.2f}s")
        
        sites_A_mom = {k for k in range(self.n_sites) if ((k//3)%self.Lx) < self.Lx/2}
        es_mom = self._compute_es(psi_mom, self.basis_real_list, sites_A_mom)
        
        return es_real, es_mom

    def _compute_es(self, psi, basis_list, sites_A):
        target_NA = self.n_particles // 2
        blocks = {}
        
        # We iterate over indices to avoid object creation overhead
        for idx in range(len(psi)):
            coeff = psi[idx]
            if abs(coeff) < 1e-10: continue
            
            state = basis_list[idx]
            state_A = tuple(sorted([s for s in state if s in sites_A]))
            
            if len(state_A) == target_NA:
                state_B = tuple(sorted([s for s in state if s not in sites_A]))
                if state_A not in blocks: blocks[state_A] = {}
                blocks[state_A][state_B] = coeff
        
        if not blocks: return np.array([])
        
        # Convert to matrix
        uA = list(blocks.keys()); uB = set()
        for k in uA: uB.update(blocks[k].keys())
        uB = list(uB)
        mapA = {s:i for i,s in enumerate(uA)}; mapB = {s:i for i,s in enumerate(uB)}
        
        M = np.zeros((len(uA), len(uB)), dtype=complex)
        for sA, inner in blocks.items():
            r = mapA[sA]
            for sB, val in inner.items():
                c = mapB[sB]
                M[r,c] = val
                
        try:
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
            s = s[s > 1e-12]
            return -np.log(s**2)
        except: return np.array([])

if __name__ == "__main__":
    # Settings
    engine = DualityEngine(Lx=3, Ly=2, t1=0.2, t2=1.0, phi=0.0, V=2.0)
    
    gs = engine.get_ground_state()
    es_real, es_mom = engine.transform_and_measure_es(gs)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if len(es_real) > 0:
        plt.plot(np.arange(len(es_real)), np.sort(es_real), 'o-', c='gray')
    plt.title("Real Space ES")
    plt.ylabel(r"Entanglement Energy $\xi$")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if len(es_mom) > 0:
        plt.plot(np.arange(len(es_mom)), np.sort(es_mom), 'o-', c='red')
    plt.title("Momentum Space ES")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()