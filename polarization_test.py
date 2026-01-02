import numpy as np
import itertools
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time

class PolarizationScanner:
    def __init__(self, Lx=3, Ly=2, t1=0.2, t2=1.0, phi=0.0, V=2.0):
        self.Lx, self.Ly = Lx, Ly
        self.t1, self.t2 = t1, t2 # t1 < t2 = Fragile Phase
        self.phi = phi
        self.V = V
        
        self.n_sites = 3 * Lx * Ly
        self.n_particles = Lx * Ly 
        
        # Basis
        self.basis = list(itertools.combinations(range(self.n_sites), self.n_particles))
        self.dim = len(self.basis)
        self.map = {state: i for i, state in enumerate(self.basis)}
        
        print(f"System: {Lx}x{Ly} | V={V} | Dim: {self.dim}")

    def get_coords(self, site_idx):
        """Returns physical (x,y) coordinates of a site."""
        cell = site_idx // 3
        sub = site_idx % 3
        cx, cy = cell % self.Lx, cell // self.Lx
        
        # Kagome Sublattice offsets (roughly)
        # A=(0,0), B=(0.5, 0), C=(0.25, 0.433)
        if sub == 0: off = np.array([0.0, 0.0])
        elif sub == 1: off = np.array([0.5, 0.0])
        else: off = np.array([0.25, np.sqrt(3)/4])
        
        return np.array([cx, cy]) + off

    def build_hamiltonian(self):
        print("Building H...")
        H = lil_matrix((self.dim, self.dim), dtype=complex)
        
        p_ph = np.exp(1j * self.phi); m_ph = np.exp(-1j * self.phi)
        
        hoppings = []
        neighbors = []
        
        for x in range(self.Lx):
            for y in range(self.Ly):
                cell = y*self.Lx + x
                A,B,C = 3*cell, 3*cell+1, 3*cell+2
                
                # t1 (Intra)
                hoppings.extend([(A,B,-self.t1*p_ph), (B,C,-self.t1*p_ph), (C,A,-self.t1*p_ph)])
                hoppings.extend([(B,A,-self.t1*m_ph), (C,B,-self.t1*m_ph), (A,C,-self.t1*m_ph)])
                neighbors.extend([tuple(sorted((A,B))), tuple(sorted((B,C))), tuple(sorted((C,A)))])
                
                # t2 (Inter - Periodic)
                xp = (x-1)%self.Lx; yp = (y-1)%self.Ly
                B_prev = 3*(y*self.Lx+xp)+1
                C_prev = 3*(yp*self.Lx+x)+2
                xn = (x+1)%self.Lx
                A_diag = 3*(yp*self.Lx+xn)
                
                hoppings.extend([(A,B_prev,-self.t2*m_ph), (B_prev,A,-self.t2*p_ph)])
                hoppings.extend([(B,C_prev,-self.t2*m_ph), (C_prev,B,-self.t2*p_ph)])
                hoppings.extend([(C,A_diag,-self.t2*m_ph), (A_diag,C,-self.t2*p_ph)])
                
                neighbors.extend([tuple(sorted((A,B_prev))), tuple(sorted((B,C_prev))), tuple(sorted((C,A_diag)))])

        # Fill H
        for idx, state in enumerate(self.basis):
            s_set = set(state)
            # Kinetic
            for u,v,val in hoppings:
                if (u in s_set) and (v not in s_set):
                    new_s = list(state); new_s.remove(u); new_s.append(v); new_s.sort()
                    target = self.map[tuple(new_s)]
                    H[idx, target] += val
            # Interaction
            e = sum(self.V for u,v in neighbors if u in s_set and v in s_set)
            if e!=0: H[idx, idx] += e
            
        return H.tocsr()

    def calculate_resta_polarization(self, psi):
        """
        Calculates Many-Body Polarization P = Im ln <Psi | exp(i 2pi X / Lx) | Psi>
        This detects if electrons are centered on atoms (P=0) or obstructed (P!=0).
        """
        print("Calculating Polarization...")
        
        # X-Polarization Operator: U_x = exp(i * 2pi * X_total / Lx)
        expectation_val = 0.0 + 0j
        
        for idx, coeff in enumerate(psi):
            state = self.basis[idx]
            if abs(coeff) < 1e-10: continue
            
            prob = abs(coeff)**2
            
            # Calculate Total X position of this configuration
            X_total = 0.0
            for site in state:
                coords = self.get_coords(site)
                X_total += coords[0] # X component
            
            # Resta Operator eigenvalue
            op_val = np.exp(1j * 2 * np.pi * X_total / self.Lx)
            expectation_val += prob * op_val
            
        # Polarization P = (Lx / 2pi) * Im(ln <U>)
        angle = np.angle(expectation_val)
        P = angle / (2 * np.pi)
        
        return P, abs(expectation_val)

if __name__ == "__main__":
    print("--- INVENTION OFFICER RED TEAM ---")
    
    # 1. Trivial Case
    print("\n[Control] Simulating Trivial Phase (t1=1.0, t2=0.2)...")
    eng_triv = PolarizationScanner(Lx=3, Ly=2, t1=1.0, t2=0.2, V=2.0)
    gs_triv = eigsh(eng_triv.build_hamiltonian(), k=1, which='SA')[1][:,0]
    P_triv, Mag_triv = eng_triv.calculate_resta_polarization(gs_triv)
    
    # 2. Fragile Case
    print("\n[Experiment] Simulating Fragile Phase (t1=0.2, t2=1.0)...")
    eng_frag = PolarizationScanner(Lx=3, Ly=2, t1=0.2, t2=1.0, V=2.0)
    gs_frag = eigsh(eng_frag.build_hamiltonian(), k=1, which='SA')[1][:,0]
    P_frag, Mag_frag = eng_frag.calculate_resta_polarization(gs_frag)
    
    print("\n--- RESULTS ---")
    print(f"Trivial Polarization: {P_triv:.4f}")
    print(f"Fragile Polarization: {P_frag:.4f}")
    
    print("\n--- INTERPRETATION ---")
    if abs(P_frag) > 0.1 and abs(P_triv) < 0.1:
        print("SUCCESS: The Fragile phase is NOT trivial.")
        print("The electrons are locked in an 'Obstructed' position.")
    else:
        print("FAILURE: Both phases are trivial.")