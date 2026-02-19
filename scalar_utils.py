from utils import *
import numpy as np
from scipy.signal import convolve2d

class ScalarMesher(GridMesher):
    def __init__(self, film_map, conductance_func):
        self.conductance_func = conductance_func
        super().__init__(film_map, resistances=None) 
    
    def _compute_conductance(self, val_a, val_b):
        """
        Internal use method to compute the conductance between two adjacent cells based on their scalar values using the provided conductance function.
        """
        return self.conductance_func(val_a, val_b)

class ScalarSolver(Solver):
    def __init__(self, film_map, conductance_func, split_size=50):

        self.mesher = ScalarMesher(film_map, conductance_func)
        self.split_size = split_size
        self.shape = film_map.shape

        self.n_chunks_y = (self.shape[0] + split_size - 1) // split_size
        self.n_chunks_x = (self.shape[1] + split_size - 1) // split_size

        self.sub_networks = [None] * (self.n_chunks_y * self.n_chunks_x)    # Original local networks
        self.reduced_networks = [None] * (self.n_chunks_y * self.n_chunks_x)    # Reduced networks
        
        self.coarse_network = None
        self.full_solution = None
        
        self._build_hierarchy()

# not really necessary
class ScalarAnalysis(ElectricalAnalysis):
    def __init__(self, solver):
        super().__init__(solver)
    
    def compute_maps(self):
        maps = super().compute_maps()
        
        # compute the gradients needed fo the flux function
        self.grad_P = np.gradient(maps['power_density'])
        maps['grad_P_x'] = self.grad_P[0]
        maps['grad_P_y'] = self.grad_P[1]

        return maps


class ScalarSimulation(Simulation):
    def __init__(self, film_map, conductance_func=None, flux_func=None, split_size=50, dbg=False):
        self.conductance_func = conductance_func if conductance_func is not None else self.default_conductance_func
        self.flux_func = flux_func if flux_func is not None else self.default_flux_func
        
        self.solver = ScalarSolver(film_map, self.conductance_func, split_size)
        self.analysis = ScalarAnalysis(self.solver)
        self.film_map = film_map.astype(float)
        self.maps = None
    
    def default_flux_func(self, grad_rho, E, P_avg, grad_P, rho):
        """
        Default flux function that computes the mass flux based on the local densities, electric field, power density, and their gradients.
        This is a simple example and can be replaced with a more complex function as needed.
        """
        
        """mu_0 = 1.0  # Base mobility
        gamma = 0.1   # Exponent coeffifient
        P_ref = 1.0  # Reference power density for scaling
        mobility = 10#*mu_0 * np.exp(gamma * (np.sum(P_avg) / (P_ref * P_avg.size)))  # Diffusion coefficient that increases with power density
        """
        
        grad_P_clamped = np.tanh(grad_P * 0.1) * 1  # Clamp the gradient to prevent excessive fluxes

        k_e = 3  # Coefficient for electric field contribution
        k_d = 0.1  # Coefficient for diffusion contribution
        k_p = 1  # Coefficient for power-driven contribution

        E_flux = k_e * E * rho  # Drift flux component
        D_flux = -k_d * grad_rho  # Diffusion flux component
        P_flux = -k_p * grad_P * rho  # Power-driven flux component

        return E_flux + D_flux + P_flux
    
    def default_conductance_func(self, val_a, val_b, k=10):
        """
        Default conductance function that computes the conductance between two adjacent cells based on their scalar values.
        This is a simple example and can be replaced with a more complex function as needed.
        """

        out = 0.1 + k * (val_a + val_b) / 2.0  # Average value as conductance, scaled by k
        #out = 0.1 + np.exp(k * (val_a + val_b) / 2.0 - k)  # Exponential function to create sharper contrast between low and high values
        
        return out

    def _compute_mass_fluxes(self, V_map, P_map):
        """
        Compute the mass fluxes on the links between nodes based on the local densities, electric field, power density, and their gradients using the provided flux function.
        """

        rho = self.film_map

        # Avg power density on the links
        P_avg_x = 0.5 * (P_map[:, 1:] + P_map[:, :-1])
        P_avg_y = 0.5 * (P_map[1:, :] + P_map[:-1, :])

        # Avg density on the links
        rho_avg_x = 0.5 * (rho[:, 1:] + rho[:, :-1])
        rho_avg_y = 0.5 * (rho[1:, :] + rho[:-1, :])

        # Electric field components (pixel centered)
        E_x, E_y = sobel_gradient(V_map)

        # P gradient on the links (pixel centered)
        grad_P_x, grad_P_y = sobel_gradient(P_map)

        # Density gradient on the links (pixel centered)
        grad_rho_x, grad_rho_y = sobel_gradient(rho)

        # compute avg gradients on the links
        
        grad_rho_x_avg = 0.5 * (grad_rho_x[:, 1:] + grad_rho_x[:, :-1])
        grad_rho_y_avg = 0.5 * (grad_rho_y[1:, :] + grad_rho_y[:-1, :])

        grad_P_x_avg = 0.5 * (grad_P_x[:, 1:] + grad_P_x[:, :-1])
        grad_P_y_avg = 0.5 * (grad_P_y[1:, :] + grad_P_y[:-1, :])

        E_x_avg = 0.5 * (E_x[:, 1:] + E_x[:, :-1])
        E_y_avg = 0.5 * (E_y[1:, :] + E_y[:-1, :])

        # Compute the fluxes using the parametric function
        J_x_internal = self.flux_func(grad_rho_x_avg, E_x_avg, P_avg_x, grad_P_x_avg, rho_avg_x)
        J_y_internal = self.flux_func(grad_rho_y_avg, E_y_avg, P_avg_y, grad_P_y_avg, rho_avg_y)

        return J_x_internal, J_y_internal
    
    def _evolve_step(self, dt, bias_v=1.0, adaptive_dt=False):
        """
        Perform a single time step of the simulation by computing the electric potential, power density, mass fluxes, and updating the film map based on the divergence of the fluxes.
        """

        electrodes, grounds, all_bound = self.get_electrodes_coords()
        bound_values = np.concatenate((np.full(len(electrodes), bias_v), np.zeros(len(grounds))))
        #self.solver.solve(all_bound, bound_values)
        
        elec_maps = self.solve_all(all_bound, bound_values)

        #elec_maps = self.analysis.compute_maps()
        elec_maps['film_map'] = self.film_map.copy()  # Include the current film map in the output maps
        V_map = self.solver.full_solution
        P_map = elec_maps['power_density']
        
        J_x_int, J_y_int = self._compute_mass_fluxes(V_map, P_map)
        
        # Pad the arrays to keep the original shape for divergence calculation
        J_x_pad = np.pad(J_x_int, ((0, 0), (1, 1)), mode='edge')
        J_y_pad = np.pad(J_y_int, ((1, 1), (0, 0)), mode='edge')
        
        # Divergence
        div_x = J_x_pad[:, 1:] - J_x_pad[:, :-1]
        div_y = J_y_pad[1:, :] - J_y_pad[:-1, :]
        divergence = div_x + div_y

        # Set divergence to zero at the boundaries to prevent unphysical fluxes
        divergence[0, :] = 0
        divergence[-1, :] = 0
        divergence[:, 0] = 0
        divergence[:, -1] = 0

        max_flux = np.max(np.abs(J_x_int)) + np.max(np.abs(J_y_int)) + 1e-12 # epsilon per evitare div by zero
        
        if adaptive_dt:
            # CFL condition for stability: dt < dx / max_flux, assuming dx=1
            dt_target = dt
            dt_cfl = 0.1 / max_flux 
            actual_dt = min(dt_target, dt_cfl)
            
            if actual_dt < 1e-9:
                print("Warning: dt is very small due to high fluxes, simulation may be unstable. Consider increasing dt or adjusting the flux function.")
        else:
            actual_dt = dt
        
        # time integration with Neumann boundary conditions (zero flux at boundaries)
        #self.film_map -= divergence * actual_dt
        #self.film_map[1:-1, 1:-1] -= divergence[1:-1, 1:-1] * actual_dt
        self.film_map[1:-1, 1:-1] -= divergence[1:-1, 1:-1] * actual_dt
        self.film_map[0, :] = self.film_map[1, :]
        self.film_map[-1, :] = self.film_map[-2, :]
        self.film_map[:, 0] = self.film_map[:, 1]
        self.film_map[:, -1] = self.film_map[:, -2]

        # Clip the film map to keep values within [0, 1]
        self.film_map = np.clip(self.film_map, 0.0, 1.0)
        
        # build updated solver and analysis with the new film map
        self.solver = ScalarSolver(self.film_map, self.conductance_func, 
                                   split_size=self.solver.split_size)
        self.analysis = ElectricalAnalysis(self.solver)

        return elec_maps

    def solve_all(self, bound_coords, bound_values):
        out = super().solve_all(bound_coords, bound_values)
        
        return out


def sobel_gradient(field):
    """
    Compute the gradient of a 2D scalar field using the Sobel operator, which is more robust to noise and captures edge information better than simple finite differences.
    Returns the gradients in the x and y directions.
    """

    Kx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]]) / 8.0
                   
    Ky = np.array([[-1, -2, -1], 
                   [0,  0,  0], 
                   [1,  2,  1]]) / 8.0

    grad_x = convolve2d(field, Kx, mode='same', boundary='symm')
    grad_y = convolve2d(field, Ky, mode='same', boundary='symm')
    
    return grad_x, grad_y