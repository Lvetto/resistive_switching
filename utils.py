import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, factorized
from matplotlib import pyplot as plt
from noise import pnoise2

def generate_film_random(size, threshold=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    film = np.random.rand(size, size) > threshold

    return film.astype(bool)

def generate_film_perlin(size, scale=10, threshold=0.5, seed=None):
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    if seed is None:
        seed = np.random.randint(0, 100)
    
    perlin_noise = np.vectorize(lambda i, j: pnoise2(i / scale, j / scale, octaves=6, base=seed))(x, y)

    film = perlin_noise > threshold

    return film.astype(bool)

class Film:
    def __init__(self, map, resistence_values=(1.0, 10**(6), 10**(7)), base_position=(0,0)):
        self.map = map
        self.size = map.shape
        self.resistence_values = resistence_values

        self.base_position = np.array(base_position)
        
        #self.num_nodes = self.size * self.size
        #self.num_edges = 2 * self.num_nodes - 2 * self.size
        
        #self.conductance_matrix = self._build_conductance_matrix()
    
    def split(self, size):
        sub_grid_size = size - 1  # Overlap by 1 row/column

        regions = []

        #edge_indices = [[] for _ in range(0, self.size, sub_grid_size)]

        for i in range(0, self.size[0], sub_grid_size):
            for j in range(0, self.size[1], sub_grid_size):
                i_end = min(i + size, self.size[0])
                j_end = min(j + size, self.size[1])
                region_map = self.map[i:i_end, j:j_end]
                base_pos = (j, i) + self.base_position
                regions.append(Film(region_map, self.resistence_values, base_position=base_pos))
                #edge_indices[j // sub_grid_size].append(((i, i_end), (j, j_end)))

        return regions
    
    @property
    def adj_matrices(self):
        if not hasattr(self, '_adj_matrices'):
            self.build_adjacency_matrix()
        return self._adj_matrices

    def build_adjacency_matrix(self):
        # build multiple adj matrices, representing different connections (metal-metal, metal-dielectric, dielectric-dielectric)

        H, W = self.map.shape
        N = H * W

        node_ids = np.arange(N).reshape(H, W)

        h_links_mm = self.map[:, :-1] & self.map[:, 1:]
        v_links_mm = self.map[:-1, :] & self.map[1:, :]

        h_links_md = (self.map[:, :-1] != self.map[:, 1:])
        v_links_md = (self.map[:-1, :] != self.map[1:, :])

        h_links_dd = (~self.map[:, :-1]) & (~self.map[:, 1:])
        v_links_dd = (~self.map[:-1, :]) & (~self.map[1:, :])

        links = [(h_links_mm, v_links_mm), (h_links_md, v_links_md), (h_links_dd, v_links_dd)]

        adj_mats = []

        for h_link, v_link in links:
            adjacency_matrix = lil_matrix((N, N), dtype=bool)

            h_link_inds = node_ids[:, :-1][h_link]
            v_link_inds = node_ids[:-1, :][v_link]

            adjacency_matrix[h_link_inds, h_link_inds + 1] = 1
            adjacency_matrix[h_link_inds + 1, h_link_inds] = 1
            adjacency_matrix[v_link_inds, v_link_inds + W] = 1
            adjacency_matrix[v_link_inds + W, v_link_inds] = 1

            adj_mats.append(csr_matrix(adjacency_matrix))
        
        self._adj_matrices = adj_mats

        return adj_mats
    
    @property
    def conductance_matrix(self):
        if not hasattr(self, '_conductance_matrix'):
            self.build_conductance_matrix()
        return self._conductance_matrix

    def build_conductance_matrix(self):
        mm_resistance, md_resistance, dd_resistance = self.resistence_values

        conductance_matrix = lil_matrix(self.adj_matrices[0].shape, dtype=float)

        conductance_matrix += self.adj_matrices[0] * (-1 / mm_resistance)
        conductance_matrix += self.adj_matrices[1] * (-1 / md_resistance)
        conductance_matrix += self.adj_matrices[2] * (-1 / dd_resistance)

        diagonal_values = np.array(-conductance_matrix.sum(axis=1)).flatten()
        conductance_matrix.setdiag(diagonal_values)

        conductance_matrix = csr_matrix(conductance_matrix)

        self._conductance_matrix = conductance_matrix

        return conductance_matrix

    @property
    def network(self):
        if not hasattr(self, '_network'):
            self.build_network()
        return self._network

    def build_network(self):
        conductance_matrix = self.build_conductance_matrix()

        # an array mapping each node to its (x, y) position
        positions = self.base_position + np.array([[j, i] for i in range(self.size[0]) for j in range(self.size[1])])

        self._network = Network(conductance_matrix, node_positions=positions)
        return self._network

class Network:
    def __init__(self, conductance_matrix, node_positions=None):

        self.conductance_matrix = conductance_matrix
        self.num_nodes = conductance_matrix.shape[0]

        self.node_positions = node_positions

    def node_mapping(self, x):
        if self.node_positions is None:
            return x
        else:
            return self.node_positions[x]

    def reduce(self, keep_mask, regularization=1e-12):

        remove_mask = ~keep_mask

        keep_idx = np.where(keep_mask)[0]
        remove_idx = np.where(remove_mask)[0]

        G = self.conductance_matrix

        G_BB = G[keep_idx, :][:, keep_idx]
        G_BI = G[keep_idx, :][:, remove_idx]
        G_IB = G[remove_idx, :][:, keep_idx]
        G_II = G[remove_idx, :][:, remove_idx]

        # add a small regularization to the diagonal of G_II to ensure it's invertible
        G_II = G_II + regularization * csr_matrix(np.eye(G_II.shape[0]))

        solve_G_II = factorized(G_II.tocsc()) 
        Y = solve_G_II(G_IB.toarray()) 
        correction_term = G_BI @ Y 
        G_schur = G_BB - correction_term

        reduced_network = Network(G_schur, node_positions=self.node_mapping(keep_idx))

        return reduced_network

    def join_networks(self, other_networks):
        networks = [self, *other_networks]

        all_positions = np.vstack([net.node_positions for net in networks])
        unique_positions, inverse_indices = np.unique(all_positions, axis=0, return_inverse=True)

        global_indices_per_network = []
        start = 0
        for net in networks:
            n = net.node_positions.shape[0]
            global_indices_per_network.append(inverse_indices[start:start+n])
            start += n

        N = unique_positions.shape[0]
        global_conductance = lil_matrix((N, N), dtype=float)

        for net, global_idx in zip(networks, global_indices_per_network):
            local_G = net.conductance_matrix
            for i_local, i_global in enumerate(global_idx):
                for j_local, j_global in enumerate(global_idx):
                    global_conductance[i_global, j_global] += local_G[i_local, j_local]
        
        combine_network = Network(csr_matrix(global_conductance), node_positions=unique_positions)

        return combine_network

    def copy(self):
        return Network(self.conductance_matrix.copy(), node_positions=self.node_positions.copy() if self.node_positions is not None else None)
    
    def solve(self, bound_nodes, bound_values, regularization=1e-12):
        G = self.conductance_matrix
        num_points = G.shape[0]

        bound_indices = np.array(bound_nodes)
        free_indices = np.setdiff1d(np.arange(num_points), bound_indices)

        G_BB = G[bound_indices[:, None], bound_indices]
        G_BF = G[bound_indices[:, None], free_indices]
        G_FB = G[free_indices[:, None], bound_indices]
        G_FF = G[free_indices[:, None], free_indices]

        G_FF = G_FF + regularization * csr_matrix(np.eye(G_FF.shape[0]))  # regularization to avoid singular matrix
        G_FB = G_FB + regularization * csr_matrix(np.eye(G_FB.shape[0], G_FB.shape[1]))  # regularization to avoid singular matrix

        V_B = np.array(bound_values)

        b = -G_FB @ V_B
        V_free = spsolve(G_FF, b)

        # combine the potentials into a single array
        V = np.zeros(num_points)
        V[bound_indices] = V_B
        V[free_indices] = V_free

        return V

    def compute_currents(self, potentials):
        G = self.conductance_matrix
        I = G @ potentials
        return I

def intersect_networks_indices(networks):
    all_positions = np.vstack([net.node_positions for net in networks])
    unique_positions, inverse_indices, counts = np.unique(all_positions, axis=0, return_inverse=True, return_counts=True)

    intersect_positions = unique_positions[counts > 1]

    intersect_indices_per_network = []
    start = 0
    for net in networks:
        n = net.node_positions.shape[0]
        net_indices = inverse_indices[start:start+n]
        intersect_indices = [i for i, pos in enumerate(net_indices) if unique_positions[pos] in intersect_positions]
        intersect_indices_per_network.append(intersect_indices)
        start += n

    return intersect_indices_per_network

def plot_network(network, ax=None, dot_color="blue", line_color='gray'):
    if ax is None:
        fig, ax = plt.subplots()

    positions = network.node_positions

    G = network.conductance_matrix

    gmax = np.max(G)
    gmin = np.min(G)
    #print(gmin, gmax)

    for i in range(network.num_nodes):
        for j in range(network.num_nodes):
            v = G[i, j]
            if i < j and v < 0:
                pos_i = positions[i]
                pos_j = positions[j]
                alpha = v / gmin    #(v - gmin) / (gmax - gmin + 1e-12)
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], color=line_color, linewidth=0.5, alpha=alpha)

    ax.scatter(positions[:, 0], positions[:, 1], s=10, color=dot_color)

    return ax

class ExperimentSimulation:
    def __init__(self, film_matrix, resistence_values=(1.0, 10**(6), 10**(7)), split_size=10):

        self.film = Film(film_matrix, resistence_values)

        self.split_size = split_size
        
        # Initialize some values to None, they will be computed when needed (they use a lot of memory and time)
        self._film_regions = None
        self._region_networks = None
        self._reduced_local_networks = []
        self._reduced_regions_network = None

    @property
    def film_regions(self):
         if self._film_regions is not None:
             return self._film_regions
         else:
             self._film_regions = self.film.split(self.split_size)
             return self._film_regions
    
    @property
    def region_networks(self):
        if self._region_networks is not None:
            return self._region_networks
        else:
            self._region_networks = [region.network for region in self.film_regions]
            return self._region_networks
    
    @property
    def reduced_regions_network(self):
        if self._reduced_regions_network is not None:
            return self._reduced_regions_network
        else:
            self._reduced_local_networks = []
            for region in self.film_regions:
                H, W = region.size
                
                node_ids = np.arange(region.network.num_nodes)

                # top and bottom rows
                mask = node_ids % W == 0
                mask |= node_ids % W == W - 1

                # left and right columns
                mask |= node_ids // W == 0
                mask |= node_ids // W == H - 1

                self._reduced_local_networks.append(region.network.reduce(mask))
            self._reduced_regions_network = self._reduced_local_networks[0].join_networks(self._reduced_local_networks[1:])
            return self._reduced_regions_network

    def solve_potentials(self, v_biases, bound_nodes):
        V_reduced, solved_positions = self.solve_reduced_potentials(v_biases, bound_nodes)

        V_full = np.zeros_like(self.film.map, dtype=float)
        count_full = np.zeros_like(self.film.map, dtype=float)

        for region in self.film_regions:
            net = region.network
            net_pos = net.node_positions  # shape (N, 2), (x, y) global positions

            # find the corresponding nodes in the reduced network
            mask = np.array([np.any(np.all(solved_positions == pos, axis=1)) for pos in net_pos])
            bound_nodes = np.where(mask)[0]
            bound_values = V_reduced[[np.where(np.all(solved_positions == pos, axis=1))[0][0] for pos in net_pos[mask]]]

            V_region = net.solve(bound_nodes, bound_values)

            # For each node in the region, add its value to the correct (y, x) in V_full
            for idx, (x, y) in enumerate(net_pos):
                xg, yg = int(x), int(y)
                if 0 <= yg < V_full.shape[0] and 0 <= xg < V_full.shape[1]:
                    V_full[yg, xg] += V_region[idx]
                    count_full[yg, xg] += 1

        # Avoid division by zero
        mask = count_full > 0
        V_full[mask] /= count_full[mask]

        return V_full

    def solve_reduced_potentials(self, v_biases, bound_nodes):
        bound_values = np.array(v_biases)

        V_reduced = self.reduced_regions_network.solve(bound_nodes, bound_values)
        solved_positions = self.reduced_regions_network.node_positions

        return V_reduced, solved_positions

def apply_lr_electrodes_and_solve(sim, v_bias):
    inds = np.arange(sim.reduced_regions_network.num_nodes)
    positions = sim.reduced_regions_network.node_mapping(inds)

    # electrodes are placed on the left and grounds on the right
    left_electrodes = np.where(positions[:, 0] < 1)[0]
    right_grounds = np.where(positions[:, 0] > sim.film.size[1] - 2)[0]

    bound_nodes = np.concatenate((left_electrodes, right_grounds))

    V_biases = np.zeros(len(left_electrodes) + len(right_grounds))
    V_biases[:len(left_electrodes)] = 1.0  # set left electrodes to 1V

    #V_reduced, solved_positions = sim.solve_reduced_potentials(V_biases, bound_nodes)
    V_solution = sim.solve_potentials(V_biases, bound_nodes)
    return V_solution

