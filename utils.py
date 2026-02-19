from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from noise import pnoise2
from time import time
import numpy as np
from scipy.sparse import coo_matrix, eye, triu
from scipy.sparse.linalg import spsolve, factorized

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        time_taken = end_time - start_time
        print(f"Function '{func.__name__}' executed in {time_taken:.4f} seconds")
        return time_taken, result
    return wrapper

class LinearNetwork:
    """
    Abstraction for a generic linear resistive network represented by its conductance matrix.
    Nodes are indexed locally (0..N-1) within the network, but can be mapped to global indices, if a global map is provided.
    """
    def __init__(self, conductance_matrix, global_indices=None):
        """
        Initialize a linear network with a given conductance matrix and optional global index mapping.

        :param conductance_matrix: the conductance matrix (sparse csr_matrix)
        :param global_indices: array mapping local node indices to global indices
        """
        self.G = conductance_matrix
        self.num_nodes = self.G.shape[0]

        # global_indices maps local indices 0..N to unique global problem indices
        self.global_indices = global_indices if global_indices is not None else np.arange(self.num_nodes)

    def reduce(self, keep_mask, regularization=1e-12):
        """
        Reduce the network by eliminating nodes that are not in the keep_mask using Schur Complement.

        :param keep_mask: Boolean array of length num_nodes indicating which nodes to keep (True) and which to eliminate (False)
        :param regularization: Regularization parameter added to the diagonal of the eliminated block to ensure numerical stability
        """

        keep_idx = np.where(keep_mask)[0]
        rem_idx = np.where(~keep_mask)[0]

        if len(rem_idx) == 0:
            return self

        # Efficient slicing of sparse matrices
        G_BB = self.G[keep_idx, :][:, keep_idx]
        G_BI = self.G[keep_idx, :][:, rem_idx]
        G_IB = self.G[rem_idx, :][:, keep_idx]
        G_II = self.G[rem_idx, :][:, rem_idx]

        # Regularization and factorization
        if regularization > 0:
            G_II += regularization * eye(G_II.shape[0], format='csr')

        solve_II = factorized(G_II.tocsc())

        # Schur: G_new = A - B * inv(D) * C
        # Note: we use the linear operator to avoid explicit inversion
        Y = solve_II(G_IB.toarray())
        # If G_IB is very large, it's better to use solve on sparse columns or iterative methods,
        # but for local blocks .toarray() is ok.

        G_schur = G_BB - G_BI @ Y

        return LinearNetwork(G_schur, global_indices=self.global_indices[keep_idx])

    def solve_local(self, boundary_indices, boundary_values, regularization=1e-12):
        """
        Solve the network given potentials on some nodes (boundary).

        :param boundary_indices: Indices of nodes with fixed potentials (boundary nodes)
        :param boundary_values: Potential values at the boundary nodes
        :param regularization: Regularization parameter added to the diagonal of the free block to ensure numerical stability
        """

        all_indices = np.arange(self.num_nodes)

        # Map input (which could be global or local indices) to local
        # Here we assume boundary_indices are already LOCAL indices (0..N) for speed
        free_mask = np.ones(self.num_nodes, dtype=bool)
        free_mask[boundary_indices] = False
        free_idx = np.where(free_mask)[0]

        if len(free_idx) == 0:
            V = np.zeros(self.num_nodes)
            V[boundary_indices] = boundary_values
            return V

        G_FF = self.G[free_idx, :][:, free_idx]
        G_FB = self.G[free_idx, :][:, boundary_indices]

        if regularization > 0:
            G_FF += regularization * eye(G_FF.shape[0])

        b = -G_FB @ boundary_values
        V_free = spsolve(G_FF, b)

        V = np.zeros(self.num_nodes)
        V[boundary_indices] = boundary_values
        V[free_idx] = V_free
        return V

    @staticmethod
    def merge(networks, total_unique_nodes=None):
        """
        Merge multiple LinearNetwork instances into a single global network. This is done by summing the conductance contributions of all networks at their respective global indices.

        :param networks: List of LinearNetwork instances to merge
        :param total_unique_nodes: Optional total number of unique global nodes across all networks. If not provided, it will be computed from the maximum global index.
        """

        rows, cols, data = [], [], []

        # If we don't know the total number of unique nodes, calculate it
        if total_unique_nodes is None:
            max_idx = 0
            for net in networks:
                max_idx = max(max_idx, np.max(net.global_indices))
            total_unique_nodes = max_idx + 1

        for net in networks:
            # Convert to COO to extract triplets (r, c, v)
            coo = coo_matrix(net.G)#net.G.tocoo()

            # Map local indices to global
            global_r = net.global_indices[coo.row]
            global_c = net.global_indices[coo.col]

            rows.append(global_r)
            cols.append(global_c)
            data.append(coo.data)

        # Concatenate everything in a single pass
        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        all_data = np.concatenate(data)

        # The creation of coo_matrix automatically sums duplicates (interface nodes)
        G_global = coo_matrix((all_data, (all_rows, all_cols)), shape=(total_unique_nodes, total_unique_nodes)).tocsr()

        return LinearNetwork(G_global, global_indices=np.arange(total_unique_nodes))

class GridMesher:
    """
    Class responsible for converting a 2D grid map of the film into a LinearNetwork representation, handling the local connectivity and resistances based on the types of nodes (MM, MD, DD).
    """
    def __init__(self, film_map, resistances):
        self.map = film_map
        self.H, self.W = film_map.shape
        self.resistances = resistances # (mm, md, dd)

    def get_global_id(self, r, c):
        """
        Return a unique global ID for the node at position (r, c) in the grid.
        (Works on numpy arrays)

        :param r: Row index or array of row indices
        :param c: Column index or array of column indices
        """

        return r * self.W + c

    def build_chunk(self, r_start, r_end, c_start, c_end):
        """
        Split the grid into a chunk defined by the given row and column ranges, and build a LinearNetwork for that chunk.

        :param r_start: start row index (inclusive). Equivalent to the y coordinate of the top-left corner of the chunk.
        :param r_end: end row index (exclusive). Equivalent to the y coordinate of the bottom-right corner of the chunk.
        :param c_start: start column index (inclusive). Equivalent to the x coordinate of the top-left corner of the chunk.
        :param c_end: end column index (exclusive). Equivalent to the x coordinate of the bottom-right corner of the chunk.
        """

        # Extract the slice with a margin of 1 pixel if possible for connections,
        sub_map = self.map[r_start:r_end, c_start:c_end]
        h_sub, w_sub = sub_map.shape

        # Create local nodes
        # Create a grid of global IDs corresponding to this patch
        rows = np.arange(r_start, r_end).reshape(-1, 1)
        cols = np.arange(c_start, c_end).reshape(1, -1)
        # Broadcasting to get matrix of global IDs
        global_ids = (rows * self.W + cols).flatten()

        # Build the adjacency matrix (similar to your original code but optimized)
        # Here we use heavy vectorization
        flat_map = sub_map.flatten()

        # Define neighbors (right and down)
        # Indices in the local flattened vector (0..h*w-1)
        idx = np.arange(h_sub * w_sub).reshape(h_sub, w_sub)

        # Horizontal links (Right)
        mask_h = (idx[:, :-1]).flatten()
        mask_h_next = (idx[:, 1:]).flatten()
        # Vertical links (Down)
        mask_v = (idx[:-1, :]).flatten()
        mask_v_next = (idx[1:, :]).flatten()

        # Compute conductances for all links in a vectorized way
        vals_h = self._compute_conductance(flat_map[mask_h], flat_map[mask_h_next])
        vals_v = self._compute_conductance(flat_map[mask_v], flat_map[mask_v_next])

        # Assemble local COO
        src = np.concatenate([mask_h, mask_v, mask_h_next, mask_v_next])
        dst = np.concatenate([mask_h_next, mask_v_next, mask_h, mask_v])
        dat = np.concatenate([vals_h, vals_v, vals_h, vals_v]) # Symmetric

        # Add diagonal
        diag_row = np.arange(h_sub * w_sub)
        # Sum over rows for negative diagonal (Kirchhoff)
        G_temp = coo_matrix((dat, (src, dst)), shape=(h_sub*w_sub, h_sub*w_sub))
        diag_val = -np.array(G_temp.sum(axis=1)).flatten()

        # Final combination
        G_final = G_temp + coo_matrix((diag_val, (diag_row, diag_row)), shape=(h_sub*w_sub, h_sub*w_sub))

        return LinearNetwork(G_final.tocsr(), global_indices=global_ids)

    def _compute_conductance(self, type_a, type_b):
        """
        Compute the conductances on a set of nodes, based on their types (MM, MD, DD).

        :param type_a: An array describing the type of the first node in each pair (M, D)
        :param type_b: An array describing the type of the second node in each pair (M, D)
        """

        mm, md, dd = self.resistances
        cond = np.zeros_like(type_a, dtype=float)

        # Bitwise logic on booleans or integers of the map
        is_mm = type_a & type_b
        is_dd = (~type_a) & (~type_b)
        is_md = type_a != type_b

        # Note: off-diagonal entries are negative conductances, and the diagonal will be the negative sum of the row (Kirchhoff's law)
        cond[is_mm] = -1.0/mm
        cond[is_dd] = -1.0/dd
        cond[is_md] = -1.0/md

        return cond

class Solver:
    """
    Class responsible for managing the multi-scale solution process: splitting the grid, building local networks, reducing them,
    assembling the global coarse network, and performing the solve with back-substitution to get the full solution.
    """
    def __init__(self, film_map, resistances, split_size=50):
        """
        Initialize the solver with the film map, resistances, and split size for the multi-scale approach.

        :param film_map: 2d array representing the film, where each value indicates the type of node (e.g., M or D)
        :param resistances: tuple of resistances (mm, md, dd) corresponding to the types of connections
        :param split_size: size of the chunks to split the grid into for local network construction
        :param dbg: boolean flag for enabling debug mode
        """

        self.mesher = GridMesher(film_map, resistances)
        self.split_size = split_size
        self.shape = film_map.shape

        self.n_chunks_y = (self.shape[0] + split_size - 1) // split_size
        self.n_chunks_x = (self.shape[1] + split_size - 1) // split_size

        self.sub_networks = [None] * (self.n_chunks_y * self.n_chunks_x)    # Original local networks
        self.reduced_networks = [None] * (self.n_chunks_y * self.n_chunks_x)    # Reduced networks
        
        self.coarse_network = None  # Assembled global network

        self.global_potentials_coarse = None
        self.full_solution = None

        self._build_hierarchy()
    
    def _get_chunk_index(self, r, c):
        """
        Internal helper function to get the chunk index from a global row and column index

        Args:
            r (int): global row index for a pixel
            c (int): global column index for a pixel

        Returns:
            tuple: chunk y index, chunk x index, linear index
        """

        cy = r // self.split_size
        cx = c // self.split_size
        idx = cy * self.n_chunks_x + cx

        return cy, cx, idx

    def _build_hierarchy(self):
        """
        Internal method to build the multi-scale hierarchy: split the grid, build local networks, reduce them, and assemble the coarse network.
        Builds the network from scratch, should only be called on initialization or after big changes to the film map.
        """

        H, W = self.shape
        S = self.split_size
        
        # Loop su tutti i chunk
        for r in range(0, H, S):
            for c in range(0, W, S):
                cy, cx, idx = self._get_chunk_index(r, c)
                self._rebuild_single_chunk(r, c, idx)
                
        self._reassemble_coarse()

    def _rebuild_single_chunk(self, r_start, c_start, idx):
        """
        Build and reduce a single chunk defined by its starting row and column, and store the results in the corresponding index of sub_networks and reduced_networks.
        """

        H, W = self.shape
        S = self.split_size
        
        # end positions with margin of 1 pixel for connectivity
        r_end = min(r_start + S + 1, H)
        c_end = min(c_start + S + 1, W)
        
        # Rebuild the local network (Grid -> Graph)
        # Note: mesher reads from self.mesher.map, which must be already updated!
        net = self.mesher.build_chunk(r_start, r_end, c_start, c_end)
        self.sub_networks[idx] = net
        
        # Identify boundary nodes for Schur reduction
        g_ids = net.global_indices
        rows = g_ids // W
        cols = g_ids % W
        
        # Boundary of the chunk WITH RESPECT TO the chunk itself (perimeter)
        # Note: we use min/max of the global indices of THIS chunk
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        
        is_boundary = (rows == r_min) | (rows == r_max) | \
                      (cols == c_min) | (cols == c_max)
        
        # Perform Schur reduction
        self.reduced_networks[idx] = net.reduce(keep_mask=is_boundary)

    def _reassemble_coarse(self):
        """
        Reassemble the global network by merging the reduced pieces
        """
        self.coarse_network = LinearNetwork.merge(self.reduced_networks, 
                                                  total_unique_nodes=self.shape[0]*self.shape[1])
    
    def update_grid(self, changed_pixels):
        """
        Update the solver by recalculating ONLY the chunks affected by the changed pixels.
        
        :param changed_pixels: list or array of tuples (r, c, new_value)
        """

        if not changed_pixels:
            return

        changed_chunks_idxs = set()
        
        for r, c, val in changed_pixels:
            self.mesher.map[r, c] = val
            _, _, idx = self._get_chunk_index(r, c)
            changed_chunks_idxs.add(idx)
            
            # Check if we are on the top edge of the current chunk (and not at the absolute edge 0)
            if r % self.split_size == 0 and r > 0:
                 _, _, idx_up = self._get_chunk_index(r - 1, c)
                 changed_chunks_idxs.add(idx_up)
            
            # Check left edge
            if c % self.split_size == 0 and c > 0:
                _, _, idx_left = self._get_chunk_index(r, c - 1)
                changed_chunks_idxs.add(idx_left)

        # Rebuild only the affected chunks
        S = self.split_size
        for idx in changed_chunks_idxs:
            # Retrieve top-left coordinates of the chunk from the linear index
            cy = idx // self.n_chunks_x
            cx = idx % self.n_chunks_x
            r_start = cy * S
            c_start = cx * S
             
            self._rebuild_single_chunk(r_start, c_start, idx)

        # Rebuild the coarse network after all local updates are done
        self._reassemble_coarse()

    def solve(self, bound_coords, bound_values):
        """
        Solve for the potentials given boundary conditions specified by bound_coords and bound_values.

        bound_coords: list of tuples or array (N, 2) with (y, x)
        bound_values: array (N,)
        """

        # map the boundary coordinates to global node IDs
        bound_coords = np.array(bound_coords)
        bound_ids = self.mesher.get_global_id(bound_coords[:, 0], bound_coords[:, 1])

        # solve the coarse system first
        V_coarse = self.coarse_network.solve_local(bound_ids, bound_values)

        # prepare the full solution map
        full_map = np.zeros(self.shape)

        # solve each sub-network with the boundary values obtained from the coarse solution
        for sub_net in self.sub_networks:

            local_ids_global = sub_net.global_indices

            known_values = V_coarse[local_ids_global]

            # build the mask for boundary nodes in the local sub-network
            rows = local_ids_global // self.shape[1]
            cols = local_ids_global % self.shape[1]
            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()
            is_boundary = (rows == r_min) | (rows == r_max) | \
                          (cols == c_min) | (cols == c_max)

            local_boundary_indices = np.where(is_boundary)[0]
            local_boundary_values = known_values[is_boundary]

            V_local = sub_net.solve_local(local_boundary_indices, local_boundary_values)

            # fill in the full solution
            full_map[rows, cols] = V_local

        self.full_solution = full_map
        return full_map

    @time_decorator
    def _build_hierarchy_timed(self):
        """
        Internal method to build the multi-scale hierarchy: split the grid, build local networks, reduce them, and assemble the coarse network.
        """

        H, W = self.shape
        S = self.split_size

        # logic for splitting the grid into chunks, while making sure to include an overlap of 1 pixel for connectivity.
        for r in range(0, H, S):
            for c in range(0, W, S):
                r_end = min(r + S + 1, H)
                c_end = min(c + S + 1, W)

                r_limit = min(r + S, H)
                c_limit = min(c + S, W)

                r_slice_end = r_limit + 1 if r_limit < H else H
                c_slice_end = c_limit + 1 if c_limit < W else W

                net = self.mesher.build_chunk(r, r_slice_end, c, c_slice_end)
                self.sub_networks.append(net)

        # reduction of each local network to keep only the boundary nodes (interface with the coarse network)
        for net in self.sub_networks:
            # identify boundary nodes in the local network based on their global indices
            g_ids = net.global_indices
            rows = g_ids // W
            cols = g_ids % W

            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()

            is_boundary = (rows == r_min) | (rows == r_max) | \
                          (cols == c_min) | (cols == c_max)

            reduced = net.reduce(keep_mask=is_boundary)
            self.reduced_networks.append(reduced)

        if self.dbg:
            print("Assembling coarse network...")

        # merge the reduced networks into a single coarse network
        self.coarse_network = LinearNetwork.merge(self.reduced_networks, total_unique_nodes=H*W)

    @time_decorator
    def solve_timed(self, bound_coords, bound_values):
        """
        Solve for the potentials given boundary conditions specified by bound_coords and bound_values.

        bound_coords: lista di tuple o array (N, 2) con (y, x)
        bound_values: array (N,)
        """

        # map the boundary coordinates to global node IDs
        bound_coords = np.array(bound_coords)
        bound_ids = self.mesher.get_global_id(bound_coords[:, 0], bound_coords[:, 1])

        # solve the coarse system first
        V_coarse = self.coarse_network.solve_local(bound_ids, bound_values)

        # prepare the full solution map
        full_map = np.zeros(self.shape)

        # solve each sub-network with the boundary values obtained from the coarse solution
        for sub_net in self.sub_networks:

            local_ids_global = sub_net.global_indices

            known_values = V_coarse[local_ids_global]

            # build the mask for boundary nodes in the local sub-network
            rows = local_ids_global // self.shape[1]
            cols = local_ids_global % self.shape[1]
            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()
            is_boundary = (rows == r_min) | (rows == r_max) | \
                          (cols == c_min) | (cols == c_max)

            local_boundary_indices = np.where(is_boundary)[0]
            local_boundary_values = known_values[is_boundary]

            V_local = sub_net.solve_local(local_boundary_indices, local_boundary_values)

            # fill in the full solution
            full_map[rows, cols] = V_local

        self.full_solution = full_map
        return full_map

class ElectricalAnalysis:
    """
    Class responsible for computing power and current density maps from the solved potentials and the network structure,
    """
    def __init__(self, solver):
        """
        Initialize the analysis, by passing the solver.

        :param solver: a Solver instance containing the network structure
        """

        self.solver = solver
        self.H, self.W = solver.shape

    def compute_maps(self, dtype=np.float32):
        """
        Compute the power density and current density maps based on the solved voltages and the network structure.
        This function avoids building the full global conductance matrix, and instead streams through the local networks.

        :param dtype: data type for the output maps. Default is np.float32.
        :return: dictionary with keys 'power_density', 'current_density', 'I_x', 'I_y'
        """

        self.V = self.solver.full_solution

        # Allocate memory for the output maps
        P_map = np.zeros((self.H, self.W), dtype=dtype)
        Ix_map = np.zeros((self.H, self.W), dtype=dtype) # Current X component
        Iy_map = np.zeros((self.H, self.W), dtype=dtype) # Current Y component

        for net in self.solver.sub_networks:
            # extracts the conductance links from the local network. Only the upper triangular part is needed to avoid double counting.
            G_upper = triu(net.G, k=1) # k=1 excludes the diagonal (sums)
            coo = G_upper.tocoo()

            if coo.nnz == 0: continue

            # map to global indices and extract conductance values
            idx_i = net.global_indices[coo.row] # Node A
            idx_j = net.global_indices[coo.col] # Node B
            g_val = -coo.data # Conductance (positive) of the link

            # Retrieve Global Voltages
            V_i = self.V.flat[idx_i]
            V_j = self.V.flat[idx_j]

            # compute current and power for each link
            dV = V_i - V_j
            I_link = g_val * dV       # Current flowing from i to j
            P_link = I_link * dV      # Power dissipated on the link (always positive)

            # convert global indices to 2D coordinates
            yi, xi = np.divmod(idx_i, self.W)
            yj, xj = np.divmod(idx_j, self.W)

            # add the power contributions to the power density map (averaging over the two nodes)
            np.add.at(P_map, (yi, xi), 0.5 * P_link)
            np.add.at(P_map, (yj, xj), 0.5 * P_link)

            # find the type of link (horizontal or vertical)
            diff = np.abs(idx_i - idx_j)

            # --- Horizontal Links (diff == 1) ---
            mask_h = (diff == 1)
            if np.any(mask_h):
                # If current flows from i -> j (Left -> Right), it is positive
                # We are accumulating the "passing" magnitude for the node
                #val_h = np.abs(I_link[mask_h])
                val_h = I_link[mask_h] # We want to keep the sign for visualization of current direction
                # Assign to the left and right node (rough spatial average)
                np.add.at(Ix_map, (yi[mask_h], xi[mask_h]), val_h * 0.5)
                np.add.at(Ix_map, (yj[mask_h], xj[mask_h]), val_h * 0.5)

            # --- Vertical Links (diff == W) ---
            mask_v = (diff == self.W)
            if np.any(mask_v):
                val_v = I_link[mask_v] # We want to keep the sign for visualization of current direction
                np.add.at(Iy_map, (yi[mask_v], xi[mask_v]), val_v * 0.5)
                np.add.at(Iy_map, (yj[mask_v], xj[mask_v]), val_v * 0.5)

        # Compute final current magnitude
        J_map = np.sqrt(Ix_map**2 + Iy_map**2)

        return {
            'power_density': P_map,
            'current_density': J_map,
            'I_x': Ix_map,
            'I_y': Iy_map
        }

    @time_decorator
    def compute_maps_timed(self, dtype=np.float32):
        """
        Compute the power density and current density maps based on the solved voltages and the network structure.
        This function avoids building the full global conductance matrix, and instead streams through the local networks.

        :param dtype: data type for the output maps. Default is np.float32.
        :return: dictionary with keys 'power_density', 'current_density', 'I_x', 'I_y'
        """

        # Allocate memory for the output maps
        P_map = np.zeros((self.H, self.W), dtype=dtype)
        Ix_map = np.zeros((self.H, self.W), dtype=dtype) # Current X component
        Iy_map = np.zeros((self.H, self.W), dtype=dtype) # Current Y component

        for net in self.solver.sub_networks:
            # extracts the conductance links from the local network. Only the upper triangular part is needed to avoid double counting.
            G_upper = triu(net.G, k=1) # k=1 excludes the diagonal (sums)
            coo = G_upper.tocoo()

            if coo.nnz == 0: continue

            # map to global indices and extract conductance values
            idx_i = net.global_indices[coo.row] # Node A
            idx_j = net.global_indices[coo.col] # Node B
            g_val = -coo.data # Conductance (positive) of the link

            # Retrieve Global Voltages
            V_i = self.V.flat[idx_i]
            V_j = self.V.flat[idx_j]

            # compute current and power for each link
            dV = V_i - V_j
            I_link = g_val * dV       # Current flowing from i to j
            P_link = I_link * dV      # Power dissipated on the link (always positive)

            # convert global indices to 2D coordinates
            yi, xi = np.divmod(idx_i, self.W)
            yj, xj = np.divmod(idx_j, self.W)

            # add the power contributions to the power density map (averaging over the two nodes)
            np.add.at(P_map, (yi, xi), 0.5 * P_link)
            np.add.at(P_map, (yj, xj), 0.5 * P_link)

            # find the type of link (horizontal or vertical)
            diff = np.abs(idx_i - idx_j)

            # --- Horizontal Links (diff == 1) ---
            mask_h = (diff == 1)
            if np.any(mask_h):
                # If current flows from i -> j (Left -> Right), it is positive
                # We are accumulating the "passing" magnitude for the node
                val_h = I_link[mask_h] # We want to keep the sign for visualization of current direction
                # Assign to the left and right node (rough spatial average)
                np.add.at(Ix_map, (yi[mask_h], xi[mask_h]), val_h * 0.5)
                np.add.at(Ix_map, (yj[mask_h], xj[mask_h]), val_h * 0.5)

            # --- Vertical Links (diff == W) ---
            mask_v = (diff == self.W)
            if np.any(mask_v):
                #val_v = np.abs(I_link[mask_v])
                val_v = I_link[mask_v] # We want to keep the sign for visualization of current direction
                np.add.at(Iy_map, (yi[mask_v], xi[mask_v]), val_v * 0.5)
                np.add.at(Iy_map, (yj[mask_v], xj[mask_v]), val_v * 0.5)

        # Compute final current magnitude
        J_map = np.sqrt(Ix_map**2 + Iy_map**2)

        return {
            'power_density': P_map,
            'current_density': J_map,
            'I_x': Ix_map,
            'I_y': Iy_map
        }

class Simulation:
    """
        High-level class that encapsulates the entire simulation process: from building the solver to computing the maps and providing utility functions for electrode coordinates.
        Also handles time evolution of the system.
    """

    def __init__(self, film_map, resistances=(1, 100, 10000), split_size=50):
        self.solver = Solver(film_map, resistances, split_size=split_size)
        self.analysis = ElectricalAnalysis(self.solver)
        self.film_map = film_map
        self.maps = None

    def solve_all(self, bound_coords, bound_values):
        """
        Compute voltage, current and power dissipation maps

        Args:
            bound_coords (np.array): numpy array of indices for the bound nodes
            bound_values (np.array): numpy array containing the values for the bias voltages

        Returns:
            dict: a dictionary containing the computed maps. Available keys are voltage, power_density, current_density, I_x, I_y, E_field
        """

        self.solver.solve(bound_coords, bound_values)
        maps = self.analysis.compute_maps()
        maps['voltage'] = self.solver.full_solution

        grad_y, grad_x = np.gradient(maps['voltage'])
        E_mag = np.sqrt(grad_x**2 + grad_y**2)
        maps['E_field'] = E_mag

        self.maps = maps

        return maps

    def find_nearest_swap_target(self, r, c, target_val, radius=5):
        """
        Find the nearest available pixel for the swap

        Args:
            r (int): row index of the center of the search window
            c (int): column index of the center of the search window
            target_val (bool): value to search for within the window
            radius (int, optional): radius of the search window. Defaults to 5.

        Returns:
            tuple or None: coordinates of the nearest available pixel or None if not found
        """

        H, W = self.solver.shape
        
        # find the window borders, making sure to stay within the image boundaries
        r_min = max(0, r - radius)
        r_max = min(H, r + radius + 1)
        c_min = max(0, c - radius)
        c_max = min(W, c + radius + 1)
        
        # extract a local window around (r, c)
        window = self.film_map[r_min:r_max, c_min:c_max]
        
        # find the indices RELATIVE to the window where window == target_val
        # np.argwhere returns an array (N, 2) of coordinates [row, col]
        candidates_rel = np.argwhere(window == target_val)
        
        if len(candidates_rel) == 0:
            return None
        
        # calculate the relative center of the window (corresponds to global r, c)
        center_rel = np.array([r - r_min, c - c_min])
        
        # calculate squared distances from (r,c)
        dists_sq = np.sum((candidates_rel - center_rel)**2, axis=1)
        
        # find the minimum distance
        min_dist = np.min(dists_sq)
        
        # filter only the candidates that are at that minimum distance
        best_candidates_idx = np.where(dists_sq == min_dist)[0]
        
        # choose one at random among the best
        chosen_idx = np.random.choice(best_candidates_idx)
        chosen_rel = candidates_rel[chosen_idx]
        
        # Convert to global coordinates
        global_pos = (r_min + chosen_rel[0], c_min + chosen_rel[1])
        
        # sanity check: if the found position is the same as the input (r, c), we can return None to indicate no valid swap target
        if global_pos == (r, c):
            return None
            
        return global_pos

    def get_electrodes_coords(self):
        """
        Returns the coordinates of the left and right electrodes in the grid.
        Return format: array (H, 2) of integers.
        """

        H, W = self.solver.shape
        rows = np.arange(H)

        # left electrode (Column 0): pairs [0,0], [1,0], ...
        left_coords = np.column_stack((rows, np.zeros(H, dtype=int)))

        # right electrode (Column W-1): pairs [0,W-1], [1,W-1], ...
        right_coords = np.column_stack((rows, np.full(H, W - 1, dtype=int)))

        # Combine both for convenience
        all_coords = np.vstack((left_coords, right_coords))

        return left_coords, right_coords, all_coords

    def _evolve_step(self, E_threshold, P_threshold, swap_radius=5, bias_v=1.0, E_prob=0.5, P_prob=0.5):
        """
        Perform a single evolution step on the system

        Args:
            swap_radius (int, optional): Size of the window used when searching for a swap candidate. Defaults to 5.
            bias_v (float, optional): Bias voltage applied during the evolution step. Defaults to 1.0.
            E_threshold (float, optional): Threshold for selecting high E field pixels.
            P_threshold (float, optional): Threshold for selecting high power density pixels.
            E_prob (float, optional): Probability of evolving a high E field pixel. Defaults to 0.5.
            P_prob (float, optional): Probability of evolving a high power density pixel. Defaults to 0.5.
        """

        electrodes, grounds, all_bound = self.get_electrodes_coords()

        change_pixels = []  # list to keep track of changed pixels for updating the solver later

        # --- EVOLUTION LOGIC ---
        # first, we compute the maps on the film
        # we then build binary maps, selecting pixels above the threshold for E and P
        # the maps are then used to select pixels to evolve (swap) with a certain probability
        # a void pixel with high enough E can become matter
        # a matter pixel with high enough P can become void
        # when a pixel is changed, another pixel of the opposite type is randomly selected within a certain radius to swap with, to keep the total amount of matter constant

        maps = self.solve_all(all_bound, np.concatenate((np.full(electrodes.shape[0], bias_v), np.zeros(grounds.shape[0]))))

        E_map = maps['E_field']
        P_map = maps['power_density']

        # create binary masks for high E and high P pixels
        high_E_mask = (E_map > E_threshold) & (self.film_map == 0) # only consider void pixels for E evolution
        high_P_mask = (P_map > P_threshold) & (self.film_map == 1) # only consider matter pixels for P evolution

        # create two maps with random values (0-1), compare them with the probabilities to decide which pixels will evolve
        # these maps are then multiplied element-wise (binary and) with the high_E_mask and high_P_mask to get the final selection of pixels to evolve
        evolve_E = (np.random.rand(*E_map.shape) < E_prob) & high_E_mask
        evolve_P = (np.random.rand(*P_map.shape) < P_prob) & high_P_mask

        # get the coordinates of the pixels to evolve
        evolve_E_coords = np.argwhere(evolve_E)
        evolve_P_coords = np.argwhere(evolve_P)

        # shuffle the coordinates to avoid any bias in the order of processing (should fix a bug causing anysotropy in time evolution)
        np.random.shuffle(evolve_E_coords)
        np.random.shuffle(evolve_P_coords)

        # perform the swaps for E evolution (void -> matter)
        for r, c in evolve_E_coords:
            swap_target = self.find_nearest_swap_target(r, c, target_val=True, radius=swap_radius)
            if swap_target is not None:
                r_swap, c_swap = swap_target
                # Swap the values in the film map
                self.film_map[r, c], self.film_map[r_swap, c_swap] = self.film_map[r_swap, c_swap], self.film_map[r, c]
                change_pixels.append((r, c, 1)) # void -> matter
                change_pixels.append((r_swap, c_swap, 0)) # matter -> void
        
        # perform the swaps for P evolution (matter -> void)
        for r, c in evolve_P_coords:
            swap_target = self.find_nearest_swap_target(r, c, target_val=False, radius=swap_radius)
            if swap_target is not None:
                r_swap, c_swap = swap_target
                # Swap the values in the film map
                self.film_map[r, c], self.film_map[r_swap, c_swap] = self.film_map[r_swap, c_swap], self.film_map[r, c]
                change_pixels.append((r, c, 0)) # matter -> void
                change_pixels.append((r_swap, c_swap, 1)) # void -> matter
        
        # update the solver with the changed pixels to avoid rebuilding the entire hierarchy
        if change_pixels:
             self.solver.update_grid(change_pixels)
        
        # still rebuild the analysis object. Should be fast enough for now
        self.analysis = ElectricalAnalysis(self.solver)

    def evolve(self, steps, E_threshold, P_threshold, swap_radius=5, bias_v=1.0, E_prob=0.5, P_prob=0.5):
        maps = []
        for step in range(steps):
            print(f"Evolution step {step+1}/{steps}...")
            self._evolve_step(E_threshold, P_threshold, swap_radius, bias_v, E_prob, P_prob)
            maps.append(self.maps)
            maps[-1]["film_map"] = self.film_map.copy() # also save the film map at each step for visualization

        return maps

    def compute_total_current(self):
        """
        Compute the total current trough the system

        Returns:
            float: the total current
        """

        if self.maps is None:
            raise ValueError("Maps not computed yet. Please run solve_all() first.")

        Ix = self.maps['I_x']
        Iy = self.maps['I_y']

        # Total current can be computed by summing the current components at the electrodes
        # Assuming left electrode is the source and right electrode is the sink, we can sum the currents at the left edge (Column 0)
        total_current = np.sum(Ix[:, 0]) # Sum of currents entering from the left edge

        return total_current
        
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

    film = perlin_noise < threshold

    return film.astype(bool)

def plot_on_film_map(film_map, data, ax=None, percentile=None, colorbar=True, log_scale=False, cmap_name='inferno'):
    """
    Plot an overlay of data (e.g., current density) on top of the film map, using a glow effect.

    :param film_map: 2D array representing the film (boolean or binary)
    :param data: 2D array of data to overlay on the film map (e.g., current density)
    :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
    :param percentile: Percentile to determine the minimum value for normalization (used to hide low "noise" values)
    :param colorbar: Whether to show a colorbar or not
    :param log_scale: Whether to use logarithmic scale for normalization
    :return: Matplotlib Axes object with the plot
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # plot the background film map in grayscale
    ax.imshow(film_map, cmap='gray', interpolation='nearest', alpha=1.0)

    J = data.copy()
    J = np.maximum(J, 1e-20)    # avoids log(0)

    if percentile is not None:
        vmin = np.percentile(J, percentile)

    else:
        vmin = J.min()

    vmax = J.max()

    # If the map is flat (all zero or constant), avoid division by zero
    if vmax <= vmin:
        vmax = vmin + 1e-12

    if log_scale:
        # --- MANUAL LOGARITHMIC NORMALIZATION (0.0 -> 1.0) ---
        # Formula: (log(x) - log(min)) / (log(max) - log(min))
        log_J = np.log10(J)
        log_min = np.log10(vmin)
        log_max = np.log10(vmax)

        norm_data = (log_J - log_min) / (log_max - log_min)
    else:
        # --- LINEAR NORMALIZATION (0.0 -> 1.0) ---
        norm_data = (J - vmin) / (vmax - vmin)

    # clip values to [0, 1]
    norm_data = np.clip(norm_data, 0.0, 1.0)

    cmap = plt.get_cmap(cmap_name)

    # cmap(value) returns (R, G, B, A) with values 0-1
    rgba_img = cmap(norm_data)

    # Modify Alpha Channel (Transparency)
    # Use normalized data to decide opacity.
    # alpha = 0 -> completely transparent (film visible underneath)
    # alpha = 1 -> full color
    rgba_img[:, :, 3] = norm_data

    # make values below noise threshold completely transparent
    rgba_img[norm_data <= 0.0, 3] = 0.0

    ax.imshow(rgba_img)
    ax.axis('off')

    if colorbar:
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)), ax=ax)

    return ax

def plot_streamlines_on_film(film_map, Ix, Iy, ax=None, step=20):
    """
    Plot streamlines for a vector field (Ix, Iy) (usually current flow) on top of the film map.

    :param film_map: 2D array representing the film (boolean or binary)
    :param Ix: 2D array representing the x-component of the vector field
    :param Iy: 2D array representing the y-component of the vector field
    :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
    :param step: Step size for downsampling the vector field for plotting
    """

    if ax is None: fig, ax = plt.subplots(figsize=(10, 10))

    H, W = film_map.shape

    # plot the film map as background
    ax.imshow(film_map, cmap='binary', alpha=0.3)

    # build a downsampled grid for streamlines
    y, x = np.mgrid[0:H:step, 0:W:step]

    # Extract downsampled vectors
    u = Ix[::step, ::step]
    v = Iy[::step, ::step]

    # Calculate magnitude for coloring the lines
    speed = np.sqrt(u**2 + v**2)

    # 'density' controls how dense the lines are
    strm = ax.streamplot(x, y, u, v, color=speed, cmap='autumn',
                         linewidth=1, density=1.5, arrowsize=1.0)

    ax.set_title("Current Flow Direction")
    ax.invert_yaxis() # Important because images have (0,0) at the top
    return ax

