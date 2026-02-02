import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, issparse
from scipy.sparse.linalg import spsolve, factorized
from noise import pnoise2

def build_cluster_random(size, threshold=0.5, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    clusters = np.random.rand(size, size) > threshold
    #np.random.randint(0, 1, (size, size)) > threshold

    return clusters.astype(bool)

def build_clusters_perlin(size, scale=10, threshold=0.5, seed=None):
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    if seed is None:
        seed = np.random.randint(0, 100)
    
    perlin_noise = np.vectorize(lambda i, j: pnoise2(i / scale, j / scale, octaves=6, base=seed))(x, y)

    clusters = perlin_noise > threshold

    return clusters.astype(bool)

def build_adjacency_matrix(mask):

    # vectorized approach to building an adj mat
    # first, build the adj mat for the entire grid, then extract the submatrix for active nodes only

    H, W = mask.shape
    N = H * W

    node_ids = np.arange(N).reshape(H, W)

    h_links = mask[:, :-1] & mask[:, 1:]
    v_links = mask[:-1, :] & mask[1:, :]

    adjacency_matrix = lil_matrix((N, N), dtype=bool)

    h_link_inds = node_ids[:, :-1][h_links]
    v_link_inds = node_ids[:-1, :][v_links]

    adjacency_matrix[h_link_inds, h_link_inds + 1] = 1
    adjacency_matrix[h_link_inds + 1, h_link_inds] = 1
    adjacency_matrix[v_link_inds, v_link_inds + W] = 1
    adjacency_matrix[v_link_inds + W, v_link_inds] = 1

    active_nodes = np.where(mask.flatten())[0]
    adjacency_matrix = adjacency_matrix[active_nodes, :][:, active_nodes]

    
    return csr_matrix(adjacency_matrix)

def build_adjacency_matrices(mask):
    # build multiple adj matrices, representing different connections (metal-metal, metal-dielectric, dielectric-dielectric)

    H, W = mask.shape
    N = H * W

    node_ids = np.arange(N).reshape(H, W)

    h_links_mm = mask[:, :-1] & mask[:, 1:]
    v_links_mm = mask[:-1, :] & mask[1:, :]

    h_links_md = (mask[:, :-1] != mask[:, 1:])
    v_links_md = (mask[:-1, :] != mask[1:, :])

    h_links_dd = (~mask[:, :-1]) & (~mask[:, 1:])
    v_links_dd = (~mask[:-1, :]) & (~mask[1:, :])

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
    
    return adj_mats

def adj_to_conductance_matrix(adj_matrix):
    adj_matrix = adj_matrix.copy().astype(float)
    resistance_value = 1.0

    conductance_matrix = adj_matrix * (-1 / resistance_value)
    diagonal_values = np.array(-conductance_matrix.sum(axis=1)).flatten()
    conductance_matrix.setdiag(diagonal_values)

    return conductance_matrix

def adjs_to_conductance_matrix(adj_matrices, resistance_values):
    mm_resistance, md_resistance, dd_resistance = resistance_values

    conductance_matrix = lil_matrix(adj_matrices[0].shape, dtype=float)

    conductance_matrix += adj_matrices[0] * (-1 / mm_resistance)
    conductance_matrix += adj_matrices[1] * (-1 / md_resistance)
    conductance_matrix += adj_matrices[2] * (-1 / dd_resistance)

    diagonal_values = np.array(-conductance_matrix.sum(axis=1)).flatten()
    conductance_matrix.setdiag(diagonal_values)

    return conductance_matrix

def split_clusters(clusters, sub_grid_size):
    sub_clusters = []
    edge_indices = []

    for i in range(0, clusters.shape[0], sub_grid_size):
        row = []
        for j in range(0, clusters.shape[1], sub_grid_size):
            # Determine the end indices, adding +1 for overlap unless at the edge
            i_end = min(i + sub_grid_size + (0 if i + sub_grid_size >= clusters.shape[0] else 1), clusters.shape[0])
            j_end = min(j + sub_grid_size + (0 if j + sub_grid_size >= clusters.shape[1] else 1), clusters.shape[1])
            row.append(clusters[i:i_end, j:j_end])
            edge_indices.append(((i, i_end), (j, j_end)))
        sub_clusters.append(row)

    sub_clusters_flat = [sub_cluster for row in sub_clusters for sub_cluster in row]

    return sub_clusters_flat, edge_indices

def reduce_conductance_matrices(conductance_matrices, sub_clusters_flat):
    # split the conductance matrices into 4 sub-matrices, representing borders and internal connections
    # (I = internal, B = border). The 4 matrices are II, IB, BI, BB

    reduced_conductance_matrices = []

    for G, C in zip(conductance_matrices, sub_clusters_flat):
        nrows, ncols = C.shape
        
        border_mask_2d = np.zeros((nrows, ncols), dtype=bool)
        border_mask_2d[0, :] = True
        border_mask_2d[-1, :] = True
        border_mask_2d[:, 0] = True
        border_mask_2d[:, -1] = True

        border_mask = border_mask_2d.flatten()
        internal_mask = ~border_mask

        border_idx = np.where(border_mask)[0]
        internal_idx = np.where(internal_mask)[0]
        
        G_BB = G[border_idx, :][:, border_idx]
        G_BI = G[border_idx, :][:, internal_idx]
        G_IB = G[internal_idx, :][:, border_idx]
        G_II = G[internal_idx, :][:, internal_idx]

        solve_G_II = factorized(G_II.tocsc()) 
        Y = solve_G_II(G_IB.toarray()) 
        correction_term = G_BI @ Y 
        G_schur = G_BB - correction_term

        reduced_conductance_matrices.append(G_schur)
    
    return reduced_conductance_matrices

def reduce_conductance_matrices_from_mask(conductance_matrices, border_masks):
    # split the conductance matrices into 4 sub-matrices, representing borders and internal connections
    # (I = internal, B = border). The 4 matrices are II, IB, BI, BB

    reduced_conductance_matrices = []

    for G, border_mask in zip(conductance_matrices, border_masks):

        internal_mask = ~border_mask

        border_idx = np.where(border_mask)[0]
        internal_idx = np.where(internal_mask)[0]
        
        G_BB = G[border_idx, :][:, border_idx]
        G_BI = G[border_idx, :][:, internal_idx]
        G_IB = G[internal_idx, :][:, border_idx]
        G_II = G[internal_idx, :][:, internal_idx]

        # add a small regularization to the diagonal of G_II to ensure it's invertible
        reg_value = 1e-12
        G_II = G_II + reg_value * csr_matrix(np.eye(G_II.shape[0]))

        solve_G_II = factorized(G_II.tocsc()) 
        Y = solve_G_II(G_IB.toarray()) 
        correction_term = G_BI @ Y 
        G_schur = G_BB - correction_term

        reduced_conductance_matrices.append(G_schur)
    
    return reduced_conductance_matrices

def get_subgrid_coords(edge_indices):
    # build a set containing the global indices of the border nodes

    sub_grid_coords = [] 
    for k, ((y_start, y_end), (x_start, x_end)) in enumerate(edge_indices):
        
        h_local = y_end - y_start
        w_local = x_end - x_start
        
        border_nodes_set = set()
        
        for x_glob in range(x_start, x_end):
            border_nodes_set.add((y_start, x_glob))
            
        for x_glob in range(x_start, x_end):
            border_nodes_set.add((y_end - 1, x_glob))
            
        for y_glob in range(y_start, y_end):
            border_nodes_set.add((y_glob, x_start))
            
        for y_glob in range(y_start, y_end):
            border_nodes_set.add((y_glob, x_end - 1))
            
        block_border_coords = sorted(list(border_nodes_set))
        
        sub_grid_coords.append(block_border_coords)

    all_interface_coords = set()
    for coords in sub_grid_coords:
        all_interface_coords.update(coords)
    
    return sub_grid_coords, all_interface_coords

def restitch_conductances(all_interface_coords, sub_grid_coords, reduced_conductance_matrices):
    # stitch togheter the reduced conductance matrices into a big one, representing conductance between border nodes only
    coord_to_id = {coord: i for i, coord in enumerate(all_interface_coords)}
    N_global = len(all_interface_coords)

    rows_list = []
    cols_list = []
    data_list = []

    for k, S_k in enumerate(reduced_conductance_matrices):
        local_coords = sub_grid_coords[k]
        global_indices = [coord_to_id[c] for c in local_coords]

        if issparse(S_k):
            S_k_coo = S_k.tocoo()
            r_local = S_k_coo.row
            c_local = S_k_coo.col
            data_local = S_k_coo.data
        else:
            local_grids = np.indices(S_k.shape) 
            r_local = local_grids[0].reshape(-1)
            c_local = local_grids[1].reshape(-1)
            
            data_local = np.asarray(S_k).reshape(-1)

        ids_array = np.array(global_indices)
        r_global = ids_array[r_local]
        c_global = ids_array[c_local]

        rows_list.append(np.asarray(r_global).reshape(-1))
        cols_list.append(np.asarray(c_global).reshape(-1))
        data_list.append(np.asarray(data_local).reshape(-1))


    M_global = coo_matrix(
        (np.concatenate(data_list), 
        (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(N_global, N_global)
    ).tocsr()

    return M_global
