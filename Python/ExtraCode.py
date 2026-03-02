import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
import sys


class SGWTObject:
    """Container for SGWT (Spectral Graph Wavelet Transform) data and parameters"""
    def __init__(self):
        self.Data = None
        self.Parameters = {}
        self.Graph = {}


def cal_laplacian(adjacency_matrix, laplacian_type="normalized"):
    """
    Compute graph Laplacian matrix.
    
    Parameters:
    -----------
    adjacency_matrix : scipy.sparse matrix
        Sparse adjacency matrix
    laplacian_type : str
        Type of Laplacian: "unnormalized", "normalized", or "randomwalk"
    
    Returns:
    --------
    L : scipy.sparse matrix
        Laplacian matrix
    """
    A = adjacency_matrix.tocsr()
    
    if laplacian_type == "unnormalized":
        # L = D - A
        degree = np.array(A.sum(axis=1)).flatten()
        D = diags(degree, format='csr')
        L = D - A
    
    elif laplacian_type == "normalized":
        # L = I - D^(-1/2) * A * D^(-1/2)
        degree = np.array(A.sum(axis=1)).flatten()
        degree_sqrt_inv = np.power(degree, -0.5)
        degree_sqrt_inv[np.isinf(degree_sqrt_inv)] = 0
        D_sqrt_inv = diags(degree_sqrt_inv, format='csr')
        L = csr_matrix(np.eye(A.shape[0])) - D_sqrt_inv @ A @ D_sqrt_inv
    
    elif laplacian_type == "randomwalk":
        # L = I - D^(-1) * A
        degree = np.array(A.sum(axis=1)).flatten()
        degree_inv = np.power(degree, -1.0)
        degree_inv[np.isinf(degree_inv)] = 0
        D_inv = diags(degree_inv, format='csr')
        L = csr_matrix(np.eye(A.shape[0])) - D_inv @ A
    
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")
    
    return L


def FastDecompositionLap(laplacian_matrix, k_eigen, which="SM"):
    """
    Perform sparse eigendecomposition of Laplacian matrix.
    
    Parameters:
    -----------
    laplacian_matrix : scipy.sparse matrix
        Sparse Laplacian matrix
    k_eigen : int
        Number of eigenvalues/eigenvectors to compute
    which : str
        "SM" for smallest magnitude (default), "LM" for largest magnitude
    
    Returns:
    --------
    dict with "evalues" (eigenvalues) and "evectors" (eigenvectors)
    """
    k_eigen = min(k_eigen, laplacian_matrix.shape[0] - 2)
    eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=k_eigen, which=which)
    
    return {
        "evalues": eigenvalues,
        "evectors": eigenvectors
    }


def sgwt_auto_scales(lmax, J, scaling_factor):
    """
    Automatically generate SGWT scales.
    
    Parameters:
    -----------
    lmax : float
        Maximum eigenvalue
    J : int
        Number of scales
    scaling_factor : float
        Scaling factor
    
    Returns:
    --------
    scales : ndarray
        Array of scales
    """
    return np.linspace(0, lmax / scaling_factor, J)


def runSpecGraph(SG, k=25, laplacian_type="normalized", length_eigenvalue=None, verbose=True):
    """
    Build spectral graph for SGWT object.
    
    Generate Graph slot information including adjacency matrix, Laplacian matrix, 
    eigenvalues, and eigenvectors.
    
    Parameters:
    -----------
    SG : SGWTObject
        SGWT object from initSGWT()
    k : int
        Number of nearest neighbors for graph construction (default: 25)
    laplacian_type : str
        Type of graph Laplacian: "unnormalized", "normalized", or "randomwalk" (default: "normalized")
    length_eigenvalue : int, optional
        Number of eigenvalues/eigenvectors to compute (default: None, uses full length)
    verbose : bool
        Whether to print progress messages (default: True)
    
    Returns:
    --------
    SG : SGWTObject
        Updated SGWT object with Graph slot populated
    """
    
    # Validate input
    if not isinstance(SG, SGWTObject):
        raise TypeError("Input must be an SGWT object from initSGWT()")
    
    if SG.Data is None:
        raise ValueError("SGWT object must have Data slot initialized")
    
    # Extract data
    data_in = SG.Data['data']
    x_col = SG.Data['x_col']
    y_col = SG.Data['y_col']
    
    # Set default length_eigenvalue to full length if not specified
    if length_eigenvalue is None:
        length_eigenvalue = len(data_in)
    
    if verbose:
        print("Building graph from spatial coordinates...")
    
    # Extract spatial coordinates
    coords = data_in[[x_col, y_col]].values
    
    # Build k-nearest neighbor graph
    distances = cdist(coords, coords, metric='euclidean')
    
    # For each point, find k nearest neighbors (excluding itself)
    nn_indices = np.argsort(distances, axis=1)[:, 1:k+1]
    
    # Create edge list
    edges = []
    for i in range(len(data_in)):
        for j in nn_indices[i]:
            edge = tuple(sorted([i, j]))
            edges.append(edge)
    
    # Remove duplicate edges
    edges = list(set(edges))
    edges = np.array(edges)
    
    # Build adjacency matrix
    n = len(data_in)
    row_indices = np.concatenate([edges[:, 0], edges[:, 1]])
    col_indices = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row_indices))
    
    A = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    
    if verbose:
        print("Computing Laplacian and eigendecomposition...")
    
    # Compute Laplacian matrix
    L = cal_laplacian(A, laplacian_type)
    
    # Eigendecomposition
    decomp = FastDecompositionLap(L, k_eigen=length_eigenvalue, which="SM")
    
    # Update SGWT object
    SG.Graph = {
        'adjacency_matrix': A,
        'laplacian_matrix': L,
        'eigenvalues': decomp['evalues'],
        'eigenvectors': decomp['evectors']
    }
    
    # Auto-generate scales if not provided (now that we have eigenvalues)
    if 'scales' not in SG.Parameters or SG.Parameters['scales'] is None:
        lmax = np.max(decomp['evalues']) * 0.95
        SG.Parameters['scales'] = sgwt_auto_scales(lmax, SG.Parameters['J'], SG.Parameters['scaling_factor'])
        if verbose:
            scales_str = ', '.join([f"{s:.4f}" for s in SG.Parameters['scales']])
            print(f"Auto-generated scales: {scales_str}")
    
    if verbose:
        print("Graph construction completed.")
    
    return SG