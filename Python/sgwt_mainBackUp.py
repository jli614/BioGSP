import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
import sys



#There's like two imported agent AI code here


class SGWTObject:
    """Container for SGWT (Spectral Graph Wavelet Transform) data and parameters"""
    def __init__(self):
        self.Data = None
        self.Parameters = {}
        self.Graph = {}






#BREAK LINE

def run_spec_graph(
    SG: SGWT,
    k: int = 25,
    laplacian_type: str = "normalized",
    length_eigenvalue: Optional[int] = None,
    verbose: bool = True,
) -> SGWT:

    if not isinstance(SG, SGWT):
        raise TypeError("Input must be an SGWT object from initSGWT()")

    if SG.Data is None:
        raise ValueError("SGWT object must have Data slot initialized")

    data = SG.Data["data"]
    x_col = SG.Data["x_col"]
    y_col = SG.Data["y_col"]

    coords = data[[x_col, y_col]].to_numpy()
    n = coords.shape[0]

    if length_eigenvalue is None:
        length_eigenvalue = n

    if verbose:
        print("Building graph from spatial coordinates...")

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(coords)
    _, idx = nn.kneighbors(coords)

    edges = set()
    for i in range(n):
        for j in idx[i]:
            if i != j:
                edges.add(tuple(sorted((i, j))))

    rows, cols = zip(*edges)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = (A + A.T).tocsr()

    if verbose:
        print("Computing Laplacian and eigendecomposition...")

    L = cal_laplacian(A, laplacian_type)

    evals, evecs = eigsh(L, k=length_eigenvalue, which="SM")

    SG.Graph = {
        "adjacency_matrix": A,
        "laplacian_matrix": L,
        "eigenvalues": evals,
        "eigenvectors": evecs,
    }

    if SG.Parameters.get("scales") is None:
        lmax = evals.max() * 0.95
        SG.Parameters["scales"] = sgwt_auto_scales(
            lmax,
            SG.Parameters["J"],
            SG.Parameters["scaling_factor"],
        )
        if verbose:
            scales_str = ", ".join(f"{s:.4f}" for s in SG.Parameters["scales"])
            print(f"Auto-generated scales: {scales_str}")

    if verbose:
        print("Graph construction completed.")

    return SG


#BREAK LINE 2

def run_sgwt(SG: SGWT, use_batch: bool = True, verbose: bool = True) -> SGWT:

    if not isinstance(SG, SGWT):
        raise TypeError("Input must be an SGWT object")

    if SG.Graph is None:
        raise ValueError("Graph slot is empty. Run run_spec_graph() first.")

    eigenvalues = SG.Graph["eigenvalues"]
    eigenvectors = SG.Graph["eigenvectors"]
    params = SG.Parameters
    signals = SG.Data["signals"]
    data = SG.Data["data"]

    if params.get("scales") is None:
        raise ValueError("Scales not found. Run run_spec_graph() first.")

    if verbose:
        print(f"Performing SGWT analysis for {len(signals)} signals...")

    forward = {}
    inverse = {}

    for sig in signals:
        if verbose:
            print(f"Processing signal: {sig}")

        sig_vec = data[sig].to_numpy()

        fwd = sgwt_forward(
            sig_vec,
            eigenvectors,
            eigenvalues,
            params["scales"],
            kernel_type=params["kernel_type"],
        )
        inv = sgwt_inverse(fwd, eigenvectors, sig_vec)

        forward[sig] = fwd
        inverse[sig] = inv

    SG.Forward = forward
    SG.Inverse = inverse

    if verbose:
        print("SGWT analysis completed.")

    return SG

#Breakline 3
def _sgwt_repr(self):
    lines = ["SGWT Object", "==========="]

    if self.Data:
        d = self.Data["data"]
        lines.append("Data:")
        lines.append(f"  Dimensions: {d.shape[0]} x {d.shape[1]}")
        lines.append(f"  Coordinates: {self.Data['x_col']}, {self.Data['y_col']}")
        lines.append(f"  Signals: {', '.join(self.Data['signals'])}")

    if self.Parameters:
        lines.append("\nParameters:")
        for k, v in self.Parameters.items():
            lines.append(f"  {k}: {v}")

    lines.append("\nStatus:")
    lines.append(f"  Graph computed: {self.Graph is not None}")
    lines.append(f"  Forward computed: {self.Forward is not None}")
    lines.append(f"  Inverse computed: {self.Inverse is not None}")

    return "\n".join(lines)


SGWT.__repr__ = _sgwt_repr
