from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from typing import Optional
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from sgwt_core import sgwt_auto_scales, sgwt_inverse, cosine_similarity, gft, igft, compute_sgwt_filters



def sgwt_auto_scales(lmax, J=5, scaling_factor=2):
    """Generate logarithmically spaced scales."""
    scales = lmax / (scaling_factor ** np.arange(J))
    return scales


def sgwt_inverse(sgwt_decomp, eigenvectors, original_signal=None):
    """
    Perform inverse SGWT decomposition to reconstruct signal(s).
    
    Parameters
    ----------
    sgwt_decomp : dict
        SGWT decomposition result with 'fourier_coefficients' -> 'filtered'
    eigenvectors : ndarray
        Eigenvectors for inverse transform
    original_signal : array-like, optional
        Original signal for error calculation
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'vertex_approximations': dict of reconstructed components
        - 'reconstructed_signal': summed reconstruction
        - 'reconstruction_error': RMSE if original_signal provided
    """
    filtered_fourier = sgwt_decomp['fourier_coefficients']['filtered']
    
    # Perform inverse GFT on all filtered coefficients using igft function
    vertex_approximations = {}
    for key in filtered_fourier.keys():
        # Use igft function for inverse transform - handles both single signals and batches
        vertex_approximations[key] = igft(filtered_fourier[key], eigenvectors)
    
    # Rename components for clarity
    if 'scaling' in vertex_approximations:
        vertex_approximations['low_pass'] = vertex_approximations.pop('scaling')
    
    # Rename wavelet scales
    wavelet_keys = [k for k in vertex_approximations.keys() if re.match(r'^wavelet_scale_', k)]
    for name in wavelet_keys:
        scale_num = re.sub(r'^wavelet_scale_', '', name)
        vertex_approximations[f'wavelet_{scale_num}'] = vertex_approximations.pop(name)
    
    # Reconstruct signal(s) - sum of all components
    reconstructed = sum(vertex_approximations.values())
    
    # Calculate reconstruction error(s) if original signal provided
    reconstruction_error = None
    if original_signal is not None:
        original_signal = np.asarray(original_signal)
        reconstructed_array = np.asarray(reconstructed)
        
        # Validate dimensions
        if original_signal.ndim == 2 and reconstructed_array.ndim == 2:
            if original_signal.shape != reconstructed_array.shape:
                raise ValueError("Dimensions of original_signal must match reconstructed signal")
            # Calculate RMSE for each signal (column)
            reconstruction_error = np.sqrt(np.mean((original_signal - reconstructed_array) ** 2, axis=0))
        elif original_signal.ndim == 1 and (reconstructed_array.ndim == 1 or reconstructed_array.shape[1] == 1):
            reconstructed_vec = reconstructed_array.flatten()
            if original_signal.shape[0] != reconstructed_vec.shape[0]:
                raise ValueError("Length of original_signal must match reconstructed signal")
            # Calculate RMSE for single signal
            reconstruction_error = np.sqrt(np.mean((original_signal - reconstructed_vec) ** 2))
        else:
            raise ValueError("Type mismatch between original_signal and reconstructed signal")
    
    return {
        'vertex_approximations': vertex_approximations,
        'reconstructed_signal': reconstructed,
        'reconstruction_error': reconstruction_error
    }


def cosine_similarity(x, y, eps=1e-12):
    """
    Calculate cosine similarity between two vectors.
    
    Parameters
    ----------
    x : array-like
        First vector
    y : array-like
        Second vector
    eps : float, default=1e-12
        Small epsilon value for numerical stability
        
    Returns
    -------
    float
        Cosine similarity value clamped to [-1, 1]
    """
    # Convert to numeric vectors
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Check for same length
    if len(x) != len(y):
        raise ValueError("Vectors must have the same length")
    
    # Calculate magnitudes
    magnitude_x = np.sqrt(np.sum(x ** 2))
    magnitude_y = np.sqrt(np.sum(y ** 2))
    
    # Handle zero magnitude cases with numerical stability
    if magnitude_x < eps and magnitude_y < eps:
        return 0.0
    if magnitude_x < eps or magnitude_y < eps:
        return 0.0
    
    # Calculate dot product and cosine similarity
    dot_product = np.sum(x * y)
    cosine_sim = dot_product / max(magnitude_x * magnitude_y, eps)
    
    # Clamp to [-1, 1] range for numerical stability
    cosine_sim = max(-1.0, min(1.0, cosine_sim))
    
    return cosine_sim




#Line Break


@dataclass
class SGWT:
    data: pd.DataFrame
    x_col: str = "x"
    y_col: str = "y"
    signals: List[str] = field(default_factory=list)
    Graph: Optional[Any] = None
    Forward: Optional[Any] = None
    Inverse: Optional[Any] = None
    Parameters: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        rows, cols = self.data.shape
        return (f"SGWT(data: {rows}x{cols}, coords: {self.x_col},{self.y_col}, "
                f"signals: {self.signals}, Graph: {self.Graph is not None})")

    def __str__(self):
        """Detailed string representation of SGWT object (like R's print method)."""
        output = []
        output.append("SGWT Object")
        output.append("===========")

        # Data information
        if self.data is not None:
            output.append("Data:")
            rows, cols = self.data.shape
            output.append(f"  Dimensions: {rows} x {cols}")
            output.append(f"  Coordinates: {self.x_col}, {self.y_col}")
            output.append(f"  Signals: {', '.join(self.signals)}")

        # Parameters
        if self.Parameters:
            output.append("\nParameters:")
            output.append(f"  k (neighbors): {self.Parameters.get('k', 'N/A')}")
            output.append(f"  J (scales): {self.Parameters.get('J', 'N/A')}")
            output.append(f"  Kernel type: {self.Parameters.get('kernel_type', 'N/A')}")
            output.append(f"  Laplacian type: {self.Parameters.get('laplacian_type', 'N/A')}")
            if self.Parameters.get("scales") is not None:
                scales_str = ", ".join([f"{s:.4f}" for s in self.Parameters["scales"]])
                output.append(f"  Scales: {scales_str}")

        # Status
        output.append("\nStatus:")
        output.append(f"  Graph computed: {self.Graph is not None}")
        output.append(f"  Forward computed: {self.Forward is not None}")
        output.append(f"  Inverse computed: {self.Inverse is not None}")

        # Reconstruction errors
        if self.Inverse is not None:
            output.append("\nReconstruction Errors:")
            for sig, inv_data in self.Inverse.items():
                err = inv_data.get("reconstruction_error")
                if err is not None:
                    output.append(f"  {sig}: {err:.6f}")

        return "\n".join(output)


def init_sgwt(data_in: Union[pd.DataFrame, dict],
              x_col: str = "x",
              y_col: str = "y",
              signals: Optional[Union[str, List[str]]] = None,
              scales: Optional[List[float]] = None,
              J: int = 5,
              scaling_factor: float = 2,
              kernel_type: str = "heat") -> SGWT:
    """
    Initialize an SGWT object.

    Parameters
    - data_in: pandas DataFrame (or object convertible to DataFrame) with coordinate columns and signal columns
    - x_col, y_col: names of coordinate columns
    - signals: list of signal column names (if None, auto-detect all non-coordinate columns)
    - scales, J, scaling_factor, kernel_type: parameters stored for later processing
    """
    if data_in is None:
        raise ValueError("data_in must be provided")

    # ensure DataFrame
    if not isinstance(data_in, pd.DataFrame):
        data_in = pd.DataFrame(data_in)

    cols = list(data_in.columns)
    if x_col not in cols or y_col not in cols:
        raise ValueError(f"Data must contain columns: {x_col} and {y_col}")

    # auto-detect signals if not provided
    if signals is None:
        signals_list = [c for c in cols if c not in (x_col, y_col)]
        if len(signals_list) == 0:
            raise ValueError("No signal columns found in data")
    else:
        if isinstance(signals, str):
            signals_list = [signals]
        else:
            signals_list = list(signals)

    # validate signal columns exist
    missing = [s for s in signals_list if s not in cols]
    if missing:
        raise ValueError(f"Signal columns not found in data: {', '.join(missing)}")

    params = {
        "scales": scales,
        "J": J,
        "scaling_factor": scaling_factor,
        "kernel_type": kernel_type
    }

    sg = SGWT(
        data=data_in,
        x_col=x_col,
        y_col=y_col,
        signals=signals_list,
        Graph=None,
        Forward=None,
        Inverse=None,
        Parameters=params
    )

    return sg

def run_spec_graph(SG,
                   k: int = 25,
                   laplacian_type: str = "normalized",
                   length_eigenvalue: Optional[int] = None,
                   verbose: bool = True):
    """
    Build spectral graph and eigendecomposition for an SGWT-like object.

    Expects `SG` to be the Python SGWT object returned by `init_sgwt`,
    and to have attributes: data (pandas.DataFrame), x_col, y_col, Parameters (dict).
    Places a dict into `SG.Graph` with keys:
      adjacency_matrix (scipy.sparse.csr_matrix),
      laplacian_matrix (scipy.sparse.csr_matrix or ndarray),
      eigenvalues (ndarray),
      eigenvectors (ndarray)
    """
    # Basic validation
    if not hasattr(SG, "data"):
        raise ValueError("Input must be an SGWT-like object with `data` attribute")
    data_in = SG.data
    x_col = SG.x_col
    y_col = SG.y_col
    n = data_in.shape[0]

    if length_eigenvalue is None:
        length_eigenvalue = n

    if verbose:
        print("Building graph from spatial coordinates...")

    coords = data_in[[x_col, y_col]].values

    # Build k-nearest neighbor graph (include self in neighbors, will remove)
    n_neighbors = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # Build symmetric adjacency matrix
    A = sparse.lil_matrix((n, n), dtype=float)
    for i in range(n):
        for j in indices[i]:
            if j == i:
                continue
            A[i, j] = 1.0
            A[j, i] = 1.0
    A = A.tocsr()

    if verbose:
        print("Computing Laplacian and eigendecomposition...")

    # Compute Laplacian matrix
    lt = laplacian_type.lower()
    if lt == "unnormalized":
        L = csgraph.laplacian(A, normed=False)
    elif lt == "normalized":
        L = csgraph.laplacian(A, normed=True)
    elif lt == "randomwalk":
        # random-walk Laplacian: I - D^{-1} A
        deg = np.array(A.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            inv_deg = np.reciprocal(deg)
        inv_deg[np.isinf(inv_deg)] = 0.0
        Dinv = sparse.diags(inv_deg)
        L = sparse.eye(n) - Dinv.dot(A)
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")

    # Eigendecomposition: try sparse solver when k < n-1, else dense
    k_ev = int(min(length_eigenvalue, n))
    if k_ev < n - 1 and sparse.issparse(L):
        try:
            vals, vecs = eigsh(L, k=k_ev, which="SM")
        except Exception:
            # Fallback to dense decomposition
            L_dense = L.toarray() if sparse.issparse(L) else np.array(L)
            vals_all, vecs_all = np.linalg.eigh(L_dense)
            vals = vals_all[:k_ev]
            vecs = vecs_all[:, :k_ev]
    else:
        L_dense = L.toarray() if sparse.issparse(L) else np.array(L)
        vals_all, vecs_all = np.linalg.eigh(L_dense)
        vals = vals_all[:k_ev]
        vecs = vecs_all[:, :k_ev]

    # Ensure eigenvalues are sorted ascending (like R's SM)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    # Update SG.Graph
    SG.Graph = {
        "adjacency_matrix": A,
        "laplacian_matrix": L,
        "eigenvalues": vals,
        "eigenvectors": vecs
    }

    # Auto-generate scales if not provided (requires sgwt_auto_scales implementation)
    if SG.Parameters.get("scales") is None:
        lmax = float(np.max(vals)) * 0.95 if vals.size > 0 else 1.0
        # sgwt_auto_scales must be implemented elsewhere and imported
        SG.Parameters["scales"] = sgwt_auto_scales(lmax, SG.Parameters.get("J", 5),
                                                  SG.Parameters.get("scaling_factor", 2))
        if verbose:
            print("Auto-generated scales:", SG.Parameters["scales"])

    if verbose:
        print("Graph construction completed.")

    return SG


def run_sgwt(SG, use_batch: bool = True, verbose: bool = True):
    """
    Perform SGWT forward and inverse transforms on all signals in the SGWT object.

    Applies batch processing when `use_batch=True` and multiple signals exist.
    Assumes Graph slot is populated by `run_spec_graph()`.

    Parameters
    ----------
    SG : SGWT
        SGWT object with Graph slot populated
    use_batch : bool, default True
        Whether to use batch processing for multiple signals
    verbose : bool, default True
        Whether to print progress messages

    Returns
    -------
    SGWT
        Updated SGWT object with Forward and Inverse slots populated
    """
    # Validate input
    if not isinstance(SG, SGWT):
        raise ValueError("Input must be an SGWT object")
    if SG.Graph is None:
        raise ValueError("Graph slot is empty. Run run_spec_graph() first.")

    # Extract components
    eigenvalues = SG.Graph["eigenvalues"]
    eigenvectors = SG.Graph["eigenvectors"]
    params = SG.Parameters
    signals = SG.signals
    data_in = SG.data

    # Scales should have been generated in run_spec_graph
    if params.get("scales") is None:
        raise ValueError("Scales not found. Make sure to run run_spec_graph() before run_sgwt().")

    n_signals = len(signals)

    if verbose:
        print(f"Performing SGWT analysis for {n_signals} signal(s)...")

    if use_batch and n_signals > 1:
        # Batch processing for multiple signals
        if verbose:
            print("Using batch processing for efficiency...")

        # Create signal matrix (n_vertices x n_signals)
        signals_matrix = data_in[signals].values

        # Batch forward transform
        batch_forward = sgwt_forward(
            signals_matrix,
            eigenvectors,
            eigenvalues,
            params["scales"],
            kernel_type=params.get("kernel_type", "heat")
        )

        # Batch inverse transform
        batch_inverse = sgwt_inverse(batch_forward, eigenvectors, signals_matrix)

        # Split results back into individual signals
        forward_list = {}
        inverse_list = {}

        for i, sig in enumerate(signals):
            # Extract individual signal results from batch
            forward_list[sig] = {
                "fourier_coefficients": {
                    "original": batch_forward["fourier_coefficients"]["original"][:, i].copy(),
                    "filtered": {
                        key: val[:, i].copy()
                        for key, val in batch_forward["fourier_coefficients"]["filtered"].items()
                    }
                },
                "filters": batch_forward["filters"]
            }

            # Extract individual vertex approximations
            coefficients_individual = {}
            for comp_name, comp_val in batch_inverse["vertex_approximations"].items():
                if isinstance(comp_val, np.ndarray) and comp_val.ndim == 2:
                    coefficients_individual[comp_name] = comp_val[:, i].copy()
                else:
                    coefficients_individual[comp_name] = comp_val

            # Handle reconstructed signal
            recon_signal = batch_inverse["reconstructed_signal"]
            if isinstance(recon_signal, np.ndarray) and recon_signal.ndim == 2:
                recon_signal = recon_signal[:, i].copy()

            # Handle reconstruction error
            recon_error = batch_inverse["reconstruction_error"]
            if isinstance(recon_error, np.ndarray) and recon_error.ndim == 1:
                recon_error = recon_error[i]

            inverse_list[sig] = {
                "vertex_approximations": coefficients_individual,
                "reconstructed_signal": recon_signal,
                "reconstruction_error": recon_error
            }

    else:
        # Individual processing (original method)
        if verbose and n_signals > 1:
            print("Using individual processing...")

        forward_list = {}
        inverse_list = {}

        for sig in signals:
            if verbose:
                print(f"Processing signal: {sig}")

            # Extract signal vector
            sig_vec = data_in[sig].values

            # Forward transform
            fwd = sgwt_forward(
                sig_vec,
                eigenvectors,
                eigenvalues,
                params["scales"],
                kernel_type=params.get("kernel_type", "heat")
            )
            forward_list[sig] = fwd

            # Inverse transform
            inv = sgwt_inverse(fwd, eigenvectors, sig_vec)
            inverse_list[sig] = inv

    # Update SGWT object
    SG.Forward = forward_list
    SG.Inverse = inverse_list

    if verbose:
        print("SGWT analysis completed.")

    return SG


def run_sgcc(signal1, signal2, SG=None, eps: float = 1e-12, validate: bool = True,
             return_parts: bool = True, low_only: bool = False):
    """
    Run SGCC weighted similarity analysis in Fourier domain.

    Calculate energy-normalized weighted similarity between two signals
    using Fourier domain coefficients directly (no vertex domain reconstruction).
    Excludes DC component and uses energy-based weighting consistent with Parseval's theorem.

    Parameters
    ----------
    signal1 : str, SGWT, or dict
        Either a signal name (str) for SG object, SGWT object, or Forward result dict
    signal2 : str, SGWT, or dict
        Either a signal name (str) for SG object, SGWT object, or Forward result dict
    SG : SGWT, optional
        SGWT object (required if signal1/signal2 are signal names)
    eps : float, default 1e-12
        Small numeric for numerical stability
    validate : bool, default True
        If True, check consistency
    return_parts : bool, default True
        If True, return detailed components
    low_only : bool, default False
        If True, compute only low-frequency similarity

    Returns
    -------
    dict or float
        If return_parts=True, returns dict with similarity components.
        Otherwise returns scalar similarity value S.
    """
    import warnings
    import re

    def _get_decomp(x, SG=None):
        """Helper function to extract decomposition from various input types."""
        if isinstance(x, str):
            # x is a signal name
            if SG is None or SG.Forward is None:
                raise ValueError("SG object with Forward slot required when using signal names")
            if x not in SG.Forward:
                raise ValueError(f"Signal '{x}' not found in SGWT Forward results")
            return SG.Forward[x]
        elif isinstance(x, SGWT):
            # x is an SGWT object - use first signal
            if x.Forward is None or len(x.Forward) == 0:
                raise ValueError("SGWT object must have Forward results")
            return list(x.Forward.values())[0]
        elif isinstance(x, dict) and "fourier_coefficients" in x:
            # x is already a forward decomposition
            return x
        else:
            raise ValueError("Invalid input type for signal")

    # Get decompositions
    A = _get_decomp(signal1, SG)
    B = _get_decomp(signal2, SG)

    # Extract filtered Fourier coefficients
    fourier_a = A.get("fourier_coefficients", {}).get("filtered")
    fourier_b = B.get("fourier_coefficients", {}).get("filtered")

    if fourier_a is None or fourier_b is None:
        raise ValueError("Fourier coefficients not found in Forward results")

    # Extract scaling (low-pass) Fourier coefficients, excluding DC component
    if "scaling" not in fourier_a or "scaling" not in fourier_b:
        raise ValueError("Scaling coefficients not found in filtered Fourier coefficients")

    # Get scaling coefficients and exclude DC component (first element)
    f_low_a = np.asarray(fourier_a["scaling"], dtype=float)
    f_low_b = np.asarray(fourier_b["scaling"], dtype=float)

    # Exclude DC component (first coefficient, corresponding to λ = 0)
    if len(f_low_a) > 1:
        f_low_a = f_low_a[1:]
    if len(f_low_b) > 1:
        f_low_b = f_low_b[1:]

    # Handle non-finite values
    if not np.all(np.isfinite(f_low_a)):
        warnings.warn("Non-finite values found in scaling Fourier coefficients of signal1, replacing with 0")
        f_low_a[~np.isfinite(f_low_a)] = 0.0
    if not np.all(np.isfinite(f_low_b)):
        warnings.warn("Non-finite values found in scaling Fourier coefficients of signal2, replacing with 0")
        f_low_b[~np.isfinite(f_low_b)] = 0.0

    # Ensure both scaling coefficient vectors have the same length
    min_length = min(len(f_low_a), len(f_low_b))
    if len(f_low_a) != len(f_low_b):
        if validate:
            warnings.warn(
                f"Scaling coefficients have different lengths: {len(f_low_a)} vs {len(f_low_b)}"
                f". Truncating to minimum length: {min_length}"
            )
        f_low_a = f_low_a[:min_length]
        f_low_b = f_low_b[:min_length]

    # Energies in Fourier domain (consistent with Parseval's theorem)
    E_low_a = float(np.sum(f_low_a**2))
    E_low_b = float(np.sum(f_low_b**2))

    # Low-frequency cosine similarity in Fourier domain
    c_low = cosine_similarity(f_low_a, f_low_b, eps)

    # Short-circuit for low-only
    if low_only:
        if return_parts:
            return {
                "c_low": c_low,
                "c_nonlow": np.nan,
                "w_low": 1.0,
                "w_NL": 0.0,
                "S": c_low,
                "E_low_a": E_low_a,
                "E_NL_a": np.nan,
                "E_low_b": E_low_b,
                "E_NL_b": np.nan,
                "n": len(f_low_a),
                "J": np.nan
            }
        else:
            return c_low

    # Collect wavelet Fourier coefficients (non-low frequencies)
    wavelet_names_a = [k for k in fourier_a.keys() if re.match(r"^wavelet_scale_", k)]
    wavelet_names_b = [k for k in fourier_b.keys() if re.match(r"^wavelet_scale_", k)]

    if len(wavelet_names_a) == 0 or len(wavelet_names_b) == 0:
        raise ValueError("No wavelet Fourier coefficients found")

    # Order wavelet coefficients by scale index
    def extract_scale_num(name):
        match = re.match(r"^wavelet_scale_(\d+)", name)
        return int(match.group(1)) if match else 0

    wavelet_names_a = sorted(wavelet_names_a, key=extract_scale_num)
    wavelet_names_b = sorted(wavelet_names_b, key=extract_scale_num)

    # Handle different numbers of scales by using the minimum common scales
    min_scales = min(len(wavelet_names_a), len(wavelet_names_b))
    if len(wavelet_names_a) != len(wavelet_names_b):
        if validate:
            warnings.warn(
                f"Different numbers of wavelet scales: {len(wavelet_names_a)} vs {len(wavelet_names_b)}"
                f". Using first {min_scales} scales for comparison."
            )
        wavelet_names_a = wavelet_names_a[:min_scales]
        wavelet_names_b = wavelet_names_b[:min_scales]

    # Extract and flatten wavelet Fourier coefficients
    wavelet_coeffs_a = [np.asarray(fourier_a[name], dtype=float) for name in wavelet_names_a]
    wavelet_coeffs_b = [np.asarray(fourier_b[name], dtype=float) for name in wavelet_names_b]

    # Flatten all wavelet coefficients into single vectors
    f_wave_a = np.concatenate(wavelet_coeffs_a) if wavelet_coeffs_a else np.array([])
    f_wave_b = np.concatenate(wavelet_coeffs_b) if wavelet_coeffs_b else np.array([])

    # Handle non-finite values
    if not np.all(np.isfinite(f_wave_a)):
        warnings.warn("Non-finite values found in wavelet Fourier coefficients of signal1, replacing with 0")
        f_wave_a[~np.isfinite(f_wave_a)] = 0.0
    if not np.all(np.isfinite(f_wave_b)):
        warnings.warn("Non-finite values found in wavelet Fourier coefficients of signal2, replacing with 0")
        f_wave_b[~np.isfinite(f_wave_b)] = 0.0

    # Ensure both wavelet coefficient vectors have the same length
    min_wave_length = min(len(f_wave_a), len(f_wave_b))
    if len(f_wave_a) != len(f_wave_b):
        if validate:
            warnings.warn(
                f"Wavelet coefficients have different lengths: {len(f_wave_a)} vs {len(f_wave_b)}"
                f". Truncating to minimum length: {min_wave_length}"
            )
        f_wave_a = f_wave_a[:min_wave_length]
        f_wave_b = f_wave_b[:min_wave_length]

    # Energies in Fourier domain for wavelet components
    E_NL_a = float(np.sum(f_wave_a**2))
    E_NL_b = float(np.sum(f_wave_b**2))

    # Number of scales and effective signal length (excluding DC)
    J = len(wavelet_names_a)
    n = len(f_low_a)

    # Non-low cosine similarity in Fourier domain
    c_nonlow = cosine_similarity(f_wave_a, f_wave_b, eps)

    # Energy-based macro weights (consistent with Parseval's theorem and Littlewood-Paley)
    w_low_a = E_low_a / (E_low_a + E_NL_a + eps)
    w_low_b = E_low_b / (E_low_b + E_NL_b + eps)
    w_low = np.clip(0.5 * (w_low_a + w_low_b), 0, 1)
    w_NL = 1.0 - w_low
    S = w_low * c_low + w_NL * c_nonlow

    # Validation
    if validate:
        if len(wavelet_names_a) != len(wavelet_names_b):
            raise ValueError("Number of wavelet scales must match")
        if J < 1:
            raise ValueError("Each SGWT decomposition must have at least 1 wavelet scale")

    # Return results
    if return_parts:
        return {
            "c_low": c_low,
            "c_nonlow": c_nonlow,
            "w_low": w_low,
            "w_NL": w_NL,
            "S": S,
            "E_low_a": E_low_a,
            "E_NL_a": E_NL_a,
            "E_low_b": E_low_b,
            "E_NL_b": E_NL_b,
            "n": n,
            "J": J
        }
    else:
        return S


def sgwt_forward(signal, eigenvectors, eigenvalues, scales, lmax=None, kernel_type="heat"):
    """
    Perform forward SGWT transform on signal(s).
    
    Parameters
    ----------
    signal : ndarray
        Signal vector (1D) or batch of signals (2D, shape: vertices x signals)
    eigenvectors : ndarray
        Eigenvectors of the graph Laplacian (shape: vertices x vertices)
    eigenvalues : ndarray
        Eigenvalues of the graph Laplacian
    scales : array-like
        Scales for wavelet filters
    lmax : float, optional
        Maximum eigenvalue for filter normalization. If None, uses max(eigenvalues)
    kernel_type : str, default="heat"
        Type of kernel: "heat", "meyer", etc.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'fourier_coefficients': dict with 'original' and 'filtered' coefficients
        - 'filters': list of computed filters
    """
    signal = np.asarray(signal)
    eigenvectors = np.asarray(eigenvectors)
    eigenvalues = np.asarray(eigenvalues)
    
    # Validate dimensions
    if signal.ndim == 2:
        if eigenvectors.shape[0] != signal.shape[0]:
            raise ValueError("Number of vertices in signals matrix must match eigenvectors")
    else:
        if eigenvectors.shape[0] != signal.shape[0]:
            raise ValueError("Number of vertices in signal vector must match eigenvectors")
    
    # Compute filters
    filters = compute_sgwt_filters(eigenvalues, scales, lmax, kernel_type)
    
    # Transform signal(s) to spectral domain using GFT
    # gft handles both vectors and matrices automatically
    signal_hat = gft(signal, eigenvectors)
    
    # Store original and filtered Fourier coefficients
    fourier_coefficients = {
        'original': signal_hat,
        'filtered': {}
    }
    
    # Apply filters and store filtered Fourier coefficients
    for i, filt in enumerate(filters):
        # Apply filter - works for both single signals and matrices
        filtered_spectrum = signal_hat * np.asarray(filt).flatten()[:, np.newaxis] if signal_hat.ndim == 2 else signal_hat * np.asarray(filt).flatten()
        fourier_coefficients['filtered'][f'wavelet_scale_{i}' if i > 0 else 'scaling'] = filtered_spectrum
    
    # Rename keys for consistency
    keys_list = list(fourier_coefficients['filtered'].keys())
    for i, key in enumerate(keys_list):
        if i == 0:
            fourier_coefficients['filtered']['scaling'] = fourier_coefficients['filtered'].pop(key)
        else:
            new_key = f'wavelet_scale_{i-1}'
            if key != new_key:
                fourier_coefficients['filtered'][new_key] = fourier_coefficients['filtered'].pop(key)
    
    return {
        'fourier_coefficients': fourier_coefficients,
        'filters': filters
    }


def sgwt_energy_analysis(SG, signal_name=None):
    """
    Analyze energy distribution across frequency scales in SGWT decomposition.
    
    Parameters
    ----------
    SG : SGWT
        An SGWT object with Forward results computed
    signal_name : str, optional
        Name of the signal to analyze. If None, defaults to the first signal.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'scale': scale name (e.g., 'low_pass', 'wavelet_1', etc.)
        - 'energy': energy value at that scale (in Fourier domain)
        - 'energy_ratio': normalized energy (ratio to total energy)
        - 'scale_value': scale parameter value
        - 'signal': name of the signal analyzed
        
    Raises
    ------
    ValueError
        If input is not an SGWT object, Forward results not computed, 
        or signal name not found
    """
    # Validate input
    if not isinstance(SG, SGWT):
        raise ValueError("Input must be an SGWT object")
    if SG.Forward is None:
        raise ValueError("SGWT object must have Forward results computed")
    
    # Default to first signal if not specified
    if signal_name is None:
        signal_name = list(SG.Forward.keys())[0]
    
    # Validate signal exists
    if signal_name not in SG.Forward:
        raise ValueError(f"Signal '{signal_name}' not found in SGWT Forward results")
    
    # Get Forward results and scales from Parameters
    forward_result = SG.Forward[signal_name]
    fourier_coeffs = forward_result['fourier_coefficients']['filtered']
    scales = SG.Parameters['scales']
    
    if fourier_coeffs is None:
        raise ValueError("Fourier coefficients not found in Forward results")
    
    # Calculate energies in Fourier domain (consistent with Parseval's theorem)
    energies = []
    scale_names = []
    scale_values = []
    
    # Scaling (low-pass) energy - exclude DC component
    if 'scaling' in fourier_coeffs:
        scaling_coeffs = np.asarray(fourier_coeffs['scaling']).flatten()
        # Exclude DC component (first coefficient)
        if len(scaling_coeffs) > 1:
            scaling_coeffs = scaling_coeffs[1:]
        scaling_energy = np.sum(np.abs(scaling_coeffs) ** 2)
        
        energies.append(scaling_energy)
        scale_names.append('low_pass')
        scale_values.append(scales[0])  # Use first scale for scaling function
    
    # Wavelet energies - exclude DC components
    wavelet_keys = [k for k in fourier_coeffs.keys() if re.match(r'^wavelet_scale_', k)]
    
    if len(wavelet_keys) > 0:
        # Extract scale indices and sort
        scale_indices = []
        for key in wavelet_keys:
            idx = int(re.sub(r'^wavelet_scale_', '', key))
            scale_indices.append(idx)
        
        # Sort wavelet keys by scale index
        sorted_pairs = sorted(zip(scale_indices, wavelet_keys))
        sorted_indices = [idx for idx, _ in sorted_pairs]
        sorted_keys = [key for _, key in sorted_pairs]
        
        for i, wavelet_key in enumerate(sorted_keys):
            wavelet_coeffs = np.asarray(fourier_coeffs[wavelet_key]).flatten()
            # Exclude DC component if present
            if len(wavelet_coeffs) > 1:
                wavelet_coeffs = wavelet_coeffs[1:]
            wavelet_energy = np.sum(np.abs(wavelet_coeffs) ** 2)
            
            energies.append(wavelet_energy)
            scale_names.append(f'wavelet_{sorted_indices[i]}')
            scale_values.append(scales[sorted_indices[i]])
    
    # Calculate energy ratios
    total_energy = np.sum(energies)
    energy_ratios = [e / total_energy if total_energy > 0 else 0 for e in energies]
    
    # Create results DataFrame
    energy_df = pd.DataFrame({
        'scale': scale_names,
        'energy': energies,
        'energy_ratio': energy_ratios,
        'scale_value': scale_values,
        'signal': signal_name
    })
    
    return energy_df


def plot_sgwt_decomposition(SG, signal_name=None, plot_scales=None, ncol=3):
    """
    Plot SGWT decomposition components including original, scaling, wavelets, and reconstructed signals.
    
    Parameters
    ----------
    SG : SGWT
        An SGWT object with Forward and Inverse results computed
    signal_name : str, optional
        Name of the signal to plot. If None, defaults to the first signal.
    plot_scales : list of int, optional
        Wavelet scales to plot (e.g., [1, 2, 3]). If None, defaults to first 4 scales.
    ncol : int, default=3
        Number of columns in the subplot grid
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the subplot grid
        
    Raises
    ------
    ValueError
        If input is not an SGWT object, Forward/Inverse results not computed,
        or signal name not found
    """
    # Validate input
    if not isinstance(SG, SGWT):
        raise ValueError("Input must be an SGWT object")
    if SG.Forward is None or SG.Inverse is None:
        raise ValueError("SGWT object must have Forward and Inverse results computed")
    
    # Default to first signal if not specified
    if signal_name is None:
        signal_name = list(SG.Forward.keys())[0]
    
    # Validate signal exists
    if signal_name not in SG.Forward:
        raise ValueError(f"Signal '{signal_name}' not found in SGWT results")
    
    # Get decomposition and inverse results
    inverse_result = SG.Inverse[signal_name]
    coefficients = inverse_result['vertex_approximations']
    
    # Default scales to plot
    if plot_scales is None:
        n_wavelets = len(coefficients) - 1  # Exclude scaling
        plot_scales = list(range(1, min(5, n_wavelets + 1)))
    
    # Prepare data for plotting
    data_in = SG.data
    x_col = SG.x_col
    y_col = SG.y_col
    
    # Helper function to create individual plots
    def create_plot(ax, data, x_col, y_col, fill_var, title, subtitle=None):
        """
        Create a single tile plot on the given axes.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        data : pd.DataFrame
            Data containing x, y coordinates and fill variable
        x_col : str
            Column name for x coordinates
        y_col : str
            Column name for y coordinates
        fill_var : str
            Column name for color values
        title : str
            Plot title
        subtitle : str, optional
            Plot subtitle
        """
        # Create scatter plot with tiles (using color mapping)
        x_vals = data[x_col].values
        y_vals = data[y_col].values
        z_vals = data[fill_var].values
        
        # Normalize the values for color mapping
        norm = mcolors.Normalize(vmin=np.nanmin(z_vals), vmax=np.nanmax(z_vals))
        # Use square markers to produce tiled/square appearance instead of circular dots
        scatter = ax.scatter(
            x_vals,
            y_vals,
            c=z_vals,
            cmap='viridis',
            s=300,
            norm=norm,
            edgecolors='none',
            marker='s',
            linewidths=0
        )
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(title, fontsize=10, ha='center')
        if subtitle is not None:
            ax.text(0.5, -0.1, subtitle, transform=ax.transAxes, 
                   fontsize=8, ha='center')
        ax.set_aspect('equal')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Calculate grid dimensions
    plot_list = ['original', 'scaling']
    plot_list.extend([f'wavelet_{i}' for i in plot_scales 
                     if f'wavelet_{i}' in coefficients])
    plot_list.append('reconstructed')
    
    n_plots = len(plot_list)
    nrow = int(np.ceil(n_plots / ncol))
    
    # Create figure and subplots
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.5, nrow * 3.5))
    
    # Flatten axes array for easier iteration (handles both single and multiple subplots)
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot original signal
    plot_data = data_in.copy()
    plot_data['original'] = data_in[signal_name].astype(float)
    create_plot(axes[plot_idx], plot_data, x_col, y_col, 'original', 
               f'Original Signal: {signal_name}')
    plot_idx += 1
    
    # Plot scaling function coefficients (low_pass)
    if 'low_pass' in coefficients:
        plot_data = data_in.copy()
        plot_data['scaling'] = np.real(coefficients['low_pass']).flatten()
        create_plot(axes[plot_idx], plot_data, x_col, y_col, 'scaling', 
                   'Low-pass (Scaling)')
        plot_idx += 1
    
    # Plot wavelet coefficients at selected scales
    wavelet_names = [k for k in coefficients.keys() if k.startswith('wavelet_')]
    for scale_num in plot_scales:
        wavelet_name = f'wavelet_{scale_num}'
        if wavelet_name in wavelet_names:
            plot_data = data_in.copy()
            plot_data[wavelet_name] = np.real(coefficients[wavelet_name]).flatten()
            create_plot(axes[plot_idx], plot_data, x_col, y_col, wavelet_name,
                       f'Band-pass Scale {scale_num}')
            plot_idx += 1
    
    # Plot reconstructed signal
    plot_data = data_in.copy()
    reconstructed_signal = inverse_result['reconstructed_signal']
    if isinstance(reconstructed_signal, np.ndarray):
        plot_data['reconstructed'] = reconstructed_signal.flatten()
    else:
        plot_data['reconstructed'] = reconstructed_signal
    
    rmse = inverse_result.get('reconstruction_error', None)
    subtitle = None
    if rmse is not None:
        if isinstance(rmse, np.ndarray):
            rmse = rmse.item() if rmse.size == 1 else rmse[0]
        subtitle = f'RMSE: {rmse:.4f}'
    
    create_plot(axes[plot_idx], plot_data, x_col, y_col, 'reconstructed',
               'Reconstructed', subtitle)
    plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    return fig
    