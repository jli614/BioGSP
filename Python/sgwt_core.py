import numpy as np
import re


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


def gft(signal, eigenvectors):
    """
    Graph Fourier Transform (forward transform).
    
    Projects signal(s) onto the eigenvector basis of the graph Laplacian.
    
    Parameters
    ----------
    signal : ndarray
        Signal vector (1D, shape: n_vertices) or batch of signals (2D, shape: n_vertices x n_signals)
    eigenvectors : ndarray
        Eigenvectors of the graph Laplacian (shape: n_vertices x n_vertices)
        
    Returns
    -------
    ndarray
        Fourier coefficients. Same shape as signal if signal is 1D,
        or 2D array with same number of columns if signal is 2D (n_vertices x n_signals)
    """
    signal = np.asarray(signal, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    
    # Compute Fourier coefficients: signal_hat = U^T @ signal
    # Works for both 1D and 2D arrays (matrix multiplication handles broadcasting)
    signal_hat = eigenvectors.T @ signal
    
    return signal_hat


def igft(fourier_coefficients, eigenvectors):
    """
    Inverse Graph Fourier Transform (inverse transform).
    
    Reconstructs signal(s) from Fourier coefficients using the eigenvector basis.
    
    Parameters
    ----------
    fourier_coefficients : ndarray
        Fourier coefficients (1D, shape: n_vertices) or batch (2D, shape: n_vertices x n_signals)
    eigenvectors : ndarray
        Eigenvectors of the graph Laplacian (shape: n_vertices x n_vertices)
        
    Returns
    -------
    ndarray
        Reconstructed signal(s). Same shape as fourier_coefficients
    """
    fourier_coefficients = np.asarray(fourier_coefficients, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    
    # Reconstruct signal: signal = U @ signal_hat
    # Works for both 1D and 2D arrays (matrix multiplication handles broadcasting)
    signal_reconstructed = eigenvectors @ fourier_coefficients
    
    return signal_reconstructed


def sgwt_get_kernels(kernel_type="heat"):
    """
    Get kernel family (scaling and wavelet functions).
    
    Parameters
    ----------
    kernel_type : str, default="heat"
        Type of kernel: "heat", "meyer", or "mexican_hat"
        
    Returns
    -------
    dict
        Dictionary with 'scaling' and 'wavelet' function closures
    """
    if kernel_type == "mexican_hat":
        def scaling_fun(x, scale_param):
            t = x / scale_param
            return np.exp(-0.5 * t ** 2)
        
        def wavelet_fun(x, scale_param):
            t = x / scale_param
            return (t ** 2) * np.exp(-0.5 * t ** 2)
        
        return {'scaling': scaling_fun, 'wavelet': wavelet_fun}
    
    elif kernel_type == "meyer":
        def scaling_fun(x, scale_param):
            a = 0.5 * scale_param
            b = 1.0 * scale_param
            if x <= a:
                return 1.0
            elif x >= b:
                return 0.0
            else:
                s = (x - a) / (b - a)
                return np.cos(np.pi / 2 * s)
        
        def wavelet_fun(x, scale_param):
            a = 0.5 * scale_param
            b = 1.0 * scale_param
            c = 2.0 * scale_param
            if x <= a or x >= c:
                return 0.0
            elif x <= b:
                s = (x - a) / (b - a)
                return np.sin(np.pi / 2 * s)
            else:
                s = (x - b) / (c - b)
                return np.cos(np.pi / 2 * s)
        
        return {'scaling': scaling_fun, 'wavelet': wavelet_fun}
    
    elif kernel_type == "heat":
        def scaling_fun(x, scale_param):
            # Heat kernel scaling: exponential decay h(t) = exp(-t)
            t = x / scale_param
            return np.exp(-t)
        
        def wavelet_fun(x, scale_param):
            # Heat kernel wavelet: derivative-like g(t) = t * exp(-t)
            t = x / scale_param
            return t * np.exp(-t)
        
        return {'scaling': scaling_fun, 'wavelet': wavelet_fun}
    
    else:
        raise ValueError(f"Kernel type '{kernel_type}' not supported. Use 'mexican_hat', 'meyer', or 'heat'.")


def compute_sgwt_filters(eigenvalues, scales, lmax=None, kernel_type="heat"):
    """
    Compute SGWT filter bank (scaling and wavelet filters).
    
    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalues of the graph Laplacian
    scales : array-like
        Scales for the wavelet filters
    lmax : float, optional
        Maximum eigenvalue. If None, uses max(eigenvalues) * 0.95
    kernel_type : str, default="heat"
        Type of kernel: "heat", "meyer", or "mexican_hat"
        
    Returns
    -------
    list
        List of filters [scaling_filter, wavelet_filter_1, wavelet_filter_2, ...]
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    scales = np.asarray(scales, dtype=float)
    
    if lmax is None:
        lmax = np.max(eigenvalues) * 0.95  # Avoid numerical issues at lambda_max
    
    J = len(scales)  # Number of scales
    
    # Initialize filter bank: J wavelets + 1 scaling function
    filters = []
    
    # Get unified kernel family
    kernels = sgwt_get_kernels(kernel_type)
    
    # Scaling function (low-pass filter)
    scaling_filter = np.array([kernels['scaling'](lam, scales[0]) for lam in eigenvalues])
    filters.append(scaling_filter)
    
    # Wavelet functions at different scales
    for j in range(J):
        wavelet_filter = np.array([kernels['wavelet'](lam, scales[j]) for lam in eigenvalues])
        filters.append(wavelet_filter)
    
    return filters