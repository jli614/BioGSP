import numpy as np
import pandas as pd
from sgwt_main import init_sgwt, run_spec_graph, run_sgwt, sgwt_energy_analysis, plot_sgwt_decomposition
import matplotlib
matplotlib.interactive(False)
import matplotlib.pyplot as plt
# Set random seed for reproducibility
np.random.seed(123)

# Step 1: Create synthetic data
x = np.tile(np.arange(1, 11), 10)
y = np.repeat(np.arange(1, 11), 10)
signal1 = np.sin(0.5 * np.tile(np.arange(1, 11), 10)) + np.random.normal(0, 0.1, 100)
signal2 = np.cos(0.3 * np.repeat(np.arange(1, 11), 10)) + np.random.normal(0, 0.1, 100)

demo_data = pd.DataFrame({
    'x': x,
    'y': y,
    'signal1': signal1,
    'signal2': signal2
})

# Step 2: Initialize SGWT object
SG = init_sgwt(demo_data, 
               signals=['signal1', 'signal2'], 
               J=4, 
               kernel_type="heat")

# Step 3: Build spectral graph (scales auto-generated based on eigenvalues)
SG = run_spec_graph(SG, k=8, laplacian_type="normalized", length_eigenvalue=30)

# Visualize Fourier modes - (plot_FM not yet translated)
# fourier_modes = plot_FM(SG, mode_type="both", n_modes=5, ncol=5)
# print(fourier_modes)

# Step 4: Run SGWT analysis
SG = run_sgwt(SG)

# Step 5: Check results
print(SG)

# Analyze energy distribution - (sgwt_energy_analysis not yet translated)
energy_analysis = sgwt_energy_analysis(SG, "signal1")
print(energy_analysis)

# Visualize decomposition using tiled squares (imshow) to match R output
sig = 'signal1'
inv = SG.Inverse[sig]
coeffs = inv['vertex_approximations']

# Build ordered list of plots: original, low_pass, wavelets (1..), reconstructed
plot_keys = ['original', 'low_pass']
wavelet_keys = sorted([k for k in coeffs.keys() if k.startswith('wavelet_')],
                      key=lambda s: int(s.split('_')[-1]))
plot_keys.extend(wavelet_keys)
plot_keys.append('reconstructed')

# Prepare grid mapping helper
def to_grid(df, values):
    tmp = df.copy()
    tmp['_val'] = np.asarray(values).flatten()
    grid = tmp.pivot(index='y', columns='x', values='_val')
    # ensure numeric ordering of rows/cols
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    return grid.values

# Prepare figure layout
ncol = 3
n_plots = len(plot_keys)
nrow = int(np.ceil(n_plots / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 4))
axes = np.array(axes).reshape(-1)

plot_idx = 0
# Original
orig_grid = to_grid(demo_data, demo_data[sig].values)
axes[plot_idx].imshow(orig_grid, cmap='viridis', origin='lower', aspect='equal', interpolation='nearest')
axes[plot_idx].set_title(f'Original Signal: {sig}', fontsize=10)
axes[plot_idx].axis('off')
plot_idx += 1

# Low-pass (scaling) if present
if 'low_pass' in coeffs:
    grid = to_grid(demo_data, np.real(coeffs['low_pass']))
    axes[plot_idx].imshow(grid, cmap='viridis', origin='lower', aspect='equal', interpolation='nearest')
    axes[plot_idx].set_title('Low-pass (Scaling)', fontsize=10)
    axes[plot_idx].axis('off')
    plot_idx += 1

# Wavelet scales
for wk in wavelet_keys:
    if plot_idx >= len(axes):
        break
    grid = to_grid(demo_data, np.real(coeffs[wk]))
    scale_num = wk.split('_')[-1]
    axes[plot_idx].imshow(grid, cmap='viridis', origin='lower', aspect='equal', interpolation='nearest')
    axes[plot_idx].set_title(f'Band-pass Scale {scale_num}', fontsize=10)
    axes[plot_idx].axis('off')
    plot_idx += 1

# Reconstructed
recon = inv['reconstructed_signal']
recon_grid = to_grid(demo_data, recon)
rmse = inv.get('reconstruction_error', None)
axes[plot_idx].imshow(recon_grid, cmap='viridis', origin='lower', aspect='equal', interpolation='nearest')
subtitle = None
if rmse is not None:
    try:
        if isinstance(rmse, np.ndarray):
            rmse = rmse.item() if rmse.size == 1 else rmse[0]
    except Exception:
        pass
    subtitle = f'RMSE: {rmse:.4f}'
axes[plot_idx].set_title('Reconstructed' + (f'\n{subtitle}' if subtitle else ''), fontsize=10)
axes[plot_idx].axis('off')
plot_idx += 1

# Hide any unused axes
for ax in axes[plot_idx:]:
    ax.set_visible(False)

plt.tight_layout()
fig.savefig('Figure_1.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
print('Saved Figure_1.png')
plt.close(fig)

#visualize_similarity_xy