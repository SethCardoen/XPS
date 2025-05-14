import numpy as np
import matplotlib.pyplot as plt

# Define file paths
mg_path = "./GR_SiC_0_C/Mg300w/70_130.txt"
al_path = "./GR_SiC_0_C/Al300w/70_130.txt"

# Load the data
mg_data = np.loadtxt(mg_path, delimiter="\t")
al_data = np.loadtxt(al_path, delimiter="\t")

# Extract x (binding energy) and y (intensity) values
x_mg = mg_data[:, 0]
y_mg = mg_data[:, 1]
x_al = al_data[:, 0]
y_al = al_data[:, 1]

# Crop data to the 95–110 eV range
mg_mask = (x_mg >= 95) & (x_mg <= 110)
al_mask = (x_al >= 95) & (x_al <= 110)
x_mg, y_mg = x_mg[mg_mask], y_mg[mg_mask]
x_al, y_al = x_al[al_mask], y_al[al_mask]

# Plot setup
plt.figure(figsize=(10, 6))

# Plot base spectra in light grey
plt.plot(x_mg, y_mg, color='lightgrey', linewidth=1.5)
plt.plot(x_al, y_al, color='lightgrey', linewidth=1.5)

# Highlight Mg peaks
peak1_mg = (x_mg > 99) & (x_mg < 101.5)
peak2_mg = (x_mg > 101.5) & (x_mg < 105)
plt.plot(x_mg[peak1_mg], y_mg[peak1_mg], color='#1b9e77', label='Mg SiC')
plt.plot(x_mg[peak2_mg], y_mg[peak2_mg], color='#d95f02', label='Mg Oxidised SiC')

# Highlight Al peaks
peak1_al = (x_al > 99) & (x_al < 101.5)
peak2_al = (x_al > 101.5) & (x_al < 105)
plt.plot(x_al[peak1_al], y_al[peak1_al], color='#7570b3', label='Al SiC')
plt.plot(x_al[peak2_al], y_al[peak2_al], color='#e7298a', label='Al Oxidised SiC')

# Final plot formatting
plt.title("Full Spectrum Comparison: GR_SiC_0_C")
plt.xlabel("Binding Energy (eV)")
plt.ylabel("Intensity (a.u.)")
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.show()