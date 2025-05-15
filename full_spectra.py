# plot full spectra

import numpy as np
import matplotlib.pyplot as plt

# Define file paths
mg_path = "./GR_SiC_0_C/Mg300w/full_scan.txt"
al_path = "./GR_SiC_0_C/Al300w/full_scan.txt"

#mg_path = "./GR_SiC_450_C/Mg300w/full_scan.txt"
#al_path = "./GR_SiC_450_C/Al300w/full_scan.txt"

mg_data = np.loadtxt(mg_path, delimiter="\t")
al_data = np.loadtxt(al_path, delimiter="\t")

# Plot full spectrum
plt.figure(figsize=(10, 6))
x_mg = mg_data[:, 0]
y_mg = mg_data[:, 1]
x_al = al_data[:, 0]
y_al = al_data[:, 1]


plt.plot(x_mg, y_mg, label='Mg source', linewidth=1.5)
plt.plot(x_al, y_al, label='Al source', linewidth=1.5)

plt.title("Full Spectrum Comparison: GR_SiC_0_C")
plt.xlabel("Binding Energy (eV)")
plt.ylabel("Intensity (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()