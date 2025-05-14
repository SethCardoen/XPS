import numpy as np
import matplotlib.pyplot as plt

def shirley_background(x, y, tol=1e-5, max_iter=5000):
    y = np.array(y)
    x = np.array(x)
    y0, y1 = y[0], y[-1]
    bg = np.linspace(y0, y1, len(y)) # Initial guess: linear interpolation between ends

    for _ in range(max_iter):
        integral = np.cumsum(y - bg)
        integral -= integral[0]
        integral = y0 + (y1 - y0) * (integral / integral[-1])
        if np.max(np.abs(bg - integral)) < tol:
            break
        bg = integral

    return bg

# Select element: "Mg" or "Al"
element = "Al"
#element = "Mg"
base_path = f"GR_SiC_0_C/{element}300w/" # before annealing
#base_path = f"GR_SiC_250_C/{element}300w/" # after annealing
files = ["70_130.txt", "140_200.txt", "260_300.txt", "500_550.txt"]
titles = ["70–130", "140–200", "260–300", "500–550"]

crop_ranges = {
    "70_130": (96, 110),
    "140_200": (145, 160),
    "260_300": (280, 290),
    "500_550": (526, 540)
}

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

for ax, file, title in zip(axs.flat, files, titles):
    label = file.replace(".txt", "")
    data = np.loadtxt(base_path + file, delimiter="\t")
    x = data[:, 0]
    y = data[:, 1]
    y_err = np.sqrt(y)

    # Crop data before background subtraction
    start, end = crop_ranges[label]
    mask = (x >= start) & (x <= end)
    x = x[mask]
    y = y[mask]
    y_err = y_err[mask]

    # Apply Shirley background subtraction to cropped data
    bg = shirley_background(x, y)
    y_corrected = y - bg

    ax.plot(x, y, label="Raw data", color='red', alpha=1.0)
    ax.errorbar(x, y, yerr=y_err, fmt='none', ecolor='blue', alpha=0.2, elinewidth=1, capsize=2)
    ax.plot(x, y_corrected, label="Background-subtracted")
    ax.plot(x, bg, color='black', linewidth=1.5, label="Shirley background")
    ax.set_title(f"{element} {title}")
    ax.set_xlabel("Binding Energy")
    ax.set_ylabel("Intensity")
    ax.invert_xaxis()
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.suptitle(f"{element} XPS Data with Shirley Background (2×2)", y=1.03)
plt.show()