# Shows influence to SI O C destribution with annealing



import numpy as np
import matplotlib.pyplot as plt

# Define annealing folders
annealing_dirs = {
    "0°C": "GR_SiC_0_C",
    "250°C": "GR_SiC_250_C",
    "450°C": "GR_SiC_450_C"
}
annealing_labels = list(annealing_dirs.keys())
temperature_values = [0, 250, 450]

# Element mapping
element_map = {
    "70_130": "Si",
    "140_200": "Si",
    "260_300": "C",
    "500_550": "O"
}

# X-ray constants
lambda_Al = {"O": 15.85, "C": 18.95, "Si": 21.2}
sigma_Al = {"O": 0.04005, "C": 0.01367, "Si": 0.01303}

# Data files
files = ["70_130.txt", "140_200.txt", "260_300.txt", "500_550.txt"]

# Crop ranges
crop_ranges = {
    "70_130": (96, 110),
    "140_200": (145, 160),
    "260_300": (280, 290),
    "500_550": (526, 540)
}

# Background subtraction
def shirley_background(y, max_iter=100, tol=1e-5):
    y = np.array(y)
    y0, y1 = y[0], y[-1]
    background = np.linspace(y0, y1, len(y))
    for _ in range(max_iter):
        prev_background = background.copy()
        integral = np.cumsum(y - background)
        integral -= integral[0]
        integral = y0 + (y1 - y0) * (integral / integral[-1])
        if np.max(np.abs(integral - prev_background)) < tol:
            break
        background = integral
    return background

# Voigt profile
from scipy.optimize import curve_fit
from scipy.special import wofz

def voigt(x, amplitude, center, sigma, gamma):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

max_voigts = 3
percentage_data = {"Si": [], "C": [], "O": []}

for label in annealing_labels:
    base_path = f"{annealing_dirs[label]}/Al300w/"
    ratio_dict = {}
    cropped_bgsub_data = {}
    x_cropped_dict = {}

    for file in files:
        fname = base_path + file
        try:
            data = np.loadtxt(fname, delimiter="\t")
            x = data[:, 0]
            y = data[:, 1]
            start, end = crop_ranges[file.replace(".txt", "")]
            mask = (x >= start) & (x <= end)
            x_crop = x[mask]
            y_crop = y[mask]
            bg = shirley_background(y_crop)
            y_corrected = y_crop - bg
            cropped_bgsub_data[file.replace(".txt", "")] = y_corrected
            x_cropped_dict[file.replace(".txt", "")] = x_crop
        except FileNotFoundError:
            print(f"[Warning] Missing file: {fname}")
            continue

    for key in crop_ranges:
        x_vals = x_cropped_dict.get(key)
        y_vals = cropped_bgsub_data.get(key)
        if x_vals is None or y_vals is None:
            continue
        amp_guess = np.max(y_vals)
        center_guess = x_vals[np.argmax(y_vals)]
        width = (x_vals[-1] - x_vals[0]) / 5
        best_chi_sq = np.inf
        best_fit = None
        best_popt = None

        for n_voigts in range(1, max_voigts + 1):
            p0 = []
            offset = (x_vals[-1] - x_vals[0]) / (2 * n_voigts)
            for i in range(n_voigts):
                amp = amp_guess / n_voigts
                center = np.clip(center_guess + (i - n_voigts // 2) * offset, min(x_vals), max(x_vals))
                sigma = max(width, 0.1)
                gamma = max(width, 0.1)
                p0 += [amp, center, sigma, gamma]
            lower_bounds = [0, min(x_vals), 1e-6, 1e-6] * n_voigts
            upper_bounds = [np.inf, max(x_vals), max(x_vals)-min(x_vals), max(x_vals)-min(x_vals)] * n_voigts

            try:
                popt, _ = curve_fit(lambda x, *params: sum(voigt(x, *params[i:i+4]) for i in range(0, len(params), 4)),
                                    x_vals, y_vals, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=20000)
                fit_result = sum(voigt(x_vals, *popt[i:i+4]) for i in range(0, len(popt), 4))
                chi_sq = np.sum(((y_vals - fit_result) / np.sqrt(np.maximum(y_vals, 1))) ** 2) / (len(y_vals) - len(popt))
                if chi_sq < best_chi_sq:
                    best_chi_sq = chi_sq
                    best_fit = fit_result
                    best_popt = popt
            except RuntimeError:
                continue

        if best_popt is not None:
            total_area = np.trapezoid(sum(voigt(x_vals, *best_popt[i:i+4]) for i in range(0, len(best_popt), 4)), x_vals)
            measured_element = element_map[key]
            lam = lambda_Al[measured_element]
            sig = sigma_Al[measured_element]
            ratio = -total_area / (lam * sig)
            ratio_dict[key] = ratio

    total_ratio = 0
    element_contributions = {}
    for key, value in ratio_dict.items():
        el = element_map[key]
        if el == "Si" and key != "140_200":
            continue
        total_ratio += value
        element_contributions[el] = element_contributions.get(el, 0) + value

    for el in ["Si", "C", "O"]:
        percent = (element_contributions.get(el, 0) / total_ratio) * 100 if total_ratio > 0 else 0
        percentage_data[el].append(percent)

# Plotting 1x3 layout
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

elements = ["Si", "C", "O"]
for i, el in enumerate(elements):
    axs[i].plot(temperature_values, percentage_data[el], marker='o')
    axs[i].set_ylabel(f"{el} (%)")
    axs[i].set_title(f"{el} Percentage vs Annealing Temperature")
    axs[i].grid(True)
    y_vals = percentage_data[el]
    y_center = np.mean(y_vals)
    axs[i].set_ylim(y_center - 2.5, y_center + 2.5)

axs[1].set_xlabel("Annealing Temperature (°C)")
plt.tight_layout()
plt.show()