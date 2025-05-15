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

# element selection
element = "Mg"

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
area_fraction_data = {"Si": [], "C": [], "O": []}
all_voigt_plots = []

for label in annealing_labels:
    base_path = f"{annealing_dirs[label]}/{element}300w/"
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
        if key not in ["70_130", "260_300", "500_550"]:
            continue
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
            all_voigt_plots.append((f"{label} - {key}", x_vals, y_vals, best_fit))

    for el in ["Si", "C", "O"]:
        percentage_data[el].append(0)

    element_area_contributions = {"Si": 0.0, "C": 0.0, "O": 0.0}
    for key, fit in zip(ratio_dict.keys(), all_voigt_plots[-len(ratio_dict):]):
        label_fit, x_fit, _, y_fit = fit
        el = element_map[key]
        area = np.trapezoid(y_fit, x_fit)
        element_area_contributions[el] += area

    total_area = sum(element_area_contributions.values())

    print(f"\nAreas for {label}:")
    for el in ["Si", "C", "O"]:
        print(f"{el} area: {element_area_contributions[el]}")
    print(f"Total area: {total_area}")

    for el in ["Si", "C", "O"]:
        percent = (element_area_contributions[el] / total_area * 100) if total_area > 0 else 0
        percentage_data[el][-1] = percent

    for el in ["Si", "C", "O"]:
        fraction = (element_area_contributions[el] / total_area) if total_area != 0 else 0
        area_fraction_data[el].append(fraction)

elements = ["Si", "C", "O"]

fig_frac, axs_frac = plt.subplots(1, 3, figsize=(15, 4))

for i, el in enumerate(elements):
    percent_vals = [v * 100 for v in area_fraction_data[el]]
    axs_frac[i].plot(temperature_values, percent_vals, marker='o')
    axs_frac[i].set_ylabel(f"{el} (%)")
    axs_frac[i].set_title(f"{el} Area Fraction vs Annealing Temperature")
    axs_frac[i].grid(True)
    y_center = np.mean(percent_vals)
    y_range = max(5, (max(percent_vals) - min(percent_vals)) * 1.2)
    axs_frac[i].set_ylim(y_center - y_range / 2, y_center + y_range / 2)

axs_frac[1].set_xlabel("Annealing Temperature (°C)")
plt.tight_layout()
plt.show()

# Plot all Voigt fits in a 3x3 grid
fig_voigt, axs_voigt = plt.subplots(3, 3, figsize=(15, 12))
axs_voigt = axs_voigt.flatten()

filtered_voigt_plots = [tpl for tpl in all_voigt_plots if any(k in tpl[0] for k in ["70_130", "260_300", "500_550"])]
for i, (title, x, y, fit) in enumerate(filtered_voigt_plots[:9]):
    axs_voigt[i].plot(x, y, label='Data')
    axs_voigt[i].plot(x, fit, label='Voigt Fit', linestyle='--')
    title = title.replace("70_130", "Silicon").replace("260_300", "Carbon").replace("500_550", "Oxygen")
    axs_voigt[i].set_title(title)
    axs_voigt[i].legend()
    axs_voigt[i].grid(True)
    axs_voigt[i].relim()
    axs_voigt[i].autoscale_view()

plt.tight_layout()
plt.show()