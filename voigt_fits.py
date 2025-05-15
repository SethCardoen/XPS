# not really necessairy just extra

import numpy as np
import sympy as sp
import math

# Optimization and curve fitting libraries
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

# Statistical and special functions
from scipy.stats import chi2
from scipy.stats.distributions import chi2
from scipy.special import erf
from scipy.special import wofz

# Plotting library
import matplotlib.pyplot as plt

# Random number generation
import random as rd

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

# Single Voigt profile used in fitting
def voigt(x, amplitude, center, sigma, gamma):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

# Triple Voigt sum function used to model spectra with three components
def triple_voigt(x, *params):
    assert len(params) == 12, "Expected 12 parameters: 3 sets of (amp, center, sigma, gamma)"
    return sum(voigt(x, *params[i:i+4]) for i in range(0, 12, 4))

# Double Voigt sum function used to model spectra with two components
def double_voigt(x, *params):
    assert len(params) == 8, "Expected 8 parameters: 2 sets of (amp, center, sigma, gamma)"
    return voigt(x, *params[0:4]) + voigt(x, *params[4:8])

# === INPUT PARAMETERS ===

# Element identity for each energy range
element_map = {
    "70_130": "Si",
    "140_200": "Si",
    "260_300": "C",
    "500_550": "O"
}

# Select X-ray source: "Mg" or "Al"
element = "Mg"

# select annealing themperature
#base_path = f"GR_SiC_250_C/Al300w/"  # 250°C annealing
base_path = f"GR_SiC_450_C/Al300w/" # 450°C annealing

# Data files to process
files = ["70_130.txt", "140_200.txt", "260_300.txt", "500_550.txt"]

# Corresponding titles (elements)
titles = ["Si", "Si", "C", "O"]

# Maximum number of Voigt components to test during fitting
max_voigts = 3

# fit manually and get width and center from database (about since can have shift)
# also remove extra Si in percentage
# also get width of void fit --> to see if gets narrower or wider to contaminations

# Mg: 2 for Si, 3 for carbon (see picture), 2 oxigen
# sp2 & SP3+rest plot for annealing (percentages) to show trend what is going on


# === END INPUT PARAMETERS ===


# === Parameter definitions for lambda and sigma based on element and X-ray source ===

lambda_Mg = {
    "O": 12.8,
    "C": 16.0,
    "Si": 18.95
}

lambda_Al = {
    "O": 15.85,
    "C": 18.95,
    "Si": 21.2
}

sigma_Mg = {
    "O": 0.06354,
    "C": 0.02228,
    "Si": 0.01895
}

sigma_Al = {
    "O": 0.04005,
    "C": 0.01367,
    "Si": 0.01303
}

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# === Cropping and background subtraction for each data file ===

# Crop ranges as in goeie_shirley.py
crop_ranges = {
    "70_130": (96, 110),
    "140_200": (145, 160),
    "260_300": (280, 290),
    "500_550": (526, 540)
}

# Store cropped and background-subtracted data and x arrays
cropped_bgsub_data = {}
x_cropped_dict = {}

for file in files:
    label = file.replace(".txt", "")
    try:
        data = np.loadtxt(base_path + file, delimiter="\t")
    except FileNotFoundError:
        print(f"[Warning] Missing file: {base_path + file}")
        continue
    x = data[:, 0]
    y = data[:, 1]
    # Crop first
    start, end = crop_ranges[label]
    mask = (x >= start) & (x <= end)
    x_crop = x[mask]
    y_crop = y[mask]
    # Shirley background subtraction on cropped data
    bg = shirley_background(y_crop)
    y_corrected = y_crop - bg
    cropped_bgsub_data[label] = y_corrected
    x_cropped_dict[label] = x_crop

fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))

# Prepare data_list and cropped_data_list as in goeie_shirley.py
data_list = [
    ("70_130", x_cropped_dict["70_130"], cropped_bgsub_data["70_130"]),
    ("140_200", x_cropped_dict["140_200"], cropped_bgsub_data["140_200"]),
    ("260_300", x_cropped_dict["260_300"], cropped_bgsub_data["260_300"]),
    ("500_550", x_cropped_dict["500_550"], cropped_bgsub_data["500_550"])
]

cropped_data_list = data_list  # Already cropped and background-subtracted

# === Fit Voigt profiles and determine best fit ===

voigt_fits = []
ratio_dict = {}
for label, x_vals, y_vals in cropped_data_list:
    amp_guess = np.max(y_vals)
    center_guess = x_vals[np.argmax(y_vals)]
    width = (x_vals[-1] - x_vals[0]) / 5

    best_chi_sq = np.inf
    best_fit = None
    best_popt = None
    best_n = None

    for n_voigts in range(1, max_voigts + 1):
        # Generate initial parameters for Voigt components
        p0 = []
        offset = (x_vals[-1] - x_vals[0]) / (2 * n_voigts)
        for i in range(n_voigts):
            amp = amp_guess / n_voigts
            center = np.clip(center_guess + (i - n_voigts // 2) * offset, min(x_vals), max(x_vals))
            sigma = max(width, 0.1)
            gamma = max(width, 0.1)
            p0 += [amp, center, sigma, gamma]

        # Define parameter bounds for fitting
        lower_bounds = []
        upper_bounds = []
        for i in range(n_voigts):
            lower_bounds += [0, min(x_vals), 1e-6, 1e-6]
            upper_bounds += [np.inf, max(x_vals), max(x_vals) - min(x_vals), max(x_vals) - min(x_vals)]

        def general_voigt_sum(x, *params):
            return sum(voigt(x, *params[i:i+4]) for i in range(0, len(params), 4))

        try:
            popt, _ = curve_fit(general_voigt_sum, x_vals, y_vals, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=20000)
            fit_result = general_voigt_sum(x_vals, *popt)
            # Check for unreasonably large fits
            if np.max(fit_result) < 10 * np.max(y_vals):
                full_data = np.loadtxt(base_path + f"{label}.txt", delimiter="\t")
                x_full = full_data[:, 0]
                y_full = full_data[:, 1]
                sigma_i = np.interp(x_vals, x_full, np.sqrt(y_full))
                residuals = y_vals - fit_result
                dof = len(y_vals) - len(popt)
                # Calculate reduced chi-squared to evaluate fit quality
                chi_sq = np.sum(((residuals) / sigma_i) ** 2) / dof
                if chi_sq < best_chi_sq:
                    best_chi_sq = chi_sq
                    best_fit = fit_result
                    best_popt = popt
                    best_n = n_voigts
        except RuntimeError:
            # Fitting failed for this number of Voigt components; try next
            continue

    if best_popt is not None:
        popt = best_popt
        fit_result = best_fit
        chi_sq = best_chi_sq
        n_voigts = best_n
        voigt_fits.append((label, x_vals, fit_result, popt, chi_sq))
        areas = [np.trapezoid(voigt(x_vals, *popt[i*4:(i+1)*4]), x_vals) for i in range(n_voigts)]
        centers = [popt[i*4 + 1] for i in range(n_voigts)]
        total_area = np.trapezoid(general_voigt_sum(x_vals, *popt), x_vals)
        print(f"{label}:")
        print(f"  Total area under fit: {-total_area:.2f}")
        for i, (area, center) in enumerate(zip(areas, centers), 1):
            print(f"  Voigt {i}: area = {-area:.2f}, center = {center:.2f}")

        measured_element = element_map[label]
        lam = lambda_Al[measured_element] if element == "Al" else lambda_Mg[measured_element]
        sig = sigma_Al[measured_element] if element == "Al" else sigma_Mg[measured_element]
        ratio = -total_area / (lam * sig)
        print(f"  Ratio (-area)/(lambda * sigma) for {measured_element}: {ratio:.2f}")
        ratio_dict[label] = ratio
    else:
        print(f"Voigt fit failed for {label}")
        voigt_fits.append((label, x_vals, np.zeros_like(x_vals), None, None))


# === Calculate and print percentage contributions of each element ===

print("\nPercentage contribution of each element to the total ratio:")
total_ratio = 0
element_contributions = {}

for key, value in ratio_dict.items():
    element = element_map[key]
    # Only include the second Si (i.e., "140_200") in total
    if element == "Si" and key != "140_200":
        continue
    total_ratio += value
    element_contributions[element] = element_contributions.get(element, 0) + value

for element, value in element_contributions.items():
    percent = (value / total_ratio) * 100
    print(f"{element}: {percent:.2f}%")

plt.close()

fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))
colors = ['red', 'green', 'purple']
linestyles = [':', ':', ':']

for ax, (label, x_vals, y_vals), (_, _, fit_vals, popt, chi_sq), title in zip(axs2.flat, cropped_data_list, voigt_fits, titles):
    ax.plot(x_vals, y_vals, label="Shirley reduced", color='blue')
    if popt is not None:
        if chi_sq is not None:
            legend_label = f"Total Voigt fit (χ² = {chi_sq:.2f})"
        else:
            legend_label = "Total Voigt fit (fit failed)"
        ax.plot(x_vals, fit_vals, '--', label=legend_label, color='orange')
        for i in range(len(popt) // 4):
            single_fit = voigt(x_vals, *popt[i*4:(i+1)*4])
            ax.plot(
                x_vals,
                single_fit,
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                label=f"Voigt {i+1}"
            )
    ax.set_title(title)
    ax.set_xlabel("Binding Energy")
    ax.set_ylabel("Intensity")
    ax.invert_xaxis()
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.suptitle("2×2 Grid of Background-Subtracted Data with Voigt Fits", y=1.03)
plt.show()