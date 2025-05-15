from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.integrate import trapezoid

def voigt(x, amp, cen, sigma, gamma):
    z = ((x - cen) + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def multi_voigt(x, *params):
    return sum(voigt(x, *params[i:i+4]) for i in range(0, len(params), 4))

def shirley_background(x, y, tol=1e-5, max_iter=5000):
    y = np.array(y)
    x = np.array(x)
    y0, y1 = y[0], y[-1]
    bg = np.linspace(y0, y1, len(y))  # Initial guess: linear interpolation between ends

    for _ in range(max_iter):
        integral = np.cumsum(y - bg)
        integral -= integral[0]
        integral = y0 + (y1 - y0) * (integral / integral[-1])
        if np.max(np.abs(bg - integral)) < tol:
            break
        bg = integral

    return bg
import numpy as np
import matplotlib.pyplot as plt

annealing_dirs = {
    "0°C": "GR_SiC_0_C",
    "250°C": "GR_SiC_250_C",
    "450°C": "GR_SiC_450_C"
}

#element = "Al"
element = "Mg"



silicon_data = {}
silicon_x_vals = None
crop_ranges = {
    "70_130": (96, 110),
    "140_200": (145, 160),
    "260_300": (280, 290),
    "500_550": (526, 540)
}
silicon_file = "500_550.txt"
crop_start, crop_end = crop_ranges["500_550"]

selected_label = "0°C"  # or "250°C", "450°C"

for label, folder in annealing_dirs.items():
    path = f"{folder}/{element}300w/{silicon_file}"
    try:
        data = np.loadtxt(path, delimiter="\t")
        x = data[:, 0]
        y = data[:, 1]
        mask = (x >= crop_start) & (x <= crop_end)
        x_crop = x[mask]
        y_crop = y[mask]
        bg = shirley_background(x_crop, y_crop)
        y_crop = y_crop - bg
        if silicon_x_vals is None or len(x_crop) < len(silicon_x_vals):
            silicon_x_vals = x_crop
        silicon_data[label] = y_crop[:len(silicon_x_vals)]
        y_max_al = max(y_crop[:len(silicon_x_vals)]) if 'y_max_al' not in locals() else max(y_max_al, max(y_crop[:len(silicon_x_vals)]))
    except Exception as e:
        print(f"[Warning] Failed to read {path}: {e}")

fit_results = {}

for label, y_vals in silicon_data.items():
    x = silicon_x_vals
    y = y_vals

    # Initial guesses for 4 peaks: amp, center, sigma, gamma (O 1s core level info)
    p0 = [
        max(y), 531.3, 0.2, 0.2,   # C-O compounds
        max(y)/2, 532.5, 0.2, 0.2, # hydrocarbons
        max(y)/2.5, 533.1, 0.2, 0.2, # Si oxide
        max(y)/3, 534.5, 0.2, 0.2   # adsorbed water / organics
    ]

    bounds_lower = [
        0, 531.1, 0, 0,
        0, 532.0, 0, 0,
        0, 532.8, 0, 0,
        0, 534.0, 0, 0
    ]

    primary_limit = 0.5
    secondary_limit = 0.3
    tird_limit = 0.3

    bounds_upper = [
        np.inf, 531.6, primary_limit, primary_limit,
        np.inf, 533.0, secondary_limit, secondary_limit,
        np.inf, 533.4, tird_limit, tird_limit,
        np.inf, 535.5, tird_limit, tird_limit,
    ]

    try:
        popt, _ = curve_fit(multi_voigt, x, y, p0=p0, bounds=(bounds_lower, bounds_upper))
        fit_results[label] = popt
        # Compute area under each Voigt peak
        x_dense = np.linspace(min(x), max(x), 1000)
        areas = []
        for j in range(0, len(popt), 4):
            y_component = voigt(x_dense, *popt[j:j+4])
            area = trapezoid(y_component, x_dense)
            areas.append(area)
        percentages = [100 * a / sum(areas) for a in areas]
        print(f"\nVoigt Fit Summary for {label}")
        print(f"{'Peak':<5} {'Center (eV)':>15} {'Area':>15} {'% of Total':>15}")
        for idx, (center, area, pct) in enumerate(zip(popt[1::4], areas, percentages), start=1):
            print(f"{idx:<5} {center:15.4f} {area:15.4f} {pct:15.2f}")
    except RuntimeError:
        print(f"[Warning] Fit failed for {label}")
        fit_results[label] = None

percent_results = {}
for label, popt in fit_results.items():
    if popt is not None:
        x_dense = np.linspace(min(silicon_x_vals), max(silicon_x_vals), 1000)
        areas = []
        for j in range(0, len(popt), 4):
            y_component = voigt(x_dense, *popt[j:j+4])
            area = trapezoid(y_component, x_dense)
            areas.append(area)
        percent_results[label] = [100 * a / sum(areas) for a in areas]

y_max_dict = {label: y_max_al * 1.1 for label in silicon_data}
# Based on silicon intensity

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs = axs.flatten()

for i, (label, y_vals) in enumerate(silicon_data.items()):
    ax = axs[i]
    ax.plot(silicon_x_vals, y_vals, label=label)
    if fit_results[label] is not None:
        x_fit = silicon_x_vals
        y_fit = multi_voigt(x_fit, *fit_results[label])
        ax.plot(x_fit, y_fit, '--', label="Voigt Fit")
        total_area = trapezoid(y_fit, x_fit)
        print(f"Total area for {label}: {total_area}")
        params = fit_results[label]
        x_dense = np.linspace(x_fit.min(), x_fit.max(), 2000)
        for j in range(0, len(params), 4):
            y_component = voigt(x_dense, *params[j:j+4])
            if j == 0:
                label_component = "C-O compounds"
            elif j == 4:
                label_component = "Hydrocarbons"
            elif j == 8:
                label_component = "Si oxides"
            elif j == 12:
                label_component = "Adsorbed water / organics"
            else:
                label_component = None
            ax.plot(x_dense, y_component, linestyle='-', alpha=0.7, label=label_component)
    # Add vertical lines before setting the title
    #ax.axvline(99.0, linestyle=':', color='lightgray')
    #ax.axvline(102.0, linestyle=':', color='lightgray')
    #ax.axvline(104.0, linestyle=':', color='lightgray')
    ax.set_title(f"{element} {label}")
    ax.legend(loc="upper right")
    ax.set_xlabel("Binding Energy (eV)")
    ax.set_ylabel("Intensity (a.u.)")
    # ax.grid(True)
    ax.set_ylim(0, y_max_dict[label])
    ax.invert_xaxis()

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle("Oxygen 1s Core Level Fits", fontsize=16, y=.98)
plt.show()

# Plot percentages vs. temperature
fig_pct, ax_pct = plt.subplots(figsize=(8, 5))

temperatures = ["0°C", "250°C", "450°C"]
components = ["C-O compounds", "Hydrocarbons", "Si oxides", "Adsorbed water / organics"]
colors = ["tab:green", "tab:red", "tab:purple", "tab:brown"]

for i, comp in enumerate(components):
    values = [percent_results[temp][i] if temp in percent_results else None for temp in temperatures]
    ax_pct.plot([0, 250, 450], values, marker='o', label=comp, color=colors[i])

ax_pct.set_title("Oxygen Peak Component Percentages vs Temperature")
ax_pct.set_xlabel("Temperature (°C)")
ax_pct.set_ylabel("Percentage of Total Area (%)")
ax_pct.legend()
ax_pct.grid(True)
fig_pct.tight_layout()
plt.show()
