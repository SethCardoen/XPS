# makes fits for carbon with physical relevance and plots change after annealing



from scipy.optimize import curve_fit
from scipy.special import wofz

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

carbon_data = {}
carbon_x_vals = None
crop_ranges = {
    "70_130": (96, 110),
    "140_200": (145, 160),
    "260_300": (280, 290),
    "500_550": (526, 540)
}
carbon_file = "260_300.txt"
crop_start, crop_end = crop_ranges["260_300"]

for source_element in ["Al", "Mg"]:
    for label, folder in annealing_dirs.items():
        path = f"{folder}/{source_element}300w/{carbon_file}"
        try:
            data = np.loadtxt(path, delimiter="\t")
            x = data[:, 0]
            y = data[:, 1]
            mask = (x >= crop_start) & (x <= crop_end)
            x_crop = x[mask]
            y_crop = y[mask]
            bg = shirley_background(x_crop, y_crop)
            y_crop = y_crop - bg
            if carbon_x_vals is None or len(x_crop) < len(carbon_x_vals):
                carbon_x_vals = x_crop
            carbon_data[f"{source_element}_{label}"] = y_crop[:len(carbon_x_vals)]
            if source_element == "Al":
                y_max_al = max(y_crop[:len(carbon_x_vals)]) if 'y_max_al' not in locals() else max(y_max_al, max(y_crop[:len(carbon_x_vals)]))
            elif source_element == "Mg":
                y_max_mg = max(y_crop[:len(carbon_x_vals)]) if 'y_max_mg' not in locals() else max(y_max_mg, max(y_crop[:len(carbon_x_vals)]))
        except Exception as e:
            print(f"[Warning] Failed to read {path}: {e}")

fit_results = {}

for label, y_vals in carbon_data.items():
    x = carbon_x_vals
    y = y_vals

    # Initial guesses for 4 peaks: amp, center, sigma, gamma
    p0 = [
        max(y), 284.4, 0.1, 0.1,
        max(y)/2, 284.7, 0.1, 0.1,
        max(y)/3, 285.5, 0.1, 0.1,
        max(y)/4, 286.0, 0.1, 0.1,
    ]

    bounds_lower = [
        0, 284.2, 0, 0,
        0, 284.55, 0, 0,
        0, 285.5, 0, 0,
        0, 286, 0, 0,
    ]
    bounds_upper = [
        np.inf, 284.55, np.inf, np.inf,
        np.inf, 284.9, np.inf, np.inf,
        np.inf, 286.0, np.inf, np.inf,
        np.inf, 286.5, np.inf, np.inf,
    ]

    try:
        popt, _ = curve_fit(multi_voigt, x, y, p0=p0, bounds=(bounds_lower, bounds_upper))
        fit_results[label] = popt
        from scipy.integrate import trapezoid

        x_dense = np.linspace(x.min(), x.max(), 2000)
        y_total = multi_voigt(x_dense, *popt)
        total_area = trapezoid(y_total, x_dense)

        print(f"\nVoigt fit for {label}")
        print(f"{'Component':<10} {'Center':>10} {'Bounds':>20} {'FWHM':>10} {'Area':>15} {'% of Total':>15}")
        for i in range(0, len(popt), 4):
            amp, cen, sigma, gamma = popt[i:i+4]
            fwhm = 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2 * gamma)**2 + (2.3548 * sigma)**2)
            y_comp = voigt(x_dense, amp, cen, sigma, gamma)
            area = trapezoid(y_comp, x_dense)
            percent = 100 * area / total_area
            name = f"Peak {i//4 + 1}"
            lb = bounds_lower[i+1]
            ub = bounds_upper[i+1]
            print(f"{name:<10} {cen:10.3f} [{lb:.2f}, {ub:.2f}] {fwhm:10.3f} {area:15.3f} {percent:15.2f}%")
    except RuntimeError:
        print(f"[Warning] Fit failed for {label}")
        fit_results[label] = None

y_max_dict = {"Al": y_max_al * 1.1, "Mg": y_max_mg * 1.1}

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

for i, (label, y_vals) in enumerate(carbon_data.items()):
    ax = axs[i]
    ax.plot(carbon_x_vals, y_vals, label=label)
    if fit_results[label] is not None:
        x_fit = np.linspace(carbon_x_vals.min(), carbon_x_vals.max(), 2000)
        y_fit = multi_voigt(x_fit, *fit_results[label])
        ax.plot(x_fit, y_fit, '--', label="Voigt Fit")
        params = fit_results[label]
        for j in range(0, len(params), 4):
            y_component = voigt(x_fit, *params[j:j+4])
            if j == 0:
                label_component = "sp2"
            elif j == 4:
                label_component = "sp3"
            elif j == 8:
                label_component = "hydro carbons/carbon compounds"
            elif j == 12:
                label_component = "hydro carbons/carbon compounds"
            else:
                label_component = None
            ax.plot(x_fit, y_component, linestyle='-', alpha=0.7, label=label_component)
    # Add vertical lines before setting the title
    #ax.axvline(284.55, linestyle=':', color='lightgray')
    #ax.axvline(285.0, linestyle=':', color='lightgray')
    ax.set_title(label)
    ax.legend(loc="upper right")
    ax.set_xlabel("Binding Energy (eV)")
    ax.set_ylabel("Intensity (a.u.)")
    # ax.grid(True)
    element_key = label.split("_")[0]
    ax.set_ylim(0, y_max_dict[element_key])
    ax.invert_xaxis()

fig.tight_layout()
plt.show()

# --- Summary plots of component percentage vs. temperature for Al and Mg ---
import matplotlib.pyplot as plt

al_percentages = {
    "0°C": [36.65, 32.57, 22.30, 8.47],
    "250°C": [48.87, 34.53, 2.71, 13.88],
    "450°C": [48.87, 34.53, 2.71, 13.88]
}

mg_percentages = {
    "0°C": [37.04, 39.65, 9.64, 13.67],
    "250°C": [46.08, 33.73, 8.88, 11.30],
    "450°C": [46.08, 33.73, 8.88, 11.30]
}

temperatures = [0, 250, 450]
labels = ["sp2", "sp3", "hydrocarbons (1)", "hydrocarbons (2)"]

component_colors = {
    "sp2": 'tab:green',
    "sp3": 'tab:red',
    "hydro carbons/carbon compounds": 'tab:purple'
}

fig, (ax_al, ax_mg) = plt.subplots(1, 2, figsize=(14, 6))

# Update color list to match legend colors from the plot
summary_component_colors = [component_colors["sp2"], component_colors["sp3"], 'tab:purple', 'tab:brown']

for i, label in enumerate(labels):
    ax_al.plot(
        temperatures,
        [al_percentages[f"{t}°C"][i] for t in temperatures],
        label=label, marker='o', color=summary_component_colors[i]
    )
    ax_mg.plot(
        temperatures,
        [mg_percentages[f"{t}°C"][i] for t in temperatures],
        label=label, marker='o', color=summary_component_colors[i]
    )

ax_al.set_title("Al: Percentage of Total Area vs Temperature")
ax_al.set_xlabel("Temperature (°C)")
ax_al.set_ylabel("Percentage of Total Area (%)")
ax_al.legend()
ax_al.grid(True)

ax_mg.set_title("Mg: Percentage of Total Area vs Temperature")
ax_mg.set_xlabel("Temperature (°C)")
ax_mg.set_ylabel("Percentage of Total Area (%)")
ax_mg.legend()
ax_mg.grid(True)

fig.tight_layout()
plt.show()
