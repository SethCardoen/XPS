# makes fits for carbon with physical relevance and plots change after annealing



from scipy.optimize import curve_fit
from scipy.special import wofz

#element_to_plot = "Al"
element_to_plot = "Mg"

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

    # --- Configurable upper bounds for peak widths ---
    upper_green = 0.5   # for sp2 graphene
    upper_red = 0.3     # for sp3 C=C
    upper_purple = 0.5  # for hydro carbons
    upper_brown = 0.5   # for additional hydrocarbons

    # Initial guesses for 4 peaks: amp, center, sigma, gamma
    p0 = [
        max(y), 284.5, 0.25, 0.25,
        max(y)/2, 284.8, 0.2, 0.2,
        max(y)/3, 285.5, 0.1, 0.1,
        max(y)/4, 286.0, 0.1, 0.1,
    ]

    bounds_lower = [
        0, 284.3, 0, 0,
        0, 284.65, 0, 0,
        0, 285.4, 0, 0,
        0, 285.9, 0, 0,
    ]

    bounds_upper = [
        np.inf, 284.5, upper_green, upper_green,
        np.inf, 284.85, upper_red, upper_red,
        np.inf, 286.5, upper_purple, upper_purple,
        np.inf, 286.5, upper_brown, upper_brown,
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
        for i in range(0, 16, 4):
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

plot_data = [(label, y_vals) for label, y_vals in carbon_data.items() if label.startswith(element_to_plot)]
fig, axs = plt.subplots(1, len(plot_data), figsize=(5 * len(plot_data), 5))
if len(plot_data) == 1:
    axs = [axs]
fig.suptitle("Carbon Voigt Fits", fontsize=16, y=0.98)

for i, (label, y_vals) in enumerate(plot_data):
    ax = axs[i]
    ax.plot(carbon_x_vals, y_vals, label=label)
    if fit_results[label] is not None:
        x_fit = np.linspace(carbon_x_vals.min(), carbon_x_vals.max(), 2000)
        y_fit = multi_voigt(x_fit, *fit_results[label])
        ax.plot(x_fit, y_fit, '--', label="Voigt Fit")
        params = fit_results[label]
        for j in range(0, 16, 4):
            y_component = voigt(x_fit, *params[j:j+4])
            if j == 0:
                label_component = "sp2 graphene"
            elif j == 4:
                label_component = "sp3 C=C"
            elif j == 8:
                label_component = "hydro carbons"
            elif j == 12:
                label_component = "hydro carbons"
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

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --- Summary plots of component percentage vs. temperature for Al and Mg using actual fits ---

if element_to_plot == "Al":
    al_percentages = {}

    for label, params in fit_results.items():
        if params is None:
            continue

        element, temp_label = label.split("_")
        if element != "Al":
            continue
        x_dense = np.linspace(carbon_x_vals.min(), carbon_x_vals.max(), 2000)
        total_y = multi_voigt(x_dense, *params)
        total_area = trapezoid(total_y, x_dense)

        percentages = []
        for i in range(0, len(params), 4):
            y_comp = voigt(x_dense, *params[i:i+4])
            area = trapezoid(y_comp, x_dense)
            percent = 100 * area / total_area
            percentages.append(percent)

        al_percentages[temp_label] = percentages

    temperatures = [0, 250, 450]
    labels = ["sp2", "sp3", "hydrocarbons", "extra"]

    component_colors = {
        "sp2": 'tab:green',
        "sp3": 'tab:red',
        "hydro carbons/carbon compounds": 'tab:purple',
        "extra": 'tab:brown'
    }

    fig, ax_al = plt.subplots(1, 1, figsize=(7, 6))
    summary_component_colors = [
        component_colors["sp2"],
        component_colors["sp3"],
        component_colors["hydro carbons/carbon compounds"],
        component_colors["extra"]
    ]

    for i, label in enumerate(labels):
        ax_al.plot(
            temperatures,
            [al_percentages[f"{t}°C"][i] if f"{t}°C" in al_percentages else None for t in temperatures],
            label=label, marker='o', color=summary_component_colors[i]
        )

    ax_al.set_title("Al: Percentage of Total Area vs Temperature")
    ax_al.set_xlabel("Temperature (°C)")
    ax_al.set_ylabel("Percentage of Total Area (%)")
    ax_al.legend()
    ax_al.grid(True)

    fig.tight_layout()
    plt.show()

elif element_to_plot == "Mg":
    mg_percentages = {}

    for label, params in fit_results.items():
        if params is None:
            continue

        element, temp_label = label.split("_")
        if element != "Mg":
            continue
        x_dense = np.linspace(carbon_x_vals.min(), carbon_x_vals.max(), 2000)
        total_y = multi_voigt(x_dense, *params)
        total_area = trapezoid(total_y, x_dense)

        percentages = []
        for i in range(0, len(params), 4):
            y_comp = voigt(x_dense, *params[i:i+4])
            area = trapezoid(y_comp, x_dense)
            percent = 100 * area / total_area
            percentages.append(percent)

        mg_percentages[temp_label] = percentages

    temperatures = [0, 250, 450]
    labels = ["sp2", "sp3", "hydrocarbons", "extra"]

    component_colors = {
        "sp2": 'tab:green',
        "sp3": 'tab:red',
        "hydro carbons/carbon compounds": 'tab:purple',
        "extra": 'tab:brown'
    }

    fig, ax_mg = plt.subplots(1, 1, figsize=(7, 6))
    fig.suptitle("Carbon Peak Fit Evolution for Mg", fontsize=16, y=0.98)
    summary_component_colors = [
        component_colors["sp2"],
        component_colors["sp3"],
        component_colors["hydro carbons/carbon compounds"],
        component_colors["extra"]
    ]

    for i, label in enumerate(labels):
        ax_mg.plot(
            temperatures,
            [mg_percentages[f"{t}°C"][i] if f"{t}°C" in mg_percentages else None for t in temperatures],
            label=label, marker='o', color=summary_component_colors[i]
        )

    ax_mg.set_title("Mg: Percentage of Total Area vs Temperature")
    ax_mg.set_xlabel("Temperature (°C)")
    ax_mg.set_ylabel("Percentage of Total Area (%)")
    ax_mg.legend()
    ax_mg.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
