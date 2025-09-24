import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, savgol_filter, peak_widths
from sklearn.metrics import mean_squared_error
import time

""" --- Functions --- """

def detect_and_trim_stable_tail(t, y, window=10, std_threshold=0.05):
    """
    Detects a stable area at the end of the measurement results and cuts it off.
    Parameters:
    - t: Time vector
    - y: Ultrasound velocity values
    - window: Window size within which the standard deviation is calculated
    - std_threshold: Limit value of the standard deviation, values below which are interpreted as a steady state
    Returns:
    - Abbreviated t and y tables with the steady state removed from the end.
    """

    for i in range(len(y) - window):
        std = np.std(y[i:i + window])
        if std < std_threshold:
            return t[:i], y[:i]
    return t, y

def smooth_signal(signal, window):
    """
    Smooths the signal with a moving average while preserving the original length.
    Parametrit:
     - signal: Observed ultrasonic velocity values
     - window: Window size for calculating the average 
         -- Determines how many consecutive values are taken into account when calculating the average
    Returns:
     - A smoothed signal with the original length.
    """

    if window < 2:
        return signal
    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode='edge')
    smoothed = np.convolve(padded, np.ones(window) / window, mode='valid')
    return smoothed[:len(signal)]

def estimate_initial_params(t, y, n_components=7, model_type="logistic"):
    """
    Estimate initial values for parameter fitting.
    Finds derivative peaks and uses them as component midpoints.
    Parameters: t = time, y = measured ultrasonic velocity, n_components = number of components.
    Returns a parameter list in the form [A, k, t0, (nu), C] * n_components.
    """

    dy = savgol_filter(y, window_length=31, polyorder=3, deriv=1, delta=(t[1] - t[0]))

    # Peak searching
    peaks, _ = find_peaks(dy, prominence=0.55)
    #print(f"        Found {len(peaks)} peaks.")

    # LCalculate the prominences and arrange the peaks in order of importance
    prominences = peak_prominences(dy, peaks)[0]
    sorted_indices = np.argsort(prominences)[::-1]
    sorted_peaks = peaks[sorted_indices]

    # Select n_components most significant peaks
    selected_peaks = sorted_peaks[:n_components]
    print(f"        The selected peaks: {selected_peaks}")

    # Initial values of components according to selected peaks
    params = []
    for i in selected_peaks:
        t0 = t[i]
        A = (np.max(y) - np.min(y)) / n_components
        k = 1.0
        C = 0.0
        if model_type == "richards":
            nu = 1.0
            params += [A, k, t0, nu, C]
        else:
            params += [A, k, t0, C]
    return params

def logistic_sum(t, *params):
    """
    Calculates the sum of multiple logistic functions.
    The parameters A, k, t0, and C for each component are listed in [A1, k1, t01, C1, A2, ...].
    """

    n = len(params) // 4
    result = np.zeros_like(t)
    for i in range(n):
        A, k, t0, C = params[4 * i:4 * i + 4]
        exp_arg = -k * (t - t0)
        exp_arg = np.clip(exp_arg, -100, 100)
        result += A / (1 + np.exp(exp_arg)) + C
    return np.nan_to_num(result)

def logistic_sum_derivative(t, *params):
    """
    Calculates the total derivative for the sum of logistic functions.
    Parameters from the same list [A1, k1, t01, C1, A2, ...].
    """

    n = len(params) // 4
    result = np.zeros_like(t)
    for i in range(n):
        A, k, t0, _ = params[4 * i:4 * i + 4]
        exp_arg = -k * (t - t0)
        exp_arg = np.clip(exp_arg, -100, 100)
        exp_term = np.exp(exp_arg)
        denom = (1 + exp_term) ** 2
        result += A * k * exp_term / denom
    return np.nan_to_num(result)

def logistic_component_derivatives(t, *params):
    """
    Calculates the derivative of each component separately.
    Returns a list of component-specific derivatives.
    """

    n = len(params) // 4
    components = []
    for i in range(n):
        A, k, t0, _ = params[4 * i:4 * i + 4]
        exp_arg = -k * (t - t0)
        exp_arg = np.clip(exp_arg, -100, 100)
        exp_term = np.exp(exp_arg)
        denom = (1 + exp_term) ** 2
        component = A * k * exp_term / denom
        components.append(np.nan_to_num(component))
    return components

def gompertz_sum(t, *params):
    """
    Calculates the sum of multiple Gompertz functions.
    The parameters A, k, t0, and C for each component are listed in [A1, k1, t01, C1, A2, ...].
    """

    n = len(params) // 4
    result = np.zeros_like(t)
    for i in range(n):
        A, k, t0, C = params[4 * i:4 * i + 4]
        inner_exp = -k * (t - t0)
        inner_exp = np.clip(inner_exp, -500, 500)
        exp_arg = -np.exp(inner_exp)
        exp_arg = np.clip(exp_arg, -500, 500)
        result += A * np.exp(exp_arg) + C
    return np.nan_to_num(result)

def gompertz_sum_derivative(t, *params):
    """
    Calculates the total derivative for the sum of Gompertz functions.
    Parameters from the same list [A1, k1, t01, C1, A2, ...].
    """

    n = len(params) // 4
    result = np.zeros_like(t)
    for i in range(n):
        A, k, t0, _ = params[4 * i:4 * i + 4]
        inner_exp = -k * (t - t0)
        inner_exp = np.clip(inner_exp, -500, 500)
        exp_inner = np.exp(inner_exp)
        result += A * k * exp_inner * np.exp(-exp_inner)
    return np.nan_to_num(result)

def gompertz_component_derivatives(t, *params):
    """
    Calculates the derivative of each component separately.
    Returns a list of component-specific derivatives.
    """

    n = len(params) // 4
    components = []
    for i in range(n):
        A, k, t0, _ = params[4 * i:4 * i + 4]
        inner_exp = -k * (t - t0)
        inner_exp = np.clip(inner_exp, -500, 500)
        exp_inner = np.exp(inner_exp)
        deriv = A * k * exp_inner * np.exp(-exp_inner)
        components.append(np.nan_to_num(deriv))
    return components

def richards_sum(t, *params):
    """
    Calculates the sum of multiple Richards functions.

    f(t) = A / (1 + ν * exp(-k * (t - t₀)))^(1/ν) + C

    The parameters A, k, t0, nu (v), and C for each component are listed in [A1, k1, t01, nu1, C1, A2, ...].
    """

    n = len(params) // 5
    result = np.zeros_like(t)
    for i in range(n):
        A, k, t0, nu, C = params[5 * i:5 * i + 5]
        exp_arg = -k *(t - t0) # ennen exp_arg = -k * nu * (t - t0)
        exp_arg = np.clip(exp_arg, -100, 100)
        exp_term = np.exp(exp_arg)
        exp_term = np.clip(exp_term, 0, 1e20)  # tiukempi raja
        denom = (1 + nu * exp_term) ** (1 / nu) # kerroin nu puuttui
        denom = np.clip(denom, 1e-10, 1e100)
        result += A / denom + C
    return np.nan_to_num(result)

def richards_sum_derivative(t, *params):
    """
    Calculates the total derivative for the sum of Richards functions.

    f'(t) = A * k * exp(-k * (t - t₀)) / (1 + ν * exp(-k * (t - t₀)))^(1/ν + 1)

    Parameters from the same list [A1, k1, t01, nu1, C1, A2, ...].
    """

    n = len(params) // 5
    result = np.zeros_like(t)
    for i in range(n):
        A, k, t0, nu, _ = params[5 * i:5 * i + 5]
        exp_arg = -k * (t - t0) # ennen exp_arg = -k * nu * (t - t0)
        exp_arg = np.clip(exp_arg, -100, 100)
        exp_term = np.exp(exp_arg)
        exp_term = np.clip(exp_term, 0, 1e20)  # tiukempi raja
        denom_base = 1 + nu * exp_term # kerroin nu puuttui
        denom_base = np.clip(denom_base, 1e-10, 1e100)
        denom_deriv = denom_base ** ((1 / nu) + 1)
        denom_deriv = np.clip(denom_deriv, 1e-10, 1e100)
        term = A * k * exp_term / denom_deriv
        result += np.nan_to_num(term)
    return result

def richards_component_derivatives(t, *params):
    """
    Calculates the derivative of each component separately.

    f'(t) = A * k * exp(-k * (t - t₀)) / (1 + ν * exp(-k * (t - t₀)))^(1/ν + 1)

    Returns a list of component-specific derivatives.
    """

    n = len(params) // 5
    components = []
    for i in range(n):
        A, k, t0, nu, _ = params[5 * i:5 * i + 5]
        exp_arg = -k * (t - t0) # ennen exp_arg = -k * nu * (t - t0)
        exp_arg = np.clip(exp_arg, -100, 100)
        exp_term = np.exp(exp_arg)
        exp_term = np.clip(exp_term, 0, 1e20)
        denom = (1 + nu * exp_term) ** ((1 / nu) + 1) # kerroin nu puuttui
        denom = np.clip(denom, 1e-10, 1e100)
        term = A * k * exp_term / denom
        components.append(np.nan_to_num(term))
    return components

def compute_bic(y, y_fit, p):
    """
    Calculates the BIC for the selected model. Estimates the ln-likelihood with a normal error distribution assumption
    for the given model and observation data, in which case no value is calculated for the standard deviation.

    Parameters:
    y = ultrasonic velocity values, y_fit = model predictions for sound velocity values, p = number of parameters

    Assume that the residuals (y - y_fit) are independent and normally distributed.

    BIC is calculated using the formula:

    BIC = ln(n)p + n * ln(sum((y - y_fit)²/n))
    """

    resid = y - y_fit
    n = len(y)
    mse = np.mean(resid ** 2)
    bic = np.log(n) * p + n * np.log(mse)
    return bic

def compute_aic(y, y_fit, p):
    """
    Calculates the AIC for the selected model. Estimates the ln-likelihood with the assumption of a normal error distribution
    for the given model and observation data, in which case no value is calculated for the standard deviation.

    Parameters:
    y = ultrasonic velocity values, y_fit = model predictions for sound velocity values, p = number of parameters

    Assume that the residuals (y - y_fit) are independent and normally distributed.

    AIC is calculated using the formula:

    AIC = 2p + n * ln(sum((y - y_fit)²/n))
    """

    resid = y - y_fit
    n = len(y)
    mse = np.mean(resid ** 2)
    aic = 2 * p + n * np.log(mse)
    return aic

def adjusted_r2(y_true, y_pred, p):
    """
    Calculates the adjusted R-squared value for a given set of true values,
    predicted values, and the number of predictors in the model. Adjusted
    R-squared takes into account the number of predictors to provide a
    better evaluation metric for models with different numbers of features.

    The formula used for adjusted R-squared is:
        R²_adj = 1 - [(SSE_res / (n - p)) / (SSE_tot / (n - 1))]
    Where SSE_res is the sum of squared residuals, SSE_tot is the total sum
    of squares, n is the number of observations, and p is the number of
    predictors.
    """

    n = len(y_true)
    sse_res = np.sum((y_true - y_pred) ** 2)
    sse_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2_adj = 1 - (sse_res / (n - p)) / (sse_tot / (n - 1))
    return r2_adj

def fit_with_n_components(t: object, y: object, n_components: object, model_type: object = 'logistic') -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    """
    Fits with the given model type and number of components.
    Returns the fit values, optimized parameters, and MSE values for the data and derivative.
    """

    initial_guess = estimate_initial_params(t, y, n_components=n, model_type=model_type)

    # Parametrien raja-arvot
    A_bounds = (0, 2 * np.max(y))
    k_bounds = (0.01, 3)
    t0_bounds = (np.min(t), np.max(t))
    nu_bounds = (0.1, 10)
    C_bounds = (-np.max(y), np.max(y))
    lower_bounds = []
    upper_bounds = []
    for _ in range(n_components):
        if model_type == 'richards':
            lower_bounds.extend([A_bounds[0], k_bounds[0], t0_bounds[0], nu_bounds[0], C_bounds[0]])
            upper_bounds.extend([A_bounds[1], k_bounds[1], t0_bounds[1], nu_bounds[1], C_bounds[1]])
        else:
            lower_bounds.extend([A_bounds[0], k_bounds[0], t0_bounds[0], C_bounds[0]])
            upper_bounds.extend([A_bounds[1], k_bounds[1], t0_bounds[1], C_bounds[1]])
    bounds = (tuple(lower_bounds), tuple(upper_bounds))
    initial_guess = np.clip(initial_guess, lower_bounds, upper_bounds)

    if model_type == 'logistic':
        model_func = logistic_sum
        model_deriv = logistic_sum_derivative
    elif model_type == 'gompertz':
        model_func = gompertz_sum
        model_deriv = gompertz_sum_derivative
    elif model_type == 'richards':
        model_func = richards_sum
        model_deriv = richards_sum_derivative
    else:
        raise ValueError("Tuntematon mallityyppi")

    try:
        popt, pcov = curve_fit(
            model_func,
            t,
            y,
            p0=initial_guess,
            bounds=bounds,
            maxfev=20000
        )

        y_fit = model_func(t, *popt)
        dy_model = model_deriv(t, *popt)
        dy_data = savgol_filter(y, window_length=31, polyorder=3, deriv=1, delta=(t[1] - t[0]))
        mse_data = mean_squared_error(y, y_fit)
        mse_deriv = mean_squared_error(dy_data, dy_model)

        return y_fit, popt, mse_data, mse_deriv
    except RuntimeError:
        return None, None, np.inf, np.inf

def get_derivatives_by_model(t, y, popt, model_type):
    """
    Calculates the model fit, derivative, and component derivatives.
    Parameters:
    - t: Time vector
    - y: Original data
    - popt: Fitted parameters
    - model_type: Model type ('logistic', 'gompertz', or 'richards')
    Returns:
    - y_fit: Fit produced by the model
    - dy_fit: Derivative of the model
    - component_derivs: Derivative of each component separately
    """

    model_funcs = {
        'logistic': (logistic_sum, logistic_sum_derivative, logistic_component_derivatives),
        'gompertz': (gompertz_sum, gompertz_sum_derivative, gompertz_component_derivatives),
        'richards': (richards_sum, richards_sum_derivative, richards_component_derivatives)
    }

    if model_type not in model_funcs:
        raise ValueError("Tuntematon mallityyppi")

    y_fit_func, dy_func, comp_func = model_funcs[model_type]
    y_fit = y_fit_func(t, *popt)
    dy_fit = dy_func(t, *popt)
    component_derivs = comp_func(t, *popt)

    return y_fit, dy_fit, component_derivs

def compute_peak_areas(signal, peaks, left_ips, right_ips):
    """
    Compute the area under each peak defined by `left_ips` and `right_ips` in the given signal.

    This function iterates over the indices of the peaks, and for each peak, it calculates the
    area under the curve within the boundaries defined by `left_ips` and `right_ips`. The area
    is calculated using the trapezoidal rule for numerical integration.
    """
    areas = []
    for i in range(len(peaks)):
        left = int(np.floor(left_ips[i]))
        right = int(np.ceil(right_ips[i]))
        x = np.arange(left, right)
        y = signal[left:right]
        area = np.trapezoid(y, dx=(t[1] - t[0]))
        areas.append(area)
    return np.array(areas)


""" --- Main ---"""


model_type = input("Should we use the 'logistic', 'gompertz' or 'richards' model? ").strip().lower()

# Read the file
df = pd.read_excel(
    r"C:Data.xlsx",
    sheet_name="UT",
    skiprows=8)

# Remove spaces between the beginning and end of column names
df.columns = df.columns.str.strip()

# Change the column to number format (replace commas with periods)
df["m/s"] = df["m/s"].astype(str).str.replace(" ", "").str.replace(",", ".").astype(float)
df["Minute"] = df["Minute"].astype(str).str.replace(" ", "").str.replace(",", ".").astype(float)

# Remove rows that do not contain velocity data
df = df.dropna(subset=["m/s"])

# Length of original data
len_data = len(df)

# Store valid data in time variable t and ultrasonic velocity variable y
t = df["Minute"].values
y = df["m/s"].values

# Ask if you want automatic end trimming
use_trimming = input("Do you want to use automatic end trimming? (y/n): ").strip().lower() == "y"

t_full, y_full = t.copy(), y.copy()

if use_trimming:
    t, y = detect_and_trim_stable_tail(t_full, y_full, window=10, std_threshold=0.001)
    print(f"Removed {len(t_full) - len(t)} points from the end.")
else:
    print("The entire time series is used.")

# Length of filtered data
len_data_new = len(t)

# Numerical derivative
dy = savgol_filter(y, window_length=31, polyorder=3, deriv=1, delta=(t[1] - t[0]))

# Automatic selection of component quantities using the BIC_der criterion. You can also use other criteria.
use_auto_n_components = input("Should we automatically search for the optimal number of components? (y/n): ").strip().lower() == "y"

if use_auto_n_components:
    best_bic = np.inf
    best_bic_der = np.inf
    best_aic = np.inf
    best_aic_der = np.inf
    best_mse_deriv = None
    best_popt = None
    best_n = None
    for n in range(2, 11):
        print(f"  Fitting with {n} components...")
        start = time.time()
        y_fit, popt, mse_data, mse_deriv = fit_with_n_components(t, y, n, model_type=model_type)
        duration = time.time() - start
        if popt is not None:
            if model_type == 'logistic':
                model_deriv = logistic_sum_derivative
            elif model_type == 'gompertz':
                model_deriv = gompertz_sum_derivative
            else:
                model_deriv = richards_sum_derivative
            y_fit_deriv = model_deriv(t, *popt)
            aic = compute_aic(y, y_fit, len(popt))
            bic = compute_bic(y, y_fit, len(popt))
            bic_der = compute_bic(dy, y_fit_deriv, len(popt))
            aic_der = compute_aic(dy, y_fit_deriv, len(popt))
            r2 = adjusted_r2(y, y_fit, len(popt))
            residuals = y - y_fit
            sse = np.sum(residuals ** 2)
            print(f"      ready in ({duration:.2f}s) | SSE = {sse:.0f} | AIC = {aic:.0f} | BIC = {bic:.0f} | AIC_der = {aic_der:.0f} | BIC_der = {bic_der:.0f} | MSE = {mse_data:.0f} | MSE_deriv = {mse_deriv:.2f} | R^2_adj = {r2:.5f} | p lkm = {len(popt)}")
            if bic_der < best_bic_der:
                best_bic_der = bic_der
                best_popt = popt
                best_n = n
                best_mse_deriv = mse_deriv
        else:
            print(" failed.")
    if best_popt is None:
        raise RuntimeError("The fitting failed with all component quantities.")
    print(f"\nOptimal number of components: {best_n} (MSED = {best_mse_deriv:.3f})")
    popt = best_popt
else:
    n = int(input("How many components are used for fitting? "))
    y_fit, popt, mse_data, mse_deriv = fit_with_n_components(t, y, n, model_type=model_type)
    if popt is None:
        raise RuntimeError("The fitting failed.")
    best_n = n
    best_mse_deriv = mse_deriv

y_fit, dy_fit, component_derivs = get_derivatives_by_model(t, y, popt, model_type)

# Print the points of the peaks of the derivative function of the fit
upv_peaks, _ = find_peaks(dy_fit)

# Calculate the prominences for all peaks found in the derivative of the fit
upv_prominences = peak_prominences(dy_fit, upv_peaks)[0]

# Arrange peaks according to prominence
upv_sorted_indices = np.argsort(upv_prominences)[::-1]
upv_sorted_peaks = upv_peaks[upv_sorted_indices]
upv_component_peaks = upv_sorted_peaks[:len(component_derivs)]

# Print the peaks in order of importance
print(f"        Acceleration peaks: {upv_component_peaks}")


""" Calorimeter data """

df2 = pd.read_excel(
    r"C:Data.xlsx",
    sheet_name="CM")
df2.columns = df2.columns.str.strip()
df2["Time\n[min]"] = df2["Time\n[min]"].astype(str).str.replace(" ", "").str.replace(",", ".").str.replace("\n", "").astype(float)
df2["Thermal Power\n[W/g]"] = df2["Thermal Power\n[W/g]"].astype(str).str.replace(" ", "").str.replace(",", ".").astype(float)

# Search for thermal power peaks
t2 = df2["Time\n[min]"][:len_data_new].values
y2 = df2["Thermal Power\n[W/g]"][:len_data_new].values
cm_peaks, _ = find_peaks(y2)

# Calculate the prominences for all peaks found in the calorimeter data.
cm_prominences = peak_prominences(y2, cm_peaks)[0]

# Arrange peaks according to prominence
cm_sorted_indices = np.argsort(cm_prominences)[::-1]
cm_sorted_peaks = cm_peaks[cm_sorted_indices]
cm_component_peaks = cm_sorted_peaks[:len(component_derivs)]

# Print thermal power peaks in order of importance
print(f"        Calorimetry peaks: {cm_component_peaks}")

# Half widths of peaks
widths_upv, _, left_ips_upv, right_ips_upv = peak_widths(dy_fit, upv_component_peaks, rel_height=0.5)
widths_cm, _, left_ips_cm, right_ips_cm = peak_widths(y2, cm_component_peaks, rel_height=0.5)

# Surface areas of peaks
areas_upv = compute_peak_areas(dy_fit, upv_component_peaks, left_ips_upv, right_ips_upv)
areas_cm = compute_peak_areas(y2, cm_component_peaks, left_ips_cm, right_ips_cm)

# Relative areas
total_area_upv = sum(areas_upv)
total_area_cm = sum(areas_cm)
rel_areas_upv = [a / total_area_upv for a in areas_upv]
rel_areas_cm = [a / total_area_cm for a in areas_cm]

# Recognition of matching peaks (time difference < 15 min)
coincident_peaks = []
for i, pu in enumerate(upv_component_peaks):
    for j, pc in enumerate(cm_component_peaks):
        if abs(t[pu] - t[pc]) < 15:
            coincident_peaks.append((pu, pc))

# Printing the results
print("Relative areas of UPV spikes:", np.round(rel_areas_upv, 3))
print("Widths of UPV spikes:", np.round(widths_upv, 0))
print("Relative areas of CM peaks:", np.round(rel_areas_cm, 3))
print("Widths of CM peaks:", np.round(widths_cm, 0))
print("Matching peaks (time difference < 15 min):")
for pu, pc in coincident_peaks:
    print(f"Time: {t[pu]} min, UPV area (relative): {rel_areas_upv[np.where(upv_component_peaks == pu)[0][0]]:.4f}, "
          f"CM area (relative): {rel_areas_cm[np.where(cm_component_peaks == pc)[0][0]]:.4f}")

""" Vicat """
df3 = pd.read_excel(
    r"C:Data.xlsx",
    sheet_name="V")
df3.columns = df3.columns.str.strip()
df3["Time [min]"] = df3["Time [min]"].astype(str).str.replace(" ", "").str.replace(",", ".").str.replace("\n", "").astype(float)
df3["Depth [mm]"] = df3["Depth [mm]"].astype(str).str.replace(" ", "").str.replace(",", ".")
df3["Depth [mm]"] = pd.to_numeric(df3["Depth [mm]"], errors='coerce')
df3 = df3.dropna(subset=["Depth [mm]"])

t3 = df3["Time [min]"].values
y3 = df3["Depth [mm]"].values

# Initial setting: first time when the value is 3-9 mm
initial_set_time = next((t for t, y in zip(t3, y3) if 3 <= y <= 9), None)

# Final set: first time when the value is < 0.5 mm
final_set_time = next((t for t, y in zip(t3, y3) if y < 0.5), None)

""" Graphs """

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. Graph: Filtered vs. original data vs. fit
axs[0, 0].plot(t, y, label="Filtered")
axs[0, 0].plot(df["Minute"], df["m/s"], label="Original", alpha=0.3)
axs[0, 0].plot(t, y_fit, label="Model fit")
axs[0, 0].set_title("Velocity Development and Modeling Over Time")
axs[0, 0].set_xlabel("Time [min]")
axs[0, 0].set_ylabel("Velocity [m/s]")
axs[0, 0].legend()
removed = len_data - len_data_new
axs[0, 0].text(0.05, 0.95, f"Removed points: {removed}\nMSE = {mean_squared_error(y, y_fit):.0f}", transform=axs[0, 0].transAxes,
               fontsize=10, verticalalignment='top')

# 2. Graph: Derivative of the fit vs. numerical derivative
axs[0, 1].plot(t, dy, label='Numerical 1st derivative', alpha=0.6)
axs[0, 1].plot(t, dy_fit, label='Derivative of the model', linewidth=2)
axs[0, 1].set_title("Acceleration Estimation: Numerical vs. Model Derivative")
axs[0, 1].set_ylabel("Acceleration [(m/s)/min]")
axs[0, 1].set_xlabel("Time [min]")
axs[0, 1].legend()
axs[0, 1].text(0.05, 0.90, f"MSED = {best_mse_deriv:.2f}",
               transform=axs[0, 1].transAxes, fontsize=10, verticalalignment='top')

# 3. Graph: Derivatives of the fit and components vs. numerical derivative
for i, deriv in enumerate(component_derivs):
    axs[1, 0].plot(t, deriv, linestyle='--', label=f'Component {i + 1}')
axs[1, 0].plot(t, dy_fit, color='black', linewidth=2, label='Total sum')
axs[1, 0].plot(t, dy, label='Numerical 1st derivative', alpha=0.6)
axs[1, 0].set_title("Acceleration Components, Sum Function, and Numerical Derivative")
axs[1, 0].set_xlabel("Time [min]")
axs[1, 0].set_yscale('symlog')
axs[1, 0].set_ylim(0, 1000)
axs[1, 0].set_ylabel("Acceleration [(m/s)/min]")
axs[1, 0].legend()
axs[1, 0].text(0.05, 0.95, f"{model_type} model", transform=axs[1, 0].transAxes, fontsize=10, verticalalignment='top')

# 4. Graph: Comparison of Ultrasonic Acceleration, Thermal Power and Vicat Penetration

# Two vertical axes
ax1 = axs[1, 1]
ax2 = ax1.twinx()

# Data on the left axis (derivative of the model's sum function)
ax1.plot(t, dy_fit, color='blue', label='Fit')
ax1.set_ylabel("Acceleration [(m/s)/min]", color='blue')
ax1.set_yscale('symlog')
ax1.set_ylim(-10, 1.1*max(dy_fit))
ax1.tick_params(axis='y', labelcolor='blue')

# 1. Right axis data (Thermal power peaks)
ax2.plot(t2, y2, color='red', label='Thermal Power')
ax2.set_ylabel("Thermal Power [W/g]", color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 2. RIght axis data (Vicat)
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
ax3.plot(t3, y3, color='green', label='Vicat')
ax3.set_ylabel("Depth [mm]", color='green')
ax3.tick_params(axis='y', labelcolor='green')

# Vertical lines for initial and final set times
if initial_set_time is not None:
    ax3.axvline(x=initial_set_time, color='green', linestyle='--', alpha=0.6, label='Initial Setting')
if final_set_time is not None:
    ax3.axvline(x=final_set_time, color='purple', linestyle='--', alpha=0.6, label='Final Set')

# Legends
lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2, ax3]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.legend(lines, labels, loc='upper right')

ax1.set_title("Ultrasonic Acceleration, Thermal Power and Vicat Penetration")
ax1.set_xlabel("Time [min]")

plt.tight_layout(h_pad=4.0)
plt.subplots_adjust(top=0.95, bottom=0.07)

# Separate component graph
fig_single, ax_single = plt.subplots()
for i, deriv in enumerate(component_derivs):
    ax_single.plot(t, deriv, linestyle='--', label=f'Component {i + 1}')
ax_single.plot(t, dy_fit, color='black', linewidth=2, label='Total sum')
ax_single.set_yscale('symlog')
ax_single.set_ylim(0, 1.1*max(dy_fit))
ax_single.set_title("Acceleration Components and Their Sum Function")
ax_single.set_xlabel("Time [min]")
ax_single.set_ylabel("Acceleration [(m/s)/min]")
ax_single.legend()
# ax_single.text(0.05, 0.95, f"{model_type} model", transform=ax_single.transAxes, fontsize=10, verticalalignment='top')

# Separate summary graph
fig_extra, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(t, y, color='blue', label='Ultrasonic Testing')
ax1.set_ylabel("Ultrasonic velocity [m/s]", color='blue')
# ax1.set_xlim(min(t3)-50, max(t3)+50) # Scaling to make Vicat stand out more
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(t2, y2, color='red', label='Calorimetry')
ax2.set_ylabel("Thermal Power [W/g]", color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
ax3.plot(t3, y3, color='green', label='Vicat')
ax3.set_ylabel("Depth [mm]", color='green')
ax3.tick_params(axis='y', labelcolor='green')

if initial_set_time is not None:
    ax3.axvline(x=initial_set_time, color='green', linestyle='--', alpha=0.6, label='Initial Setting')
if final_set_time is not None:
    ax3.axvline(x=final_set_time, color='purple', linestyle='--', alpha=0.6, label='Final Set')

lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2, ax3]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.legend(lines, labels, loc='upper right')
ax1.set_title("Ultrasonic Acceleration, Thermal Power and Vicat Penetration")
ax1.set_xlabel("Time [min]")
fig_extra.tight_layout()

plt.show()


