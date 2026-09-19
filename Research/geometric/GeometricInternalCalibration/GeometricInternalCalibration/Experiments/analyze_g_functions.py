import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

def reconstruct_isotonic(params):
    """Reconstructs the sklearn IsotonicRegression prediction function from saved params."""
    iso = IsotonicRegression()
    # Manually restore internal state
    iso.X_thresholds_ = np.array(params['X_thresholds'])
    iso.y_thresholds_ = np.array(params['y_thresholds'])
    iso.X_min_ = params['X_min']
    iso.X_max_ = params['X_max']
    iso.y_min_ = params['y_min']
    iso.y_max_ = params['y_max']
    # Create a custom predict function that mimics sklearn's behavior
    # Sklearn's isotonic regression is a piecewise linear interpolator
    def predict(X):
        X = np.asarray(X)
        # Handle out-of-bounds values
        X_clipped = np.clip(X, iso.X_min_, iso.X_max_)
        # Use linear interpolation between thresholds
        return np.interp(X_clipped, iso.X_thresholds_, iso.y_thresholds_)
    
    iso.predict = predict
    return iso

def compare_functions(json_path):
    with open(json_path, 'r') as f:
        results = json.load(f)

    # Extract params
    sgc_data = results.get('global_random', {})
    coord_data = results.get('coordinate_sampling', {})

    if 'calibrator_params' not in sgc_data or 'calibrator_params' not in coord_data:
        print("Error: calibrator_params not found in JSON. Did you run the updated code?")
        return

    sgc_params = sgc_data['calibrator_params']
    coord_params = coord_data['calibrator_params']

    # Check if isotonic_params exist
    if 'isotonic_params' not in sgc_params:
        print(f"Error: SGC calibrator_params missing 'isotonic_params'. Found keys: {sgc_params.keys()}")
        print(f"Fitting method: {sgc_params.get('fitting_method', 'unknown')}")
        return
    
    if 'isotonic_params' not in coord_params:
        print(f"Error: Coordinate calibrator_params missing 'isotonic_params'. Found keys: {coord_params.keys()}")
        print(f"Fitting method: {coord_params.get('fitting_method', 'unknown')}")
        return

    # 1. Reconstruct Functions (G)
    # Note: These functions map Normalized Stability [0,1] -> Probability [0,1]
    g_sgc = reconstruct_isotonic(sgc_params['isotonic_params'])
    g_coord = reconstruct_isotonic(coord_params['isotonic_params'])

    # 2. Evaluate on a grid
    x_grid = np.linspace(0, 1, 1000)
    y_sgc = g_sgc.predict(x_grid)
    y_coord = g_coord.predict(x_grid)

    # 3. Calculate Distance (Liron's metric)
    # RMSE (Root Mean Square Error) between the two functions
    rmse = np.sqrt(np.mean((y_sgc - y_coord)**2))
    max_diff = np.max(np.abs(y_sgc - y_coord))

    print("="*60)
    print("COMPARISON OF G FUNCTIONS (Isotonic Regression Curves)")
    print("="*60)
    print(f"RMSE between functions:     {rmse:.6f}")
    print(f"Max absolute difference:    {max_diff:.6f}")
    print("-" * 60)
    print("Interpretation:")
    print("If RMSE is low (< 0.05), the methods learned nearly identical calibration maps.")
    print("This implies the Coordinate method captures the same signal as the Layer method.")
    print("="*60)

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, y_sgc, label='SGC (Layers) G(x)', linewidth=2)
    plt.plot(x_grid, y_coord, label='Coordinate G(x)', linewidth=2, linestyle='--')
    plt.fill_between(x_grid, y_sgc, y_coord, alpha=0.2, color='gray', label='Difference')
    
    plt.title(f"Comparison of Learned Calibration Functions (G)\nRMSE: {rmse:.5f}")
    plt.xlabel("Normalized Stability Score")
    plt.ylabel("Calibrated Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("g_function_comparison.png")
    print("Plot saved to g_function_comparison.png")

if __name__ == "__main__":
    # Replace with your actual output file path
    compare_functions("calibration_comparison/ablation_baseline_cross_entropy_cifar100_resnet18_seed28.json")

