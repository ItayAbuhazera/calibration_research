## Project Structure

- `calibrators/`: Calibration algorithms implementation
  - `base_calibrator.py`: Abstract base class for calibrators
  - `calibrators.py`: Main calibration implementations
  - `geometric_calibrators.py`: Geometric-based calibration methods
  - `non_parametric_calibrators.py`: Non-parametric calibration methods
  - `parametric_calibrators.py`: Parametric calibration methods
  - `specialized_calibrators.py`: Specialized calibration implementations
  - `ensemble_calibrators.py`: Ensemble calibration methods

- `models/`: Model architectures and factory
  - `model_architectures.py`: Neural network and other model architectures
  - `model_factory.py`: Factory functions for model creation

- `utils/`: Utility functions
  - `calibration.py`: Calibration methods
  - `data_utils.py`: Dataset loading and preprocessing
  - `metrics.py`: Calibration metrics calculation
  - `model_utils.py`: Model training and evaluation utilities
  - `utils.py`: General utility functions

## Quick Start

The notebook `Example_Run.ipynb` demonstrates basic usage. Here's a typical workflow:

1. Load and prepare dataset:
```python
from utils.data_utils import load_and_split_data
from scripts.main import prepare_model
from calibrators.geometric_calibrators import GeometricCalibrator
from utils.metrics import CalibrationMetrics

# Load dataset
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    dataset_name="MNIST", 
    random_state=42
)

# Create and train model
model = prepare_model(
    X_train, y_train, X_val, y_val,
    dataset_name="MNIST",
    random_state=42,
    model_type="RF"
)

# Initialize and apply calibrator
try:
    calibrator = GeometricCalibrator(
        model=model,
        X_train=X_train,
        y_train=y_train
    )
    
    calibrator.fit(X_val, y_val)
    calibrated_probs = calibrator.calibrate(X_test)
    
    # Evaluate calibration
    metrics = CalibrationMetrics(calibrated_probs, model.predict(X_test), y_test)
    print(f"ECE: {metrics.ece():.4f}")
    
except Exception as e:
    print(f"Error during calibration: {e}")
```

## Running Experiments

The main experiments from the paper can be reproduced using `Example_Run.ipynb`:

1. Dataset selection:
   - Standard benchmarks: MNIST, Fashion-MNIST, CIFAR10, SignLanguage
   - Complex tasks: CIFAR100 (with pretrained EfficientNet), GTSRB

2. Model comparison:
   - Random Forest (RF)
   - Gradient Boosting (GB) 
   - CNN (pretrained EfficientNet for CIFAR100)

3. Data reduction methods:
   - None (baseline)
   - Feature reduction: Avgpool, Maxpool, PCA
   - Sample reduction: randset, randpix

The notebook reproduces the key experiments from Sections 4.2 and 5 of the paper, including:
- Calibration quality comparison (ECE metrics)
- Runtime performance analysis
- Impact of different data reduction techniques
- Comparison with state-of-the-art calibration methods