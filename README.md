
# DCC-GARCH-VaR Model with Comprehensive Diagnostics

A sophisticated Python implementation of Dynamic Conditional Correlation - Generalized Autoregressive Conditional Heteroskedasticity - Value at Risk (DCC-GARCH-VaR) modeling with integrated data diagnostics, multithreaded processing, and Basel II compliance testing.

## Overview

This project provides an enterprise-grade framework for portfolio risk management. The implementation combines cutting-edge financial modeling techniques with comprehensive diagnostic capabilities to ensure robust and reliable risk estimates.

**Key Innovation**: Integrated diagnostic system that automatically selects optimal model parameters based on data characteristics, significantly improving model performance and reliability.

## Features

### 🔬 **Comprehensive Data Diagnostics**
- **Stationarity Testing**: Augmented Dickey-Fuller (ADF) and KPSS tests with conflicting result handling
- **Distribution Analysis**: Shapiro-Wilk, Jarque-Bera, and D'Agostino normality tests with automatic distribution selection
- **Volatility Clustering Detection**: ARCH tests and Ljung-Box tests on squared returns
- **Outlier Detection**: Multi-sigma rule implementation with statistical thresholds
- **Cointegration Analysis**: Engle-Granger tests for long-run relationships between assets

### ⚡ **High-Performance Computing**
- **Multithreaded GARCH Fitting**: Parallel processing for multiple assets using ThreadPoolExecutor
- **Robust Numerical Methods**: Eigenvalue decomposition and Cholesky factorization for matrix stability
- **Fallback Models**: Exponential smoothing alternatives when GARCH fitting fails

### 📊 **Advanced VaR Modeling**
- **Multiple Confidence Levels**: Simultaneous VaR calculation for 99%, 95%, and 90% confidence levels
- **Dynamic Correlations**: Time-varying correlation matrices using DCC methodology
- **Distribution Flexibility**: Support for Normal, Student-t, and Skewed-t distributions with automatic selection
- **Consistent Windowing**: Maintains exact window sizes throughout backtesting for reliable results

### 🎯 **Regulatory Compliance**
- **Basel II Traffic Light Testing**: Green/Yellow/Red zone classification
- **Kupiec Unconditional Coverage Tests**: Statistical validation of VaR model accuracy
- **Exception Rate Monitoring**: Real-time tracking of VaR breaches
- **Comprehensive Reporting**: Detailed model performance and diagnostic reports

### 📈 **Advanced Visualization**
- **Multi-Alpha VaR Plots**: Simultaneous visualization of multiple confidence levels
- **Diagnostic Dashboard**: Six-panel comprehensive analysis including autocorrelation, distribution analysis, and traffic light results
- **Rolling Exception Analysis**: Time-varying model performance assessment
- **Interactive Performance Metrics**: Real-time model statistics and insights

## Installation

### Prerequisites
- Python 3.7+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- statsmodels
- arch (GARCH modeling)
- concurrent.futures (included in Python 3.2+)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd DCC-GARCH-VaR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
```txt
numpy>=1.19.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.3.0
seaborn>=0.11.0
statsmodels>=0.12.0
arch>=5.0.0
```

## Usage

### Quick Start

```bash
python main.py
```

This runs the complete analysis pipeline with synthetic data if no real data is available.

### Advanced Usage

```python
from src.dcc_garch_var import DCCGARCHVAR
from src.data_diagnostics import DataDiagnostics
import pandas as pd
import numpy as np

# Load your financial time series data
data = pd.read_csv('Stock_data.csv', index_col=0, parse_dates=True)
returns = np.log(data / data.shift(1)).dropna()

# Initialize model with comprehensive diagnostics
model = DCCGARCHVAR(
    weights=[0.4, 0.35, 0.25],           # Portfolio weights
    alphas=[0.01, 0.05, 0.1],            # VaR confidence levels
    window_size=500,                      # Rolling window size
    dist='auto',                          # Auto-select distribution
    refit_frequency=5,                    # Refit every 5 days
    enable_diagnostics=True,              # Enable diagnostic system
    n_threads=4                           # Parallel processing threads
)

# Split data for backtesting
split_point = int(0.8 * len(returns))
train_data = returns.iloc[:split_point]
test_data = returns.iloc[split_point:]

# Run comprehensive backtesting
results = model.backtest(train_data, test_data)
evaluation = model.evaluate_backtest(results)

# Generate detailed report
model.generate_comprehensive_report(results, evaluation)

# Create visualizations
model.plot_results(results)
```

### Configuration Options

```python
# Model Configuration
DCCGARCHVAR(
    weights=[0.4, 0.3, 0.3],             # Portfolio weights (must sum to 1)
    alphas=[0.01, 0.025, 0.05, 0.1],     # Multiple VaR levels
    window_size=252,                      # 1-year rolling window
    dist='auto',                          # 'auto', 't', 'normal', 'skewed'
    refit_frequency=10,                   # Refit model every N days
    scale_factor=100,                     # Return scaling (100 for %)
    enable_diagnostics=True,              # Enable diagnostic engine
    n_threads=None                        # Auto-detect CPU cores
)
```

## Data Format

Input data should be a CSV file with the following structure:

```csv
Date,Asset1,Asset2,Asset3
2020-01-01,100.5,50.2,75.8
2020-01-02,101.2,49.8,76.1
2020-01-03,99.8,51.0,75.5
```

**Requirements**:
- Date column (will be set as index)
- Price or return data for each asset
- No missing values in the analysis period
- Minimum 250 observations recommended

## Project Structure

```
DCC-GARCH-VaR/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── data_diagnostics.py        # Comprehensive diagnostic engine
│   │   ├── DataDiagnostics         # Main diagnostic class
│   │   ├── Stationarity tests      # ADF, KPSS implementation
│   │   ├── Distribution analysis   # Normality, skewness, kurtosis
│   │   ├── Volatility clustering   # ARCH effects detection
│   │   ├── Outlier detection      # Multi-sigma rules
│   │   └── Cointegration testing  # Engle-Granger tests
│   └── dcc_garch_var.py           # Core DCC-GARCH-VaR engine
│       ├── DCCGARCHVAR            # Main model class
│       ├── Multithreaded GARCH    # Parallel model fitting
│       ├── DCC optimization       # Dynamic correlation estimation
│       ├── VaR forecasting       # Multi-alpha risk calculation
│       ├── Backtesting engine    # Rolling window validation
│       ├── Traffic light testing # Basel II compliance
│       └── Visualization suite   # Comprehensive plotting
├── Stock_data.csv                 # Sample/input data file
├── README.md                      # This documentation
├── requirements.txt               # Python dependencies
└── main.py                        # Example execution script
```

## Methodology

### 1. **Data Diagnostic Phase**
The diagnostic engine automatically analyzes your data and provides recommendations:

- **Distribution Selection**: Automatically chooses between Normal, Student-t, or Skewed distributions
- **Model Parameter Adjustment**: Optimizes window size and refit frequency based on volatility clustering
- **Outlier Handling**: Identifies and manages extreme observations
- **Stationarity Validation**: Ensures proper model assumptions

### 2. **DCC-GARCH Estimation**
Two-stage estimation process with enhanced numerical stability:

**Stage 1 - Univariate GARCH**:
```
σ²ᵢ,ₜ = ωᵢ + αᵢε²ᵢ,ₜ₋₁ + βᵢσ²ᵢ,ₜ₋₁
```

**Stage 2 - Dynamic Conditional Correlation**:
```
Qₜ = (1-α-β)Q̄ + α(zₜ₋₁zₜ₋₁') + βQₜ₋₁
Rₜ = D⁻¹ₜQₜD⁻¹ₜ
```

### 3. **VaR Calculation**
Portfolio VaR using optimal distribution:
```
VaRₜ(α) = -F⁻¹(α) × √(w'Hₜw)
```

Where Hₜ = DₜRₜDₜ is the conditional covariance matrix.

### 4. **Model Validation**
- **Kupiec Test**: Unconditional coverage validation
- **Traffic Light Test**: Basel II regulatory compliance
- **Rolling Exception Analysis**: Time-varying performance assessment

## Performance Features

### Multithreading Architecture
- **Parallel GARCH Fitting**: Simultaneous model estimation for multiple assets
- **Thread-Safe Operations**: Robust concurrent processing with proper synchronization
- **Automatic CPU Detection**: Optimizes thread count based on available hardware
- **Memory Efficient**: Minimal overhead with intelligent resource management

### Numerical Stability
- **Eigenvalue Regularization**: Ensures positive definite covariance matrices
- **Cholesky Decomposition**: Numerically stable matrix operations
- **Outlier Clipping**: Robust handling of extreme values
- **Fallback Mechanisms**: Exponential smoothing when GARCH fails

## Results and Output

### Comprehensive Reports
- **Diagnostic Summary**: Complete data analysis with recommendations
- **VaR Performance**: Multi-alpha breach analysis and coverage statistics
- **Model Quality**: Goodness-of-fit metrics and stability measures
- **Regulatory Compliance**: Traffic light zones and exception rates

### Visualization Suite
- **Multi-Alpha VaR Charts**: Simultaneous confidence level display
- **Diagnostic Dashboard**: Six-panel comprehensive analysis
- **Performance Analytics**: Rolling statistics and model validation
- **Distribution Analysis**: Empirical vs theoretical comparisons

### Export Capabilities
```python
# Automatic CSV export of results
results_df = pd.DataFrame({
    'Date': results['test_dates'],
    'Portfolio_Return': results['actual_returns'],
    'VaR_99': results['VaR_forecasts'][0.01],
    'VaR_95': results['VaR_forecasts'][0.05],
    'VaR_90': results['VaR_forecasts'][0.1],
    'Volatility_Forecast': results['VaR_std_forecasts']
})
```


### Development Setup
```bash
git clone <repository>
cd DCC-GARCH-VaR
pip install -e .  # Editable installation
python -m pytest tests/  # Run test suite
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Academic References

- **Engle, R. (2002)**. "Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models." *Journal of Business & Economic Statistics*, 20(3), 339-350.

- **Bollerslev, T. (1986)**. "Generalized autoregressive conditional heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.

- **Christoffersen, P. (2003)**. *Elements of Financial Risk Management*. Academic Press.




