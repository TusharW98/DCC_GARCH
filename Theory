# Mathematical Foundation and Theory

## Introduction

This document provides the theoretical foundation for the DCC-GARCH-VaR model with integrated diagnostics. Our implementation extends the classical approaches by incorporating comprehensive data diagnostics that automatically optimize model parameters and distributions based on empirical data characteristics.

### Model Overview

The framework consists of four main components:
1. **Diagnostic Engine**: Automated data analysis and parameter optimization
2. **Univariate GARCH**: Individual asset volatility modeling
3. **DCC Estimation**: Dynamic correlation modeling
4. **VaR Calculation**: Multi-confidence level risk assessment

---

## Data Diagnostics Framework

### 1. Stationarity Testing

We employ a dual-testing approach to ensure robust stationarity assessment:

**Augmented Dickey-Fuller (ADF) Test:**

The ADF test regression equation is:

```
Δy_t = α + βt + γy_{t-1} + Σ(i=1 to p) δ_i Δy_{t-i} + ε_t
```

Where:
- H₀: γ = 0 (unit root, non-stationary)
- H₁: γ < 0 (stationary)

**KPSS Test:**

The KPSS test decomposes the series as:

```
y_t = ξt + r_t + ε_t
r_t = r_{t-1} + u_t
```

Where:
- H₀: σ²_u = 0 (stationary)
- H₁: σ²_u > 0 (non-stationary)

**Implementation Decision Rule:**
```python
if ADF_stationary AND KPSS_stationary:
    conclusion = "Stationary"
elif NOT ADF_stationary AND NOT KPSS_stationary:
    conclusion = "Non-stationary"
else:
    conclusion = "Uncertain - conflicting results"
```

### 2. Distribution Analysis

**Normality Testing Battery:**

*Shapiro-Wilk Test* (for n ≤ 5000):

The test statistic is:
```
W = [Σ(i=1 to n) a_i x_(i)]² / Σ(i=1 to n) (x_i - x̄)²
```

*Jarque-Bera Test*:

The test statistic is:
```
JB = (n/6)[S² + (K-3)²/4]
```

Where S is skewness and K is kurtosis.

**Automatic Distribution Selection:**
```python
def recommend_distribution(skewness, kurtosis, normality_tests):
    if is_normal and abs(skewness) < 0.5 and abs(kurtosis) < 1:
        return 'normal'
    elif abs(skewness) > 1.0:
        return 'skewed'
    elif kurtosis > 2:  # Fat tails
        return 't'
    else:
        return 't'  # Default for financial data
```

### 3. Volatility Clustering Detection

**ARCH Test:**

The ARCH test regression is:
```
ε̂²_t = α₀ + Σ(i=1 to q) α_i ε̂²_{t-i} + v_t
```

Test statistic: `LM = T × R² ~ χ²(q)`

**Ljung-Box Test on Squared Returns:**

The test statistic is:
```
Q_LB = T(T+2) Σ(k=1 to h) ρ̂²_k/(T-k)
```

Where ρ̂_k is the sample autocorrelation of squared returns at lag k.

### 4. Cointegration Analysis

**Engle-Granger Two-Step Method:**

Step 1: Estimate cointegrating regression
```
y_t = α + βx_t + u_t
```

Step 2: Test residuals for unit root
```
Δu_t = γu_{t-1} + Σ(i=1 to p) δ_i Δu_{t-i} + ε_t
```

---

## Univariate GARCH Models

### GARCH(p,q) Specification

For each asset i, we model:

**Return Equation:**
```
r_{i,t} = μ_i + ε_{i,t}
```

**Variance Equation:**
```
σ²_{i,t} = ω_i + Σ(j=1 to q_i) α_{i,j} ε²_{i,t-j} + Σ(k=1 to p_i) β_{i,k} σ²_{i,t-k}
```

**Standardized Residuals:**
```
z_{i,t} = ε_{i,t} / σ_{i,t}
```

### Distribution Assumptions

Based on diagnostic results, we use:

**Student-t Distribution:**

The probability density function is:
```
f(z_t; ν) = [Γ((ν+1)/2) / (√(π(ν-2)) Γ(ν/2))] × [1 + z²_t/(ν-2)]^(-(ν+1)/2)
```

**Normal Distribution:**
```
f(z_t) = (1/√(2π)) × exp(-z²_t/2)
```

### Stationarity Conditions

For GARCH(1,1): `α_i + β_i < 1`

**Implementation Note:** Our code includes automatic fallback to exponential smoothing when GARCH estimation fails:

```
σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}
```

where λ = 0.94 (RiskMetrics standard).

---

## Dynamic Conditional Correlation (DCC)

### DCC Model Specification

Following Engle (2002), the DCC model assumes:

```
H_t = D_t R_t D_t
```

Where:
- D_t = diag(√h_{1,t}, ..., √h_{n,t})
- R_t is the conditional correlation matrix

### DCC Evolution

**Step 1: Pseudo-correlation Matrix**
```
Q_t = (1-α-β)Q̄ + α z_{t-1}z'_{t-1} + β Q_{t-1}
```

Where:
- Q̄ = E[z_t z'_t] (unconditional correlation matrix)
- z_t = (z_{1,t}, ..., z_{n,t})' (standardized residuals)
- α, β ≥ 0 and α + β < 1

**Step 2: Correlation Matrix**
```
R_t = Q*_t^{-1} Q_t Q*_t^{-1}
```

Where Q*_t = diag(√q_{11,t}, ..., √q_{nn,t})

### Log-Likelihood Function

```
ℓ_t = -0.5 × [log|R_t| + z'_t R_t^{-1} z_t]
```

**Implementation Enhancement:** Our code uses Cholesky decomposition for numerical stability:

```python
L = np.linalg.cholesky(R_t)
inv_R_z = np.linalg.solve(L, z_t)
quad_form = np.sum(inv_R_z**2)
log_det = 2 * np.sum(np.log(np.diag(L)))
```

### Parameter Constraints

**Theoretical Constraints:**
- α, β > 0
- α + β < 1

**Implementation Constraints:** (Enhanced for stability)
- α ∈ [10⁻⁶, 0.3]
- β ∈ [10⁻⁶, 0.98]
- α + β < 0.995

### Optimization Strategy

We employ multiple optimization methods:

1. **L-BFGS-B** with gradient information
2. **Differential Evolution** for global optimization
3. **Diagnostic-informed starting points**

**Starting Point Selection:**
```python
def get_diagnostic_informed_starting_points():
    if high_cointegration:
        return [0.02, 0.97]  # More persistent
    elif strong_volatility_clustering:
        return [0.08, 0.88]  # More responsive
    else:
        return [0.05, 0.90]  # Moderate
```

---

## Value at Risk (VaR) Calculation

### Portfolio Return Distribution

**Portfolio Return:**
```
r_{p,t} = w' r_t
```

Where w is the vector of portfolio weights.

**Portfolio Variance:**
```
σ²_{p,t} = w' H_t w = w' D_t R_t D_t w
```

### VaR Formulation

**Parametric VaR:**
```
VaR_t(α) = -F^{-1}(α) × σ_{p,t}
```

Where F⁻¹(α) is the α-quantile of the standardized distribution.

### Distribution-Specific Quantiles

**Normal Distribution:**
```
F^{-1}(α) = Φ^{-1}(α)
```

**Student-t Distribution:**
```
F^{-1}(α) = t_ν^{-1}(α)
```

Where ν is the degrees of freedom parameter.

**Multi-Alpha Implementation:**
Our framework simultaneously calculates VaR for multiple confidence levels:
- 99% VaR: α = 0.01
- 95% VaR: α = 0.05  
- 90% VaR: α = 0.10

### One-Step-Ahead Forecasting

**Volatility Forecast:**
```
σ̂²_{i,T+1} = ω_i + α_i ε²_{i,T} + β_i σ²_{i,T}
```

**Correlation Forecast:**
```
R̂_{T+1} = Q*_{T+1}^{-1} Q_{T+1} Q*_{T+1}^{-1}
```

Where Q_{T+1} evolves according to the DCC equation.

---

## Model Validation and Testing

### Basel II Traffic Light Test

**Zone Classification:**
- **Green Zone**: Breaches ≤ Binomial₀.₉₅(n, α)
- **Yellow Zone**: Binomial₀.₉₅(n, α) < Breaches ≤ Binomial₀.₉₉₉₉(n, α)
- **Red Zone**: Breaches > Binomial₀.₉₉₉₉(n, α)

Where n is the number of observations and α is the VaR confidence level.

### Kupiec Unconditional Coverage Test

**Test Statistic:**
```
LR_UC = 2[x × ln(p̂/p) + (n-x) × ln((1-p̂)/(1-p))]
```

Where:
- x = number of violations
- n = total observations
- p̂ = x/n (observed violation rate)
- p = α (expected violation rate)

**Distribution:** LR_UC ~ χ²(1) under H₀

### Exception Rate Analysis

**Rolling Exception Rate:**
```
p̂_t = (1/w) × Σ(i=t-w+1 to t) I(Violation_i)
```

Where w is the rolling window size and I(·) is the indicator function.

**Confidence Bands:**
```
p̂_t ± 1.96 × √[p(1-p)/w]
```

---

## Implementation Considerations

### Numerical Stability

**Matrix Regularization:**
```
Q_t^{reg} = Q_t + λI_n
```

Where λ = 10⁻⁸ is the regularization parameter.

**Eigenvalue Decomposition:**
```python
eigenvals, eigenvecs = np.linalg.eigh(Q_t)
eigenvals = np.maximum(eigenvals, regularization)
Q_t = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
```

### Consistent Windowing

**Rolling Window Strategy:**
```python
# Maintain exact window size throughout backtesting
if len(window) > self.window_size:
    window = window.iloc[-self.window_size:]  # Keep exactly window_size days
```

This ensures consistent sample sizes and comparable model estimates across time.

### Diagnostic Integration

**Automatic Parameter Adjustment:**
1. **Distribution Selection**: Based on normality tests and tail behavior
2. **Window Size**: Adjusted for outlier percentage and data characteristics  
3. **Refit Frequency**: Based on volatility clustering strength
4. **Starting Parameters**: Informed by cointegration and ARCH test results



## Extensions and Future Work

### Potential Enhancements

1. **Asymmetric DCC**: Incorporate leverage effects
2. **Regime-Switching**: Allow for structural breaks
3. **High-Frequency Data**: Realized volatility integration
4. **Machine Learning**: Neural network-based correlation modeling

### Research Applications

- **Portfolio Optimization**: Integration with mean-variance frameworks
- **Risk Budgeting**: Component VaR decomposition
- **Stress Testing**: Scenario-based correlation modeling
- **Regulatory Capital**: Basel III FRTB implementation

---

## Mathematical Notation Summary

| Symbol | Description |
|--------|-------------|
| r_{i,t} | Return of asset i at time t |
| σ²_{i,t} | Conditional variance of asset i at time t |
| z_{i,t} | Standardized residual of asset i at time t |
| H_t | Conditional covariance matrix at time t |
| R_t | Conditional correlation matrix at time t |
| Q_t | Pseudo-correlation matrix at time t |
| α, β | DCC parameters |
| w | Portfolio weight vector |
| VaR_t(α) | Value at Risk at confidence level α |
| ν | Degrees of freedom (Student-t distribution) |
| ε_{i,t} | Error term for asset i at time t |
| μ_i | Mean return for asset i |
| ω_i | GARCH constant term for asset i |

---

## Key Model Equations Reference

### GARCH(1,1) Model:
```
σ²_t = ω + αε²_{t-1} + βσ²_{t-1}
```

### DCC Evolution:
```
Q_t = (1-α-β)Q̄ + αz_{t-1}z'_{t-1} + βQ_{t-1}
R_t = Q*_t^{-1} Q_t Q*_t^{-1}
```

### VaR Calculation:
```
VaR_t(α) = -Φ^{-1}(α) × √(w'H_t w)    [Normal case]
VaR_t(α) = -t_ν^{-1}(α) × √(w'H_t w)   [Student-t case]
```

### Traffic Light Zones:
```
Green:  Violations ≤ B(n, α, 0.95)
Yellow: B(n, α, 0.95) < Violations ≤ B(n, α, 0.9999)
Red:    Violations > B(n, α, 0.9999)
```

Where B(n, α, p) is the p-th percentile of Binomial(n, α).

---

## References

1. **Engle, R.F. (2002)**. "Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models." *Journal of Business & Economic Statistics*, 20(3), 339-350.

2. **Bollerslev, T. (1986)**. "Generalized autoregressive conditional heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.

3. **Christoffersen, P. (2003)**. *Elements of Financial Risk Management*. Academic Press.

4. **Kupiec, P.H. (1995)**. "Techniques for Verifying the Accuracy of Risk Measurement Models." *Journal of Derivatives*, 2(2), 73-84.

---

*This documentation provides the mathematical foundation for understanding and extending the DCC-GARCH-VaR implementation. For implementation details, see the source code documentation.*
