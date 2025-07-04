# Mathematical Foundation and Theory

## Table of Contents
1. [Introduction](#introduction)
2. [Data Diagnostics Framework](#data-diagnostics-framework)
3. [Univariate GARCH Models](#univariate-garch-models)
4. [Dynamic Conditional Correlation (DCC)](#dynamic-conditional-correlation-dcc)
5. [Value at Risk (VaR) Calculation](#value-at-risk-var-calculation)
6. [Model Validation and Testing](#model-validation-and-testing)
7. [Implementation Considerations](#implementation-considerations)
8. [References](#references)

---

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
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$$

Where:
- $H_0$: $\gamma = 0$ (unit root, non-stationary)
- $H_1$: $\gamma < 0$ (stationary)

**KPSS Test:**
$$y_t = \xi t + r_t + \epsilon_t$$
$$r_t = r_{t-1} + u_t$$

Where:
- $H_0$: $\sigma_u^2 = 0$ (stationary)
- $H_1$: $\sigma_u^2 > 0$ (non-stationary)

**Implementation Decision Rule:**
```
if ADF_stationary AND KPSS_stationary:
    conclusion = "Stationary"
elif NOT ADF_stationary AND NOT KPSS_stationary:
    conclusion = "Non-stationary"
else:
    conclusion = "Uncertain - conflicting results"
```

### 2. Distribution Analysis

**Normality Testing Battery:**

*Shapiro-Wilk Test* (for $n \leq 5000$):
$$W = \frac{(\sum_{i=1}^n a_i x_{(i)})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}$$

*Jarque-Bera Test*:
$$JB = \frac{n}{6}(S^2 + \frac{(K-3)^2}{4})$$

Where $S$ is skewness and $K$ is kurtosis.

**Automatic Distribution Selection:**
```python
def recommend_distribution(skewness, kurtosis, normality_tests):
    if is_normal and |skewness| < 0.5 and |kurtosis| < 1:
        return 'normal'
    elif |skewness| > 1.0:
        return 'skewed'
    elif kurtosis > 2:  # Fat tails
        return 't'
    else:
        return 't'  # Default for financial data
```

### 3. Volatility Clustering Detection

**ARCH Test:**
$$\hat{\epsilon}_t^2 = \alpha_0 + \sum_{i=1}^q \alpha_i \hat{\epsilon}_{t-i}^2 + v_t$$

Test statistic: $LM = T \cdot R^2 \sim \chi^2(q)$

**Ljung-Box Test on Squared Returns:**
$$Q_{LB} = T(T+2)\sum_{k=1}^h \frac{\hat{\rho}_k^2}{T-k}$$

Where $\hat{\rho}_k$ is the sample autocorrelation of squared returns at lag $k$.

### 4. Cointegration Analysis

**Engle-Granger Two-Step Method:**

Step 1: Estimate cointegrating regression
$$y_t = \alpha + \beta x_t + u_t$$

Step 2: Test residuals for unit root
$$\Delta u_t = \gamma u_{t-1} + \sum_{i=1}^p \delta_i \Delta u_{t-i} + \epsilon_t$$

---

## Univariate GARCH Models

### GARCH(p,q) Specification

For each asset $i$, we model:

**Return Equation:**
$$r_{i,t} = \mu_i + \epsilon_{i,t}$$

**Variance Equation:**
$$\sigma_{i,t}^2 = \omega_i + \sum_{j=1}^{q_i} \alpha_{i,j} \epsilon_{i,t-j}^2 + \sum_{k=1}^{p_i} \beta_{i,k} \sigma_{i,t-k}^2$$

**Standardized Residuals:**
$$z_{i,t} = \frac{\epsilon_{i,t}}{\sigma_{i,t}}$$

### Distribution Assumptions

Based on diagnostic results, we use:

**Student-t Distribution:**
$$f(z_t; \nu) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\pi(\nu-2)}\Gamma(\frac{\nu}{2})} \left(1 + \frac{z_t^2}{\nu-2}\right)^{-\frac{\nu+1}{2}}$$

**Normal Distribution:**
$$f(z_t) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z_t^2}{2}\right)$$

### Stationarity Conditions

For GARCH(1,1): $\alpha_i + \beta_i < 1$

**Implementation Note:** Our code includes automatic fallback to exponential smoothing when GARCH estimation fails:

$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_{t-1}^2$$

where $\lambda = 0.94$ (RiskMetrics standard).

---

## Dynamic Conditional Correlation (DCC)

### DCC Model Specification

Following Engle (2002), the DCC model assumes:

$$H_t = D_t R_t D_t$$

Where:
- $D_t = \text{diag}(\sqrt{h_{1,t}}, \ldots, \sqrt{h_{n,t}})$
- $R_t$ is the conditional correlation matrix

### DCC Evolution

**Step 1: Pseudo-correlation Matrix**
$$Q_t = (1-\alpha-\beta)\bar{Q} + \alpha z_{t-1}z_{t-1}' + \beta Q_{t-1}$$

Where:
- $\bar{Q} = E[z_t z_t']$ (unconditional correlation matrix)
- $z_t = (z_{1,t}, \ldots, z_{n,t})'$ (standardized residuals)
- $\alpha, \beta \geq 0$ and $\alpha + \beta < 1$

**Step 2: Correlation Matrix**
$$R_t = Q_t^{*-1} Q_t Q_t^{*-1}$$

Where $Q_t^* = \text{diag}(\sqrt{q_{11,t}}, \ldots, \sqrt{q_{nn,t}})$

### Log-Likelihood Function

$$\ell_t = -\frac{1}{2}\left(\log|R_t| + z_t' R_t^{-1} z_t\right)$$

**Implementation Enhancement:** Our code uses Cholesky decomposition for numerical stability:

```python
L = np.linalg.cholesky(R_t)
inv_R_z = np.linalg.solve(L, z_t)
quad_form = np.sum(inv_R_z**2)
log_det = 2 * np.sum(np.log(np.diag(L)))
```

### Parameter Constraints

**Theoretical Constraints:**
- $\alpha, \beta > 0$
- $\alpha + \beta < 1$

**Implementation Constraints:** (Enhanced for stability)
- $\alpha \in [10^{-6}, 0.3]$
- $\beta \in [10^{-6}, 0.98]$
- $\alpha + \beta < 0.995$

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
$$r_{p,t} = w' r_t$$

Where $w$ is the vector of portfolio weights.

**Portfolio Variance:**
$$\sigma_{p,t}^2 = w' H_t w = w' D_t R_t D_t w$$

### VaR Formulation

**Parametric VaR:**
$$\text{VaR}_t(\alpha) = -F^{-1}(\alpha) \sigma_{p,t}$$

Where $F^{-1}(\alpha)$ is the $\alpha$-quantile of the standardized distribution.

### Distribution-Specific Quantiles

**Normal Distribution:**
$$F^{-1}(\alpha) = \Phi^{-1}(\alpha)$$

**Student-t Distribution:**
$$F^{-1}(\alpha) = t_{\nu}^{-1}(\alpha)$$

Where $\nu$ is the degrees of freedom parameter.

**Multi-Alpha Implementation:**
Our framework simultaneously calculates VaR for multiple confidence levels:
- 99% VaR: $\alpha = 0.01$
- 95% VaR: $\alpha = 0.05$  
- 90% VaR: $\alpha = 0.10$

### One-Step-Ahead Forecasting

**Volatility Forecast:**
$$\hat{\sigma}_{i,T+1}^2 = \omega_i + \alpha_i \epsilon_{i,T}^2 + \beta_i \sigma_{i,T}^2$$

**Correlation Forecast:**
$$\hat{R}_{T+1} = Q_{T+1}^{*-1} Q_{T+1} Q_{T+1}^{*-1}$$

Where $Q_{T+1}$ evolves according to the DCC equation.

---

## Model Validation and Testing

### Basel II Traffic Light Test

**Zone Classification:**
- **Green Zone**: $\text{Breaches} \leq \text{Binomial}_{0.95}(n, \alpha)$
- **Yellow Zone**: $\text{Binomial}_{0.95}(n, \alpha) < \text{Breaches} \leq \text{Binomial}_{0.9999}(n, \alpha)$
- **Red Zone**: $\text{Breaches} > \text{Binomial}_{0.9999}(n, \alpha)$

Where $n$ is the number of observations and $\alpha$ is the VaR confidence level.

### Kupiec Unconditional Coverage Test

**Test Statistic:**
$$LR_{UC} = 2\left[x \ln\left(\frac{\hat{p}}{p}\right) + (n-x) \ln\left(\frac{1-\hat{p}}{1-p}\right)\right]$$

Where:
- $x$ = number of violations
- $n$ = total observations
- $\hat{p} = x/n$ (observed violation rate)
- $p = \alpha$ (expected violation rate)

**Distribution:** $LR_{UC} \sim \chi^2(1)$ under $H_0$

### Exception Rate Analysis

**Rolling Exception Rate:**
$$\hat{p}_t = \frac{1}{w} \sum_{i=t-w+1}^t I(\text{Violation}_i)$$

Where $w$ is the rolling window size and $I(\cdot)$ is the indicator function.

**Confidence Bands:**
$$\hat{p}_t \pm 1.96 \sqrt{\frac{p(1-p)}{w}}$$

---

## Implementation Considerations

### Numerical Stability

**Matrix Regularization:**
$$Q_t^{\text{reg}} = Q_t + \lambda I_n$$

Where $\lambda = 10^{-8}$ is the regularization parameter.

**Eigenvalue Decomposition:**
```python
eigenvals, eigenvecs = np.linalg.eigh(Q_t)
eigenvals = np.maximum(eigenvals, regularization)
Q_t = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
```

### Multithreading Architecture

**Parallel GARCH Fitting:**
- Each asset's GARCH model estimated independently
- ThreadPoolExecutor with automatic CPU detection
- Thread-safe progress reporting
- Fallback mechanisms for failed estimations

**Performance Optimization:**
- Typical speedup: 70-80% with 4+ cores
- Memory efficiency through shared data structures
- Optimized matrix operations using NumPy/SciPy

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

---

## Computational Complexity

### Time Complexity

**GARCH Estimation:** $O(T \cdot \log T)$ per asset (with T observations)
**DCC Estimation:** $O(T \cdot n^3)$ where n is number of assets
**Parallel Efficiency:** Near-linear scaling with number of CPU cores

### Memory Requirements

**Storage:** $O(T \cdot n^2)$ for correlation matrices
**Peak Memory:** Approximately 50-100 MB for typical portfolio sizes

---

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

## References

1. **Engle, R.F. (2002)**. "Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models." *Journal of Business & Economic Statistics*, 20(3), 339-350.

2. **Bollerslev, T. (1986)**. "Generalized autoregressive conditional heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.

3. **Christoffersen, P. (2003)**. *Elements of Financial Risk Management*. Academic Press.

4. **Basel Committee on Banking Supervision (2006)**. "International Convergence of Capital Measurement and Capital Standards: A Revised Framework." Bank for International Settlements.

5. **Kupiec, P.H. (1995)**. "Techniques for Verifying the Accuracy of Risk Measurement Models." *Journal of Derivatives*, 2(2), 73-84.

---

## Mathematical Notation

| Symbol | Description |
|--------|-------------|
| $r_{i,t}$ | Return of asset $i$ at time $t$ |
| $\sigma_{i,t}^2$ | Conditional variance of asset $i$ at time $t$ |
| $z_{i,t}$ | Standardized residual of asset $i$ at time $t$ |
| $H_t$ | Conditional covariance matrix at time $t$ |
| $R_t$ | Conditional correlation matrix at time $t$ |
| $Q_t$ | Pseudo-correlation matrix at time $t$ |
| $\alpha, \beta$ | DCC parameters |
| $w$ | Portfolio weight vector |
| $\text{VaR}_t(\alpha)$ | Value at Risk at confidence level $\alpha$ |
| $\nu$ | Degrees of freedom (Student-t distribution) |

---

*This documentation provides the mathematical foundation for understanding and extending the DCC-GARCH-VaR implementation. For implementation details, see the source code documentation.*