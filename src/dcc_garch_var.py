import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


import numpy as np
import pandas as pd
from scipy.stats import t, norm, binom
from scipy.optimize import minimize, differential_evolution
from arch import arch_model
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import matplotlib.pyplot as plt
import warnings
from .data_diagnostics import DataDiagnostics

warnings.filterwarnings('ignore')


class DCCGARCHVAR:
    """
    Enhanced DCC-GARCH VaR model integrated with comprehensive data diagnostics.
    """
    
    def __init__(self, weights, alphas=[0.01, 0.05, 0.1], window_size=500, dist='auto', 
                 refit_frequency=5, scale_factor=100, enable_diagnostics=True, n_threads=None):
        """
        Enhanced DCC-GARCH VaR model with data diagnostics integration and multithreading
        
        Parameters:
        - weights: Portfolio weights
        - alphas: List of VaR confidence levels (e.g., [0.01, 0.05, 0.1])
        - window_size: Rolling window size for model fitting
        - dist: Distribution for GARCH ('t', 'normal', or 'auto' for diagnostics-based selection)
        - refit_frequency: How often to refit the model (in days)
        - scale_factor: Scaling factor for returns (100 for percentage returns)
        - enable_diagnostics: Whether to run comprehensive data diagnostics
        - n_threads: Number of threads for parallel GARCH fitting (None for auto-detection)
        """
        self.weights = np.array(weights)
        self.alphas = alphas
        self.window_size = window_size
        self.dist = dist
        self.refit_frequency = refit_frequency
        self.regularization = 1e-8
        self.scale_factor = scale_factor
        self.enable_diagnostics = enable_diagnostics
        
        # Adjust refit frequency for small window sizes to balance computation
        if self.window_size <= 100:
            self.refit_frequency = max(5, self.refit_frequency)
        
        # Multithreading setup
        if n_threads is None:
            import os
            self.n_threads = min(len(weights), os.cpu_count() or 4)  # Use available CPUs, max = number of assets
        else:
            self.n_threads = max(1, min(n_threads, len(weights)))
        
        print(f"Initialized with {self.n_threads} threads for parallel GARCH fitting")
        
        # Initialize diagnostics
        if self.enable_diagnostics:
            self.diagnostics = DataDiagnostics(dist=dist)
        else:
            self.diagnostics = None
            
        # Store diagnostics results
        self.diagnostic_results = {}
        self.optimal_distributions = {}
        
        # Thread-safe printing lock
        self._print_lock = Lock()
    
    def run_initial_diagnostics(self, returns_data):
        """Run comprehensive diagnostics on the data"""
        if not self.enable_diagnostics:
            print("Diagnostics disabled, using default settings")
            return
            
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE DATA DIAGNOSTICS")
        print("="*80)
        
        # Run diagnostics
        self.diagnostic_results = self.diagnostics.run_data_diagnostics(returns_data)
        
        # Select optimal distributions for each asset
        for asset in returns_data.columns:
            optimal_dist = self.diagnostics.select_distribution(asset)
            self.optimal_distributions[asset] = optimal_dist
            
        # Print diagnostic summary
        self._print_diagnostic_summary()
        
        # Adjust model parameters based on diagnostics
        self._adjust_model_parameters()
    
    def _print_diagnostic_summary(self):
        """Print a summary of diagnostic findings"""
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY & MODEL RECOMMENDATIONS")
        print("="*60)
        
        for asset, diag in self.diagnostic_results.items():
            if asset != 'cointegration':
                print(f"\n{asset}:")
                print(f"  Distribution: {diag['distribution']['recommended_dist'].upper()}")
                print(f"  ARCH Effects: {'Yes' if diag['volatility_clustering']['arch_test'].get('has_arch', False) else 'No'}")
                print(f"  Stationarity: {diag['stationarity']['conclusion']}")
                print(f"  Outliers: {diag['outliers']['count_3sigma']} ({diag['outliers']['percentage_3sigma']:.1f}%)")
                
                # Recommendations based on findings
                recommendations = []
                
                if diag['volatility_clustering']['arch_test'].get('has_arch', False):
                    recommendations.append("GARCH modeling appropriate")
                    
                if diag['distribution']['recommended_dist'] == 't':
                    recommendations.append("Use t-distribution for fat tails")
                elif diag['distribution']['recommended_dist'] == 'skewed':
                    recommendations.append("Consider skewed distributions")
                    
                if diag['outliers']['percentage_3sigma'] > 5:
                    recommendations.append("High outlier percentage - robust methods recommended")
                    
                if recommendations:
                    print(f"  Recommendations: {'; '.join(recommendations)}")
    
    def _adjust_model_parameters(self):
        """Adjust model parameters based on diagnostic findings"""
        if not self.diagnostic_results:
            return
            
        print(f"\n📊 ADJUSTING MODEL PARAMETERS BASED ON DIAGNOSTICS...")
        
        # Count assets recommending t-distribution
        t_dist_count = sum(1 for asset, diag in self.diagnostic_results.items() 
                          if asset != 'cointegration' and 
                          diag['distribution']['recommended_dist'] == 't')
        
        # If majority recommend t-distribution and we're in auto mode
        if self.dist == 'auto' and t_dist_count > len(self.diagnostic_results) / 2:
            self.dist = 't'
            print(f"   → Setting distribution to t-distribution (recommended for {t_dist_count} assets)")
        elif self.dist == 'auto':
            self.dist = 'normal'
            print(f"   → Setting distribution to normal (default)")
            
        # Adjust window size based on outlier percentage
        avg_outlier_pct = np.mean([diag['outliers']['percentage_3sigma'] 
                                  for asset, diag in self.diagnostic_results.items() 
                                  if asset != 'cointegration'])
        
        if avg_outlier_pct > 10:
            self.window_size = min(self.window_size, 300)
            print(f"   → Reducing window size to {self.window_size} due to high outlier percentage ({avg_outlier_pct:.1f}%)")
            
        # Adjust refit frequency based on volatility clustering
        strong_arch_count = sum(1 for asset, diag in self.diagnostic_results.items() 
                               if asset != 'cointegration' and 
                               diag['volatility_clustering']['arch_test'].get('has_arch', False))
        
        if strong_arch_count > len(self.diagnostic_results) / 2:
            self.refit_frequency = max(1, self.refit_frequency // 2)
            print(f"   → Increasing refit frequency to every {self.refit_frequency} days due to strong volatility clustering")
    
    def dcc_log_likelihood(self, params, std_residuals, Q_bar):
        """Enhanced DCC log-likelihood with better numerical stability"""
        a, b = params
        
        # Stricter parameter constraints for stability
        if a <= 1e-6 or b <= 1e-6 or a + b >= 0.995 or a >= 0.5 or b >= 0.995:
            return 1e10
            
        T, N = std_residuals.shape
        Q_t = Q_bar.copy()
        ll = 0
        
        # Add regularization to unconditional covariance
        Q_bar_reg = Q_bar + self.regularization * np.eye(N)
        
        # Precompute constants
        one_minus_ab = 1 - a - b
        
        for t in range(1, T):
            z_t = std_residuals.iloc[t - 1].values
            
            # Check for extreme values in residuals
            if np.any(np.abs(z_t) > 10):
                z_t = np.clip(z_t, -10, 10)
            
            # DCC evolution equation with numerical safeguards
            outer_prod = np.outer(z_t, z_t)
            Q_t = one_minus_ab * Q_bar_reg + a * outer_prod + b * Q_t
            
            # Ensure positive definiteness with eigenvalue decomposition
            try:
                eigenvals, eigenvecs = np.linalg.eigh(Q_t)
                eigenvals = np.maximum(eigenvals, self.regularization)
                Q_t = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            except np.linalg.LinAlgError:
                return 1e10
            
            # Compute correlation matrix with numerical stability
            try:
                diag_Q = np.diag(Q_t)
                if np.any(diag_Q <= 0):
                    return 1e10
                    
                sqrt_diag_inv = 1 / np.sqrt(diag_Q)
                D_inv = np.diag(sqrt_diag_inv)
                R_t = D_inv @ Q_t @ D_inv
                
                # Ensure diagonal elements are exactly 1
                np.fill_diagonal(R_t, 1.0)
                
                # Check correlation bounds
                if np.any(np.abs(R_t) > 1 + 1e-6):
                    R_t = np.clip(R_t, -0.999, 0.999)
                    np.fill_diagonal(R_t, 1.0)
                
                # Compute log-likelihood contribution
                det_R = np.linalg.det(R_t)
                if det_R <= 1e-12:
                    return 1e10
                
                # Use Cholesky decomposition for numerical stability
                try:
                    L = np.linalg.cholesky(R_t)
                    inv_R_z = np.linalg.solve(L, z_t)
                    quad_form = np.sum(inv_R_z**2)
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    
                    ll += 0.5 * (log_det + quad_form)
                    
                except np.linalg.LinAlgError:
                    # Fallback to standard inverse if Cholesky fails
                    try:
                        inv_R_t = np.linalg.inv(R_t)
                        ll += 0.5 * (np.log(det_R) + z_t @ inv_R_t @ z_t)
                    except np.linalg.LinAlgError:
                        return 1e10
                        
            except (np.linalg.LinAlgError, RuntimeWarning, FloatingPointError):
                return 1e10
        
        return ll if np.isfinite(ll) and ll > 0 else 1e10

    def fit_univariate_garch(self, returns, asset_name):
        """Enhanced univariate GARCH fitting with diagnostics-based distribution selection"""
        try:
            # Data validation
            if len(returns) < 50:
                raise ValueError(f"Insufficient data for {asset_name}: {len(returns)} observations")
            
            # Use diagnostic-recommended distribution if available
            if self.enable_diagnostics and asset_name in self.optimal_distributions:
                asset_dist = self.optimal_distributions[asset_name]
                if asset_dist == 'normal':
                    asset_dist = 'normal'
                elif asset_dist in ['t', 'skewed']:
                    asset_dist = 't'
                else:
                    asset_dist = self.dist
            else:
                asset_dist = self.dist if self.dist != 'auto' else 't'
            
            # Scale returns to percentage form
            returns_scaled = returns * self.scale_factor
            
            # Remove extreme outliers based on diagnostic thresholds
            returns_clean = returns_scaled.copy()
            
            # Check for sufficient variation
            if returns_clean.std() < 1e-4:
                print(f"Warning: {asset_name} has very low volatility")
                vol = max(returns_clean.std(), 1e-2)
                residuals = returns_clean / vol
                return {
                    'volatility': vol / self.scale_factor,
                    'residuals': residuals,
                    'conditional_volatility': pd.Series([vol / self.scale_factor] * len(returns), index=returns.index),
                    'is_fallback': True
                }
            
            # Try different GARCH specifications based on diagnostics
            garch_specs = [
                {'p': 1, 'q': 1, 'power': 2.0},  # Standard GARCH(1,1)
                {'p': 1, 'q': 1, 'power': 1.0},  # AVGARCH
                {'p': 2, 'q': 1, 'power': 2.0},  # GARCH(2,1)
            ]
            
            # If strong ARCH effects detected, try higher order models first
            if (self.enable_diagnostics and asset_name in self.diagnostic_results and 
                self.diagnostic_results[asset_name]['volatility_clustering']['arch_test'].get('has_arch', False)):
                garch_specs.insert(0, {'p': 1, 'q': 2, 'power': 2.0})  # GARCH(1,2)
            
            for spec in garch_specs:
                try:
                    model = arch_model(
                        returns_clean,
                        vol='GARCH',
                        p=spec['p'], 
                        q=spec['q'],
                        power=spec.get('power', 2.0),
                        dist=asset_dist,
                        mean='Zero',
                        rescale=False
                    )
                    
                    res = model.fit(
                        disp='off', 
                        show_warning=False,
                        options={'maxiter': 1000}
                    )
                    
                    # Validate the fit
                    if hasattr(res, 'std_resid') and hasattr(res, 'conditional_volatility'):
                        std_resid = res.std_resid
                        cond_vol = res.conditional_volatility / self.scale_factor
                        
                        # Check for reasonable values
                        if (np.all(np.isfinite(std_resid)) and 
                            np.all(np.isfinite(cond_vol)) and 
                            np.all(cond_vol > 0)):
                            
                            res_dict = {
                                'model': res,
                                'std_resid': std_resid,
                                'conditional_volatility': cond_vol,
                                'is_fallback': False,
                                'distribution_used': asset_dist
                            }
                            return res_dict
                            
                except Exception as e:
                    print(f"GARCH spec {spec} failed for {asset_name}: {e}")
                    continue
            
            # If all GARCH models fail, use exponential smoothing
            print(f"All GARCH models failed for {asset_name}, using exponential smoothing")
            return self._fallback_volatility_model(returns, asset_name)
            
        except Exception as e:
            print(f"GARCH fitting completely failed for {asset_name}: {e}")
            return self._fallback_volatility_model(returns, asset_name)

    def _fallback_volatility_model(self, returns, asset_name):
        """Exponential smoothing volatility model as fallback"""
        # Use exponential weighted moving average
        lambda_param = 0.94  # RiskMetrics standard
        
        # Initialize with sample variance
        vol_squared = returns.var()
        vol_history = [np.sqrt(vol_squared)]
        
        for i in range(1, len(returns)):
            vol_squared = lambda_param * vol_squared + (1 - lambda_param) * returns.iloc[i-1]**2
            vol_history.append(np.sqrt(max(vol_squared, 1e-8)))
        
        vol_series = pd.Series(vol_history, index=returns.index)
        residuals = returns / vol_series
        
        return {
            'volatility': vol_series.iloc[-1],
            'residuals': residuals,
            'conditional_volatility': vol_series,
            'is_fallback': True
        }

    def _fit_single_garch_threaded(self, asset_data):
        """
        Thread-safe wrapper for fitting a single GARCH model
        
        Parameters:
        -----------
        asset_data : tuple
            (asset_name, returns_series) tuple for parallel processing
            
        Returns:
        --------
        tuple : (asset_name, garch_result)
        """
        asset_name, returns = asset_data
        
        try:
            result = self.fit_univariate_garch(returns, asset_name)
            
            # Thread-safe progress reporting
            with self._print_lock:
                print(f"  ✓ Completed GARCH fitting for {asset_name}")
            
            return asset_name, result
            
        except Exception as e:
            with self._print_lock:
                print(f"  ✗ GARCH fitting failed for {asset_name}: {e}")
            
            # Return fallback result
            fallback_result = self._fallback_volatility_model(returns, asset_name)
            return asset_name, fallback_result

    def fit_dcc_garch(self, returns_window, init_dcc_params=None):
        """Enhanced DCC-GARCH fitting with multithreaded GARCH estimation"""
        tickers = returns_window.columns.tolist()
        garch_models = {}
        std_residuals = pd.DataFrame(index=returns_window.index, columns=tickers)
        cond_vols = pd.DataFrame(index=returns_window.index, columns=tickers)
        df_dict = {}

        print(f"Fitting univariate GARCH models for {len(tickers)} assets using {self.n_threads} threads...")
        
        # Prepare data for parallel processing
        asset_data_list = [(asset, returns_window[asset]) for asset in tickers]
        
        # Use ThreadPoolExecutor for parallel GARCH fitting
        start_time = pd.Timestamp.now()
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            # Submit all GARCH fitting tasks
            future_to_asset = {
                executor.submit(self._fit_single_garch_threaded, asset_data): asset_data[0] 
                for asset_data in asset_data_list
            }
            
            # Collect results as they complete
            completed = 0
            total = len(tickers)
            
            for future in as_completed(future_to_asset):
                asset_name = future_to_asset[future]
                
                try:
                    asset_name, res = future.result()
                    garch_models[asset_name] = res
                    completed += 1
                    
                    # Update progress
                    progress = (completed / total) * 100
                    with self._print_lock:
                        print(f"  Progress: {completed}/{total} ({progress:.1f}%) - Latest: {asset_name}")
                    
                except Exception as e:
                    with self._print_lock:
                        print(f"  ✗ Error processing {asset_name}: {e}")
                    
                    # Create fallback result
                    garch_models[asset_name] = self._fallback_volatility_model(
                        returns_window[asset_name], asset_name
                    )
        
        end_time = pd.Timestamp.now()
        elapsed = (end_time - start_time).total_seconds()
        print(f"✓ Parallel GARCH fitting completed in {elapsed:.2f} seconds")
        
        # Process results and build residuals/volatilities
        print("Processing GARCH results...")
        
        for asset in tickers:
            res = garch_models[asset]
            
            if not res.get('is_fallback', False) and 'model' in res:
                # Standard ARCH result object
                std_residuals[asset] = res['std_resid']
                cond_vols[asset] = res['conditional_volatility']
                
                if self.dist == 't' and hasattr(res['model'], 'params'):
                    df_param = res['model'].params.get('nu', 5)
                    df_dict[asset] = max(df_param, 2.1)
                else:
                    df_dict[asset] = 5
            else:
                # Fallback model
                std_residuals[asset] = res['residuals']
                cond_vols[asset] = res['conditional_volatility']
                df_dict[asset] = 5

        # Clean residuals
        std_residuals = std_residuals.dropna()
        
        if len(std_residuals) < 50:
            raise ValueError("Insufficient data for DCC estimation after cleaning")

        # Print residual statistics
        print(f"Residual statistics after parallel processing:")
        for col in std_residuals.columns:
            print(f"  {col}: mean={std_residuals[col].mean():.4f}, std={std_residuals[col].std():.4f}")

        # Robust unconditional correlation estimation
        Q_bar = self._robust_correlation_matrix(std_residuals)
        
        # Initialize DCC parameters with diagnostics-informed starting points
        if init_dcc_params is None:
            starting_points = self._get_diagnostic_informed_starting_points()
        else:
            # Use provided parameters plus some variations
            a_init, b_init = init_dcc_params
            starting_points = [
                [a_init, b_init],
                [max(0.01, a_init * 0.8), min(0.98, b_init * 1.02)],
                [min(0.15, a_init * 1.2), max(0.80, b_init * 0.98)]
            ]

        best_ll = 1e10
        best_params = [0.05, 0.90]
        
        print("Optimizing DCC parameters...")
        
        # Try different optimization methods
        for start_params in starting_points:
            # Method 1: L-BFGS-B
            try:
                bounds = [(1e-4, 0.3), (1e-4, 0.98)]
                result = minimize(
                    self.dcc_log_likelihood,
                    start_params,
                    args=(std_residuals, Q_bar),
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-9}
                )
                
                if result.success and result.fun < best_ll:
                    a, b = result.x
                    if a > 0 and b > 0 and a + b < 0.995:
                        best_ll = result.fun
                        best_params = result.x
                        
            except Exception as e:
                print(f"L-BFGS-B optimization failed: {e}")
            
            # Method 2: Differential Evolution
            try:
                bounds = [(1e-4, 0.2), (1e-4, 0.98)]
                result = differential_evolution(
                    self.dcc_log_likelihood,
                    bounds,
                    args=(std_residuals, Q_bar),
                    maxiter=500,
                    popsize=10,
                    seed=42
                )
                
                if result.success and result.fun < best_ll:
                    a, b = result.x
                    if a > 0 and b > 0 and a + b < 0.995:
                        best_ll = result.fun
                        best_params = result.x
                        
            except Exception as e:
                print(f"Differential evolution failed: {e}")

        a, b = best_params
        
        if best_ll < 1e9:
            print(f"DCC optimization successful: a={a:.4f}, b={b:.4f}, LL={best_ll:.2f}")
        else:
            print(f"DCC optimization failed, using default: a={a:.4f}, b={b:.4f}")

        # Compute time-varying correlation matrices
        R_t_list = self._compute_dynamic_correlations(std_residuals, Q_bar, a, b)

        return {
            'R_t': R_t_list,
            'a': a,
            'b': b,
            'std_residuals': std_residuals,
            'cond_vols': cond_vols,
            'df_dict': df_dict,
            'garch_models': garch_models,
            'Q_bar': Q_bar,
            'log_likelihood': best_ll,
            'parallel_fitting_time': elapsed
        }
    
    def _get_diagnostic_informed_starting_points(self):
        """Get DCC starting points informed by diagnostic results"""
        starting_points = [
            [0.01, 0.95],  # Conservative default
            [0.05, 0.90],  # Moderate
            [0.10, 0.85],  # More responsive
        ]
        
        if self.enable_diagnostics and self.diagnostic_results:
            # Check correlation structure
            if 'cointegration' in self.diagnostic_results:
                cointegrated_pairs = sum(1 for pair, result in self.diagnostic_results['cointegration'].items() 
                                       if result.get('cointegrated', False))
                total_pairs = len(self.diagnostic_results['cointegration'])
                
                if cointegrated_pairs > total_pairs * 0.5:
                    # High cointegration suggests more persistent correlations
                    starting_points.insert(0, [0.02, 0.97])
                    print("   → High cointegration detected, using more persistent DCC parameters")
            
            # Check volatility clustering strength
            strong_arch_assets = sum(1 for asset, diag in self.diagnostic_results.items() 
                                   if asset != 'cointegration' and 
                                   diag['volatility_clustering']['arch_test'].get('has_arch', False))
            
            if strong_arch_assets > len(self.diagnostic_results) * 0.7:
                # Strong volatility clustering suggests more responsive correlations
                starting_points.insert(0, [0.08, 0.88])
                print("   → Strong volatility clustering detected, using more responsive DCC parameters")
        
        return starting_points

    def _robust_correlation_matrix(self, residuals):
        """Compute robust unconditional correlation matrix"""
        Q_bar = np.cov(residuals.T)
        
        # Ensure positive definiteness using eigenvalue regularization
        eigenvals, eigenvecs = np.linalg.eigh(Q_bar)
        eigenvals = np.maximum(eigenvals, self.regularization)
        Q_bar = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Additional regularization
        Q_bar += self.regularization * np.eye(Q_bar.shape[0])
        
        return Q_bar

    def _compute_dynamic_correlations(self, std_residuals, Q_bar, a, b):
        """Compute time-varying correlation matrices"""
        T, N = std_residuals.shape
        Q_t = Q_bar.copy()
        R_t_list = []
        
        one_minus_ab = 1 - a - b
        
        for t in range(1, T):
            z_t = std_residuals.iloc[t - 1].values
            z_t = np.clip(z_t, -10, 10)
            
            Q_t = one_minus_ab * Q_bar + a * np.outer(z_t, z_t) + b * Q_t
            
            # Ensure positive definiteness
            eigenvals, eigenvecs = np.linalg.eigh(Q_t)
            eigenvals = np.maximum(eigenvals, self.regularization)
            Q_t = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Compute correlation matrix
            diag_sqrt_inv = np.diag(1 / np.sqrt(np.diag(Q_t)))
            R_t = diag_sqrt_inv @ Q_t @ diag_sqrt_inv
            
            # Ensure valid correlation matrix
            np.fill_diagonal(R_t, 1.0)
            R_t = np.clip(R_t, -0.999, 0.999)
            np.fill_diagonal(R_t, 1.0)
            
            R_t_list.append(R_t)

        return R_t_list

    def forecast_var(self, current_fit, tickers):
        """Enhanced VaR forecasting with diagnostics-based distribution selection for multiple alphas"""
        cond_vol_forecast = {}
        
        # Get volatility forecasts
        for asset in tickers:
            garch_model = current_fit['garch_models'][asset]
            
            try:
                if not garch_model.get('is_fallback', False) and 'model' in garch_model:
                    forecast = garch_model['model'].forecast(horizon=1)
                    cond_var = forecast.variance.iloc[-1].values[0]
                    cond_vol_forecast[asset] = np.sqrt(max(cond_var, 1e-8)) / self.scale_factor
                else:
                    if 'conditional_volatility' in garch_model:
                        cond_vol_forecast[asset] = garch_model['conditional_volatility'].iloc[-1]
                    else:
                        cond_vol_forecast[asset] = garch_model.get('volatility', 0.02)
                    
            except Exception as e:
                print(f"Volatility forecast failed for {asset}: {e}")
                cond_vol_forecast[asset] = 0.02

        # Get correlation matrix forecast
        if current_fit['R_t'] and len(current_fit['R_t']) > 0:
            last_R_t = current_fit['R_t'][-1].copy()
            
            try:
                eigenvals = np.linalg.eigvals(last_R_t)
                if np.any(eigenvals <= 0):
                    eigenvals_reg = np.maximum(eigenvals, self.regularization)
                    last_R_t = last_R_t + (eigenvals_reg.min() - eigenvals.min()) * np.eye(len(tickers))
                    
            except Exception as e:
                print(f"Correlation matrix validation failed: {e}")
                last_R_t = np.eye(len(tickers))
        else:
            last_R_t = np.eye(len(tickers))

        # Construct covariance matrix
        vol_vector = np.array([cond_vol_forecast[asset] for asset in tickers])
        D_t = np.diag(vol_vector)
        H_t = D_t @ last_R_t @ D_t
        
        # Portfolio variance
        portfolio_var = self.weights @ H_t @ self.weights
        portfolio_var = max(portfolio_var, 1e-8)
        portfolio_std = np.sqrt(portfolio_var)
        
        # VaR calculation for each alpha
        VaRs = {}
        for alpha in self.alphas:
            if self.dist == 't' and 'df_dict' in current_fit:
                df_values = [df for df in current_fit['df_dict'].values() 
                            if np.isfinite(df) and df > 2]
                if df_values:
                    avg_df = np.mean(df_values)
                    var_quantile = t.ppf(alpha, avg_df)
                else:
                    var_quantile = norm.ppf(alpha)
            else:
                var_quantile = norm.ppf(alpha)
            VaR = -var_quantile * portfolio_std
            VaRs[alpha] = VaR
        
        return VaRs, portfolio_std

    def backtest(self, train_returns, test_returns):
        """Enhanced backtesting with diagnostics integration and consistent windowing for multiple alphas"""
        # Check for sufficient test days for traffic light test
        if len(test_returns) < 250:
            print(f"Warning: Number of test days ({len(test_returns)}) is less than 250. Traffic light test may not be reliable.")
        
        # Run initial diagnostics on training data
        if self.enable_diagnostics:
            self.run_initial_diagnostics(train_returns)
        
        tickers = train_returns.columns.tolist()
        
        if len(tickers) != len(self.weights):
            raise ValueError(f"Mismatch: {len(tickers)} tickers vs {len(self.weights)} weights")

        # CONSISTENT WINDOWING: Use window_size from the very beginning
        print(f"\nApplying consistent windowing from start...")
        if len(train_returns) > self.window_size:
            print(f"Using last {self.window_size} days from {len(train_returns)} training days")
            window = train_returns.iloc[-self.window_size:].copy()  # Only last window_size days
        else:
            print(f"Training data ({len(train_returns)} days) smaller than window size ({self.window_size})")
            window = train_returns.copy()
        
        print(f"Initial window: {window.index[0]} to {window.index[-1]} ({len(window)} days)")

        # Initial model fit with consistent windowing
        print("Fitting initial DCC-GARCH model with consistent window size...")
        try:
            current_fit = self.fit_dcc_garch(window)
        except Exception as e:
            print(f"Initial model fit failed: {e}")
            raise

        VaR_forecasts = {alpha: [] for alpha in self.alphas}
        VaR_std_forecasts = []
        actual_portfolio_returns = []
        test_dates = test_returns.index
        refit_failures = 0

        print(f"Starting backtesting for {len(test_returns)} days with consistent {self.window_size}-day windows...")

        for day in range(len(test_returns)):
            if day % 50 == 0:
                print(f"Progress: {day}/{len(test_returns)} ({day/len(test_returns)*100:.1f}%) - Window: {len(window)} days")

            # Forecast VaR using current window
            try:
                VaRs, portfolio_std = self.forecast_var(current_fit, tickers)
            except Exception as e:
                print(f"VaR forecast failed on day {day}: {e}")
                portfolio_std = 0.02 * np.sqrt(np.sum(self.weights**2))
                VaRs = {alpha: -norm.ppf(alpha) * portfolio_std for alpha in self.alphas}
            
            for alpha in self.alphas:
                VaR_forecasts[alpha].append(VaRs[alpha])
            VaR_std_forecasts.append(portfolio_std)

            # Actual portfolio return
            actual_return = self.weights @ test_returns.iloc[day].values
            actual_portfolio_returns.append(actual_return)

            # Update rolling window with consistent size constraint
            new_day = test_returns.iloc[[day]]
            window = pd.concat([window, new_day])
            
            # CONSISTENT WINDOWING: Always maintain exact window size
            if len(window) > self.window_size:
                window = window.iloc[-self.window_size:]  # Keep exactly window_size days
                
            # Debug: Ensure window size consistency
            if day < 5 or day % 50 == 0:
                print(f"  Day {day}: Window size = {len(window)}, Range: {window.index[0]} to {window.index[-1]}")
            
            # Periodic model refitting with consistent windowing
            if day % self.refit_frequency == 0 and day > 0:
                try:
                    print(f"Refitting model at day {day} with {len(window)}-day window...")
                    current_fit = self.fit_dcc_garch(
                        window, 
                        init_dcc_params=[current_fit['a'], current_fit['b']]
                    )
                    print(f"  Refit successful: Window {window.index[0]} to {window.index[-1]}")
                except Exception as e:
                    print(f"Model refit failed on day {day}: {e}")
                    refit_failures += 1

        print(f"Backtesting completed with consistent windowing. Refit failures: {refit_failures}")

        return {
            'VaR_forecasts': {alpha: np.array(VaR_forecasts[alpha]) for alpha in self.alphas},
            'VaR_std_forecasts': np.array(VaR_std_forecasts),
            'actual_returns': np.array(actual_portfolio_returns),
            'test_dates': test_dates,
            'refit_failures': refit_failures,
            'diagnostic_results': self.diagnostic_results,
            'consistent_window_size': self.window_size,
            'total_days_used': len(window)
        }

    def traffic_light_zone(self, alpha, exceptions, n_obs):
        """Determine traffic light zone based on number of exceptions using Basel II thresholds"""
        p = alpha
        green_threshold = binom.ppf(0.95, n_obs, p)
        yellow_threshold = binom.ppf(0.9999, n_obs, p)
        if exceptions <= green_threshold:
            return 'Green'
        elif exceptions <= yellow_threshold:
            return 'Yellow'
        else:
            return 'Red'

    def evaluate_backtest(self, results):
        """Enhanced backtest evaluation with diagnostic insights and multiple alphas"""
        actual_returns = results['actual_returns']
        evaluation = {}
        
        for alpha in self.alphas:
            VaR_forecasts_alpha = results['VaR_forecasts'][alpha]
            exceptions_alpha = np.sum(actual_returns < -VaR_forecasts_alpha)
            exception_rate_alpha = exceptions_alpha / len(actual_returns)
            expected_rate = alpha
            
            # Kupiec unconditional coverage test
            if exceptions_alpha > 0 and exception_rate_alpha < 1:
                LR_uc = 2 * (
                    exceptions_alpha * np.log(exception_rate_alpha / expected_rate) + 
                    (len(actual_returns) - exceptions_alpha) * np.log((1 - exception_rate_alpha) / (1 - expected_rate))
                )
                from scipy.stats import chi2
                p_value_uc = 1 - chi2.cdf(LR_uc, 1)
            else:
                LR_uc = 0
                p_value_uc = 1.0 if exceptions_alpha == 0 else 0.0
            
            # Traffic light zone
            zone = self.traffic_light_zone(alpha, exceptions_alpha, len(actual_returns))
            
            # VaR efficiency measures
            exceedances = actual_returns[actual_returns < -VaR_forecasts_alpha]
            avg_exceedance = np.mean(-exceedances - VaR_forecasts_alpha[actual_returns < -VaR_forecasts_alpha]) if len(exceedances) > 0 else 0
            max_exceedance = np.max(-exceedances - VaR_forecasts_alpha[actual_returns < -VaR_forecasts_alpha]) if len(exceedances) > 0 else 0
            
            evaluation[alpha] = {
                'exceptions': exceptions_alpha,
                'exception_rate': exception_rate_alpha,
                'expected_rate': expected_rate,
                'LR_uc_stat': LR_uc,
                'LR_uc_p_value': p_value_uc,
                'traffic_light_zone': zone,
                'coverage_test': 'PASS' if p_value_uc > 0.05 else 'FAIL',
                'avg_exceedance': avg_exceedance,
                'max_exceedance': max_exceedance
            }
        
        # Additional statistics
        avg_var = {alpha: np.mean(results['VaR_forecasts'][alpha]) for alpha in self.alphas}
        avg_return = np.mean(actual_returns)
        return_vol = np.std(actual_returns)
        
        # Diagnostic-informed evaluation
        diagnostic_insights = self._generate_diagnostic_insights(results)
        
        results_dict = {
            'evaluation_by_alpha': evaluation,
            'avg_var': avg_var,
            'avg_return': avg_return,
            'return_volatility': return_vol,
            'refit_failures': results.get('refit_failures', 0),
            'diagnostic_insights': diagnostic_insights
        }
        
        return results_dict

    def _generate_diagnostic_insights(self, results):
        """Generate insights based on diagnostic results and backtest performance"""
        insights = []
        
        if not self.enable_diagnostics or not self.diagnostic_results:
            return ["Diagnostics not enabled - limited insights available"]
        
        # Distribution insights
        t_dist_count = sum(1 for asset, diag in self.diagnostic_results.items() 
                          if asset != 'cointegration' and 
                          diag['distribution']['recommended_dist'] == 't')
        
        if t_dist_count > 0:
            insights.append(f"Fat-tail distributions recommended for {t_dist_count} assets - t-distribution usage appropriate")
        
        # Volatility clustering insights
        arch_assets = sum(1 for asset, diag in self.diagnostic_results.items() 
                         if asset != 'cointegration' and 
                         diag['volatility_clustering']['arch_test'].get('has_arch', False))
        
        if arch_assets > 0:
            insights.append(f"Volatility clustering detected in {arch_assets} assets - GARCH modeling justified")
        
        # Outlier insights
        high_outlier_assets = sum(1 for asset, diag in self.diagnostic_results.items() 
                                 if asset != 'cointegration' and 
                                 diag['outliers']['percentage_3sigma'] > 5)
        
        if high_outlier_assets > 0:
            insights.append(f"High outlier percentage in {high_outlier_assets} assets - robust methods beneficial")
        
        # Cointegration insights
        if 'cointegration' in self.diagnostic_results:
            cointegrated_pairs = sum(1 for result in self.diagnostic_results['cointegration'].values() 
                                   if result.get('cointegrated', False))
            if cointegrated_pairs > 0:
                insights.append(f"Found {cointegrated_pairs} cointegrated asset pairs - diversification benefits may be limited")
        
        return insights
    
    def plot_results(self, results, title="Enhanced Portfolio VaR Analysis with Diagnostics"):
        """Enhanced visualization with proper diagnostic plots and larger size for multiple alphas"""
        test_dates = results['test_dates']
        actual_returns = results['actual_returns']

        # Step 1: Create a separate MUCH LARGER figure for VaR plots
        if len(self.alphas) > 1:
            fig_var, axes_var = plt.subplots(len(self.alphas), 1, figsize=(20, 8 * len(self.alphas)), sharex=True)
            # Ensure axes_var is iterable even for a single subplot
            if len(self.alphas) == 1:
                axes_var = [axes_var]
            
            for i, alpha in enumerate(self.alphas):
                ax = axes_var[i]
                ax.plot(test_dates, actual_returns, label='Portfolio Returns', alpha=0.7, linewidth=1, color='blue')
                ax.plot(test_dates, -results['VaR_forecasts'][alpha], label=f'{int((1 - alpha) * 100)}% VaR', color='red', linewidth=2)
                
                exceptions_mask = actual_returns < -results['VaR_forecasts'][alpha]
                if np.any(exceptions_mask):
                    ax.scatter(test_dates[exceptions_mask], actual_returns[exceptions_mask], 
                            color='red', s=40, alpha=0.8, label='VaR Breaches', zorder=5)
                
                # Add statistics text box
                n_breaches = np.sum(exceptions_mask)
                breach_rate = n_breaches / len(actual_returns) * 100
                expected_rate = alpha * 100
                
                stats_text = f'Breaches: {n_breaches}\nBreach Rate: {breach_rate:.2f}%\nExpected: {expected_rate:.1f}%'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                
                ax.set_title(f'Portfolio Returns vs {int((1 - alpha) * 100)}% VaR', fontsize=14, fontweight='bold', pad=15)
                ax.set_ylabel('Return', fontsize=12)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
            axes_var[-1].set_xlabel('Date', fontsize=12)
            fig_var.suptitle('VaR Forecasts vs Actual Returns', fontsize=16, fontweight='bold', y=0.98)
            fig_var.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            # Single alpha plot - larger size
            alpha = self.alphas[0]
            fig_var, ax = plt.subplots(figsize=(20, 10))
            ax.plot(test_dates, actual_returns, label='Portfolio Returns', alpha=0.7, linewidth=1, color='blue')
            ax.plot(test_dates, -results['VaR_forecasts'][alpha], label=f'{int((1 - alpha) * 100)}% VaR', color='red', linewidth=2)
            
            exceptions_mask = actual_returns < -results['VaR_forecasts'][alpha]
            if np.any(exceptions_mask):
                ax.scatter(test_dates[exceptions_mask], actual_returns[exceptions_mask], 
                        color='red', s=40, alpha=0.8, label='VaR Breaches', zorder=5)
            
            # Add statistics
            n_breaches = np.sum(exceptions_mask)
            breach_rate = n_breaches / len(actual_returns) * 100
            expected_rate = alpha * 100
            
            stats_text = f'Breaches: {n_breaches}\nBreach Rate: {breach_rate:.2f}%\nExpected: {expected_rate:.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            ax.set_title(f'Portfolio Returns vs {int((1 - alpha) * 100)}% VaR', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Return', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            fig_var.tight_layout()
            plt.show()

        # Step 2: Create the diagnostic plots figure - MUCH LARGER
        fig_diag, axes_diag = plt.subplots(2, 3, figsize=(30, 20))

        # Plot 1: Portfolio return autocorrelation
        ax1 = axes_diag[0, 0]
        from statsmodels.tsa.stattools import acf
        try:
            lags = min(20, len(actual_returns) // 4)
            autocorr = acf(actual_returns, nlags=lags, fft=True)
            ax1.bar(range(len(autocorr)), autocorr, alpha=0.7, color='steelblue')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.axhline(y=1.96/np.sqrt(len(actual_returns)), color='red', linestyle='--', alpha=0.7, label='95% Confidence')
            ax1.axhline(y=-1.96/np.sqrt(len(actual_returns)), color='red', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Lag', fontsize=12)
            ax1.set_ylabel('Autocorrelation', fontsize=12)
            ax1.set_title('Portfolio Return Autocorrelation', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        except Exception as e:
            ax1.text(0.5, 0.5, f'Autocorrelation plot failed:\n{str(e)}', ha='center', va='center', fontsize=10)
            ax1.set_title('Portfolio Return Autocorrelation (Failed)', fontsize=14)

        # Plot 2: Return distribution with VaR overlays
        ax2 = axes_diag[0, 1]
        ax2.hist(actual_returns, bins=50, alpha=0.7, density=True, color='lightblue', 
                edgecolor='black', label='Actual Returns')
        
        # Overlay normal distribution for comparison
        x_range = np.linspace(actual_returns.min(), actual_returns.max(), 100)
        normal_pdf = norm.pdf(x_range, actual_returns.mean(), actual_returns.std())
        ax2.plot(x_range, normal_pdf, 'g--', linewidth=2, label='Normal Distribution')
        
        # Add VaR lines
        for alpha in self.alphas:
            avg_var = -np.mean(results['VaR_forecasts'][alpha])
            ax2.axvline(avg_var, linestyle='--', linewidth=2,
                        label=f'Avg {int((1 - alpha) * 100)}% VaR: {avg_var:.4f}')
        
        ax2.set_xlabel('Return', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Return Distribution vs VaR Thresholds', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Traffic Light Test Results
        ax3 = axes_diag[0, 2]
        if hasattr(self, 'traffic_light_zone'):
            zones = []
            alphas_pct = []
            breach_counts = []
            
            for alpha in self.alphas:
                exceptions = np.sum(actual_returns < -results['VaR_forecasts'][alpha])
                zone = self.traffic_light_zone(alpha, exceptions, len(actual_returns))
                zones.append(zone)
                alphas_pct.append(f"{int((1-alpha)*100)}%")
                breach_counts.append(exceptions)
            
            # Color mapping
            colors = ['green' if z == 'Green' else 'yellow' if z == 'Yellow' else 'red' for z in zones]
            bars = ax3.bar(alphas_pct, breach_counts, color=colors, alpha=0.7, edgecolor='black')
            
            # Add expected breach lines
            expected_breaches = [alpha * len(actual_returns) for alpha in self.alphas]
            ax3.plot(alphas_pct, expected_breaches, 'ko-', linewidth=2, markersize=8, label='Expected Breaches')
            
            # Add text annotations
            for i, (bar, zone, count) in enumerate(zip(bars, zones, breach_counts)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{zone}\n({count})', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax3.set_xlabel('VaR Confidence Level', fontsize=12)
            ax3.set_ylabel('Number of Breaches', fontsize=12)
            ax3.set_title('Traffic Light Test Results', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Traffic Light Test\nNot Available', ha='center', va='center', fontsize=14)
            ax3.set_title('Traffic Light Test Results', fontsize=14, fontweight='bold')

        # Plot 4: Rolling exception rate
        ax4 = axes_diag[1, 0]
        window_size = min(50, len(actual_returns) // 4)
        
        # Use the most conservative alpha (smallest value) for rolling analysis
        alpha_conservative = min(self.alphas)
        exceptions_mask = actual_returns < -results['VaR_forecasts'][alpha_conservative]
        
        try:
            rolling_exceptions = pd.Series(exceptions_mask.astype(int)).rolling(window_size, min_periods=1).mean()
            ax4.plot(test_dates, rolling_exceptions, linewidth=2, 
                    label=f'{window_size}-day Rolling Exception Rate ({int((1 - alpha_conservative) * 100)}%)')
            ax4.axhline(alpha_conservative, color='red', linestyle='--', linewidth=2, 
                        label=f'Expected Rate: {alpha_conservative:.3f}')
            
            # Add confidence bands
            conf_upper = alpha_conservative + 1.96 * np.sqrt(alpha_conservative * (1 - alpha_conservative) / window_size)
            conf_lower = max(0, alpha_conservative - 1.96 * np.sqrt(alpha_conservative * (1 - alpha_conservative) / window_size))
            ax4.axhspan(conf_lower, conf_upper, alpha=0.2, color='red', label='95% Confidence Band')
            
            ax4.set_xlabel('Date', fontsize=12)
            ax4.set_ylabel('Exception Rate', fontsize=12)
            ax4.set_title(f'Rolling {window_size}-Day Exception Rate Analysis', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        except Exception as e:
            ax4.text(0.5, 0.5, f'Rolling analysis failed:\n{str(e)}', ha='center', va='center', fontsize=10)
            ax4.set_title('Rolling Exception Rate (Failed)', fontsize=14)

        # Plot 5: VaR time series comparison
        ax5 = axes_diag[1, 1]
        for alpha in self.alphas:
            ax5.plot(test_dates, results['VaR_forecasts'][alpha], linewidth=2,
                    label=f'{int((1 - alpha) * 100)}% VaR')
        
        # Add portfolio volatility
        if 'VaR_std_forecasts' in results:
            # Scale volatility for visualization (multiply by quantile approximation)
            scaled_vol = results['VaR_std_forecasts'] * 1.96  # Approximate 95% quantile
            ax5.plot(test_dates, scaled_vol, '--', alpha=0.7, color='gray', 
                    label='Scaled Portfolio Volatility (×1.96)')
        
        ax5.set_xlabel('Date', fontsize=12)
        ax5.set_ylabel('VaR Value', fontsize=12)
        ax5.set_title('VaR Forecasts Time Series', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Plot 6: Enhanced diagnostic insights
        ax6 = axes_diag[1, 2]
        
        # Build comprehensive diagnostic text
        diagnostic_text = "DIAGNOSTIC INSIGHTS:\n" + "="*25 + "\n\n"
        
        # Basic statistics
        diagnostic_text += f"PORTFOLIO STATISTICS:\n"
        diagnostic_text += f"• Test period: {len(actual_returns)} days\n"
        diagnostic_text += f"• Mean return: {np.mean(actual_returns):.4f}\n"
        diagnostic_text += f"• Volatility: {np.std(actual_returns):.4f}\n"
        diagnostic_text += f"• Skewness: {pd.Series(actual_returns).skew():.3f}\n"
        diagnostic_text += f"• Kurtosis: {pd.Series(actual_returns).kurtosis():.3f}\n\n"
        
        # VaR performance summary
        diagnostic_text += f"VaR PERFORMANCE:\n"
        for alpha in self.alphas:
            exceptions = np.sum(actual_returns < -results['VaR_forecasts'][alpha])
            breach_rate = exceptions / len(actual_returns)
            diagnostic_text += f"• {int((1-alpha)*100)}% VaR: {exceptions} breaches ({breach_rate:.2%})\n"
        
        diagnostic_text += f"\n"
        
        # Model-specific insights
        if hasattr(self, 'diagnostic_results') and self.diagnostic_results:
            diagnostic_text += f"DATA CHARACTERISTICS:\n"
            
            # Count distribution recommendations
            dist_counts = {}
            arch_count = 0
            for asset, diag in self.diagnostic_results.items():
                if asset != 'cointegration':
                    rec_dist = diag['distribution']['recommended_dist']
                    dist_counts[rec_dist] = dist_counts.get(rec_dist, 0) + 1
                    if diag['volatility_clustering']['arch_test'].get('has_arch', False):
                        arch_count += 1
            
            for dist, count in dist_counts.items():
                diagnostic_text += f"• {count} assets → {dist.upper()} distribution\n"
            
            diagnostic_text += f"• {arch_count} assets show volatility clustering\n"
            
            if 'cointegration' in self.diagnostic_results:
                cointegrated = sum(1 for r in self.diagnostic_results['cointegration'].values() 
                                if r.get('cointegrated', False))
                diagnostic_text += f"• {cointegrated} cointegrated asset pairs found\n"
        
        # Add model performance metrics
        if 'refit_failures' in results:
            diagnostic_text += f"\nMODEL PERFORMANCE:\n"
            diagnostic_text += f"• Refit failures: {results['refit_failures']}\n"
            
        if 'consistent_window_size' in results:
            diagnostic_text += f"• Window size: {results['consistent_window_size']} days\n"
        
        # Performance assessment
        total_breaches = sum(np.sum(actual_returns < -results['VaR_forecasts'][alpha]) for alpha in self.alphas)
        if total_breaches == 0:
            diagnostic_text += f"\n⚠️  No breaches detected - model may be too conservative"
        elif any(np.sum(actual_returns < -results['VaR_forecasts'][alpha]) / len(actual_returns) > alpha * 2 
                for alpha in self.alphas):
            diagnostic_text += f"\n⚠️  High breach rates detected - model may need recalibration"
        else:
            diagnostic_text += f"\n✅ Model performance within acceptable ranges"
        
        # Use the full plot area and remove axes for text display
        ax6.text(0.02, 0.98, diagnostic_text, transform=ax6.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Set title and remove axes ticks but keep the frame
        ax6.set_title('Diagnostic Summary & Model Performance', fontsize=14, fontweight='bold')
        ax6.set_xticks([])
        ax6.set_yticks([])
        
        # Keep the frame visible like other plots
        for spine in ax6.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)

        # Adjust layout and display
        fig_diag.suptitle(title, fontsize=18, fontweight='bold')
        fig_diag.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        print(f"\n📊 Plotting completed:")
        print(f"   • VaR plots: {len(self.alphas)} confidence levels")
        print(f"   • Diagnostic plots: 6 comprehensive analyses")
        print(f"   • Total test days: {len(actual_returns)}")
        print(f"   • Portfolio assets: {len(self.weights)}")

        
    def generate_comprehensive_report(self, results, evaluation):
        """Generate a comprehensive report including diagnostics and windowing information for multiple alphas"""
        print("\n" + "="*80)
        print("COMPREHENSIVE VaR ANALYSIS REPORT WITH DIAGNOSTICS")
        print("="*80)
        
        # Window size and performance metrics
        print(f"\n⚡ MODELING SETUP & PERFORMANCE:")
        print("-" * 40)
        if 'consistent_window_size' in results:
            print(f"Consistent window size: {results['consistent_window_size']} days")
            print(f"Total training days available: {results.get('total_days_used', 'N/A')}")
            print(f"Window consistency: ENABLED ✅")
        else:
            print(f"Window consistency: DISABLED ⚠️")
            
        if 'parallel_fitting_time' in results:
            print(f"Parallel GARCH fitting time: {results['parallel_fitting_time']:.2f} seconds")
            n_assets = len([k for k in results.get('diagnostic_results', {}).keys() if k != 'cointegration'])
            if n_assets > 0:
                print(f"Average time per asset: {results['parallel_fitting_time']/n_assets:.2f} seconds")
                print(f"Assets processed in parallel: {n_assets}")
        
        if self.enable_diagnostics and self.diagnostic_results:
            print("\n📊 DATA DIAGNOSTICS SUMMARY:")
            print("-" * 40)
            
            for asset, diag in self.diagnostic_results.items():
                if asset != 'cointegration':
                    print(f"\n{asset}:")
                    print(f"  • Distribution: {diag['distribution']['recommended_dist'].upper()}")
                    print(f"  • Volatility Clustering: {'Yes' if diag['volatility_clustering']['arch_test'].get('has_arch', False) else 'No'}")
                    print(f"  • Stationarity: {diag['stationarity']['conclusion']}")
                    print(f"  • Outliers (3σ): {diag['outliers']['count_3sigma']} ({diag['outliers']['percentage_3sigma']:.1f}%)")
        
        # Standard backtest evaluation for each alpha
        print(f"\n📈 VaR BACKTEST RESULTS:")
        print("-" * 40)
        for alpha in self.alphas:
            eval_alpha = evaluation['evaluation_by_alpha'][alpha]
            print(f"\nResults for {int((1-alpha)*100)}% VaR (alpha={alpha}):")
            print(f"  Exception rate: {eval_alpha['exception_rate']:.4f} ({eval_alpha['exception_rate']*100:.2f}%)")
            print(f"  Expected rate: {eval_alpha['expected_rate']:.4f} ({eval_alpha['expected_rate']*100:.2f}%)")
            print(f"  Coverage test: {eval_alpha['coverage_test']}")
            print(f"  Traffic light zone: {eval_alpha['traffic_light_zone']}")
            print(f"  Average VaR: {evaluation['avg_var'][alpha]:.4f}")
            print(f"  Average exceedance: {eval_alpha['avg_exceedance']:.4f}")
            print(f"  Maximum exceedance: {eval_alpha['max_exceedance']:.4f}")
        
        print(f"Model Refit Failures: {evaluation['refit_failures']}")
        
        # Diagnostic insights
        if 'diagnostic_insights' in evaluation:
            print(f"\n🔍 DIAGNOSTIC INSIGHTS:")
            print("-" * 40)
            for insight in evaluation['diagnostic_insights']:
                print(f"  • {insight}")
        
        print("\n" + "="*80)