import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


import numpy as np
import pandas as pd
from scipy.stats import shapiro, jarque_bera, normaltest
from statsmodels.tsa.stattools import adfuller, kpss, coint, acf
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')


class DataDiagnostics:
    """
    Comprehensive financial data diagnostics class for time series analysis.
    """
    
    def __init__(self, dist='auto'):
        self.dist = dist
        self.data_diagnostics = {}
        self.distribution_selection = {}
    
    def run_data_diagnostics(self, returns_data):
        """Comprehensive data diagnostics including stationarity tests and distribution analysis"""
        print("Running comprehensive data diagnostics...")
        print("=" * 60)
        
        diagnostics = {}
        
        for asset in returns_data.columns:
            print(f"\nAnalyzing {asset}:")
            print("-" * 40)
            
            asset_returns = returns_data[asset].dropna()
            asset_diagnostics = {}
            
            # 1. BASIC STATISTICS
            basic_stats = {
                'count': len(asset_returns),
                'mean': asset_returns.mean(),
                'std': asset_returns.std(),
                'skewness': asset_returns.skew(),
                'kurtosis': asset_returns.kurtosis(),
                'min': asset_returns.min(),
                'max': asset_returns.max(),
                'q25': asset_returns.quantile(0.25),
                'q75': asset_returns.quantile(0.75)
            }
            
            print(f"Basic Statistics:")
            print(f"  Observations: {basic_stats['count']}")
            print(f"  Mean: {basic_stats['mean']:.6f}")
            print(f"  Std Dev: {basic_stats['std']:.6f}")
            print(f"  Skewness: {basic_stats['skewness']:.4f}")
            print(f"  Kurtosis: {basic_stats['kurtosis']:.4f}")
            
            asset_diagnostics['basic_stats'] = basic_stats
            
            # 2. STATIONARITY TESTS
            stationarity_tests = self._run_stationarity_tests(asset_returns)
            asset_diagnostics['stationarity'] = stationarity_tests
            
            print(f"\nStationarity Tests:")
            print(f"  ADF Test: {'Stationary' if stationarity_tests['adf']['stationary'] else 'Non-stationary'} (p-value: {stationarity_tests['adf']['p_value']:.4f})")
            print(f"  KPSS Test: {'Stationary' if stationarity_tests['kpss']['stationary'] else 'Non-stationary'} (p-value: {stationarity_tests['kpss']['p_value']:.4f})")
            print(f"  Overall: {stationarity_tests['conclusion']}")
            
            # 3. DISTRIBUTION ANALYSIS
            distribution_analysis = self._analyze_distribution(asset_returns)
            asset_diagnostics['distribution'] = distribution_analysis
            
            print(f"\nDistribution Analysis:")
            print(f"  Normality (Shapiro-Wilk): {'Normal' if distribution_analysis['normality']['shapiro']['normal'] else 'Non-normal'} (p-value: {distribution_analysis['normality']['shapiro']['p_value']:.4f})")
            print(f"  Normality (Jarque-Bera): {'Normal' if distribution_analysis['normality']['jarque_bera']['normal'] else 'Non-normal'} (p-value: {distribution_analysis['normality']['jarque_bera']['p_value']:.4f})")
            print(f"  Recommended Distribution: {distribution_analysis['recommended_dist']}")
            
            # 4. VOLATILITY CLUSTERING
            vol_clustering = self._test_volatility_clustering(asset_returns)
            asset_diagnostics['volatility_clustering'] = vol_clustering
            
            print(f"\nVolatility Clustering:")
            print(f"  ARCH Test: {'Present' if vol_clustering['arch_test']['has_arch'] else 'Not Present'} (p-value: {vol_clustering['arch_test']['p_value']:.4f})")
            print(f"  Ljung-Box on Squared Returns: {'Present' if vol_clustering['ljung_box']['has_clustering'] else 'Not Present'} (p-value: {vol_clustering['ljung_box']['p_value']:.4f})")
            
            # 5. OUTLIER DETECTION
            outliers = self._detect_outliers(asset_returns)
            asset_diagnostics['outliers'] = outliers
            
            print(f"\nOutlier Detection:")
            print(f"  Outliers (3-sigma): {outliers['count_3sigma']} ({outliers['percentage_3sigma']:.2f}%)")
            print(f"  Extreme Outliers (4-sigma): {outliers['count_4sigma']} ({outliers['percentage_4sigma']:.2f}%)")
            
            diagnostics[asset] = asset_diagnostics
        
        # 6. COINTEGRATION ANALYSIS (if multiple assets)
        if len(returns_data.columns) > 1:
            print(f"\nCointegration Analysis:")
            print("-" * 40)
            
            cointegration_results = self._test_cointegration(returns_data)
            diagnostics['cointegration'] = cointegration_results
            
            print(f"Engle-Granger Cointegration Test:")
            for pair, result in cointegration_results.items():
                print(f"  {pair}: {'Cointegrated' if result['cointegrated'] else 'Not Cointegrated'} (p-value: {result['p_value']:.4f})")
        
        self.data_diagnostics = diagnostics
        return diagnostics
    
    def _run_stationarity_tests(self, series):
        """Run comprehensive stationarity tests"""
        tests = {}
        
        # Augmented Dickey-Fuller Test
        try:
            adf_result = adfuller(series, autolag='AIC')
            tests['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'stationary': adf_result[1] < 0.05
            }
        except Exception as e:
            tests['adf'] = {'error': str(e), 'stationary': False}
        
        # KPSS Test
        try:
            kpss_result = kpss(series, regression='c', nlags='auto')
            tests['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'stationary': kpss_result[1] > 0.05
            }
        except Exception as e:
            tests['kpss'] = {'error': str(e), 'stationary': False}
        
        # Combined conclusion
        if tests['adf'].get('stationary', False) and tests['kpss'].get('stationary', False):
            conclusion = "Stationary"
        elif not tests['adf'].get('stationary', False) and not tests['kpss'].get('stationary', False):
            conclusion = "Non-stationary"
        else:
            conclusion = "Uncertain - conflicting results"
        
        tests['conclusion'] = conclusion
        return tests
    
    def _analyze_distribution(self, series):
        """Analyze distribution characteristics and recommend appropriate distribution"""
        analysis = {}
        
        # Normality tests
        normality_tests = {}
        
        # Shapiro-Wilk test (better for small samples)
        try:
            if len(series) <= 5000:
                shapiro_stat, shapiro_p = shapiro(series)
                normality_tests['shapiro'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'normal': shapiro_p > 0.05
                }
            else:
                normality_tests['shapiro'] = {'note': 'Sample too large for Shapiro-Wilk test'}
        except Exception as e:
            normality_tests['shapiro'] = {'error': str(e)}
        
        # Jarque-Bera test
        try:
            jb_stat, jb_p = jarque_bera(series)
            normality_tests['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'normal': jb_p > 0.05
            }
        except Exception as e:
            normality_tests['jarque_bera'] = {'error': str(e)}
        
        # D'Agostino normality test
        try:
            da_stat, da_p = normaltest(series)
            normality_tests['dagostino'] = {
                'statistic': da_stat,
                'p_value': da_p,
                'normal': da_p > 0.05
            }
        except Exception as e:
            normality_tests['dagostino'] = {'error': str(e)}
        
        analysis['normality'] = normality_tests
        
        # Distribution characteristics
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        analysis['skewness'] = skewness
        analysis['kurtosis'] = kurtosis
        analysis['excess_kurtosis'] = kurtosis
        
        # Recommend distribution based on characteristics
        recommended_dist = self._recommend_distribution(skewness, kurtosis, normality_tests)
        analysis['recommended_dist'] = recommended_dist
        
        return analysis
    
    def _recommend_distribution(self, skewness, kurtosis, normality_tests):
        """Recommend distribution based on data characteristics"""
        # Check normality first
        is_normal = False
        if 'jarque_bera' in normality_tests and normality_tests['jarque_bera'].get('normal', False):
            is_normal = True
        elif 'shapiro' in normality_tests and normality_tests['shapiro'].get('normal', False):
            is_normal = True
        
        # Decision logic
        if is_normal and abs(skewness) < 0.5 and abs(kurtosis) < 1:
            return 'normal'
        elif abs(skewness) > 1.0:
            return 'skewed'
        elif kurtosis > 2:  # Fat tails
            return 't'
        elif abs(skewness) > 0.5:
            return 'skewed'
        else:
            return 't'  # Default to t-distribution for financial data
    
    def _test_volatility_clustering(self, series):
        """Test for volatility clustering (ARCH effects)"""
        tests = {}
        
        # ARCH test
        try:
            lm_stat, lm_p, f_stat, f_p = het_arch(series, nlags=5)
            tests['arch_test'] = {
                'lm_statistic': lm_stat,
                'lm_p_value': lm_p,
                'f_statistic': f_stat,
                'f_p_value': f_p,
                'has_arch': lm_p < 0.05,
                'p_value': lm_p
            }
        except Exception as e:
            tests['arch_test'] = {'error': str(e), 'has_arch': True}
        
        # Ljung-Box test on squared returns
        try:
            squared_returns = series ** 2
            ljung_box_result = acorr_ljungbox(squared_returns, lags=10, return_df=True)
            p_value = ljung_box_result['lb_pvalue'].iloc[-1]
            tests['ljung_box'] = {
                'statistic': ljung_box_result['lb_stat'].iloc[-1],
                'p_value': p_value,
                'has_clustering': p_value < 0.05
            }
        except Exception as e:
            tests['ljung_box'] = {'error': str(e), 'has_clustering': True}
        
        return tests
    
    def _detect_outliers(self, series):
        """Detect outliers using statistical methods"""
        mean = series.mean()
        std = series.std()
        
        # 3-sigma rule
        outliers_3sigma = np.abs(series - mean) > 3 * std
        count_3sigma = outliers_3sigma.sum()
        percentage_3sigma = (count_3sigma / len(series)) * 100
        
        # 4-sigma rule (extreme outliers)
        outliers_4sigma = np.abs(series - mean) > 4 * std
        count_4sigma = outliers_4sigma.sum()
        percentage_4sigma = (count_4sigma / len(series)) * 100
        
        return {
            'count_3sigma': count_3sigma,
            'percentage_3sigma': percentage_3sigma,
            'count_4sigma': count_4sigma,
            'percentage_4sigma': percentage_4sigma,
            'outlier_indices_3sigma': series[outliers_3sigma].index.tolist(),
            'outlier_indices_4sigma': series[outliers_4sigma].index.tolist()
        }
    
    def _test_cointegration(self, returns_data):
        """Test for cointegration between assets"""
        cointegration_results = {}
        
        # Convert returns to prices for cointegration test
        prices = (1 + returns_data).cumprod()
        
        assets = list(returns_data.columns)
        
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                pair = f"{assets[i]}-{assets[j]}"
                
                try:
                    coin_result = coint(prices[assets[i]], prices[assets[j]])
                    cointegration_results[pair] = {
                        'statistic': coin_result[0],
                        'p_value': coin_result[1],
                        'critical_values': coin_result[2],
                        'cointegrated': coin_result[1] < 0.05
                    }
                except Exception as e:
                    cointegration_results[pair] = {
                        'error': str(e),
                        'cointegrated': False
                    }
        
        return cointegration_results
    
    def select_distribution(self, asset_name):
        """Select appropriate distribution for an asset based on diagnostics"""
        if self.dist != 'auto':
            return self.dist
        
        if asset_name in self.data_diagnostics:
            recommended = self.data_diagnostics[asset_name]['distribution']['recommended_dist']
            self.distribution_selection[asset_name] = recommended
            return recommended
        else:
            return 't'