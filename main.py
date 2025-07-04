
import numpy as np
import pandas as pd
from src.dcc_garch_var import DCCGARCHVAR

if __name__ == "__main__":
    print("Integrated DCC-GARCH VaR Model with Data Diagnostics")
    print("="*60)
    
    # Try to load  data first
    try:
        stock_data = pd.read_csv('Stock_data.csv')
        stock_data = stock_data.drop("Script", axis=1, errors='ignore')
        stock_data['Date'] = pd.to_datetime(stock_data["Date"], format='mixed')
        stock_data.set_index('Date', inplace=True)
        
        # Calculate log returns
        stock_returns = np.log(stock_data / stock_data.shift(1))
        stock_returns.dropna(inplace=True)
        stock_returns = stock_returns.sort_index()
        
        print(f"Loaded  data with {len(stock_returns)} observations")
        print(f"Assets: {list(stock_returns.columns)}")
        
    except Exception:
        print(" data file not found or invalid. Generating synthetic data...")
        np.random.seed(42)
        n_days = 1500
        n_assets = 3
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        base_vol = np.array([0.015, 0.018, 0.020])
        base_corr = np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.5], [0.4, 0.5, 1.0]])
        returns_data = []
        current_vol = base_vol.copy()
        for t in range(n_days):
            if t > 0:
                last = np.array(returns_data[-1])
                shock = 0.1 * (last**2) / (current_vol**2)
                current_vol = 0.95 * current_vol + 0.05 * base_vol + shock * base_vol
            mvn = np.random.multivariate_normal([0]*n_assets, base_corr)
            returns_data.append(mvn * current_vol)
        stock_returns = pd.DataFrame(returns_data, index=dates, 
                                     columns=['TECH_STOCK','FINANCIAL_STOCK','ENERGY_STOCK'])
        print(f"Generated synthetic data with {len(stock_returns)} observations")
        print(f"Assets: {list(stock_returns.columns)}")

    # Portfolio weights
    if len(stock_returns.columns) == 3:
        weights = [0.4, 0.35, 0.25]
    else:
        n = len(stock_returns.columns)
        weights = [1/n] * n
    print(f"Portfolio weights: {dict(zip(stock_returns.columns, weights))}")

    # Train-test split
    split = int(0.8 * len(stock_returns))
    train = stock_returns.iloc[:split]
    test = stock_returns.iloc[split:]
    print(f"Training: {train.index[0]} to {train.index[-1]} ({len(train)} days)")
    print(f"Testing: {test.index[0]} to {test.index[-1]} ({len(test)} days)")

    # Summary stats
    print("\nTraining summary:")
    print(train.mean().round(6))
    print(train.std().round(4))
    print(train.corr().round(3))

    # Initialize model
    var_model = DCCGARCHVAR(
        weights=weights,
        alphas=[0.01,0.05,0.1],
        window_size=50,
        dist='auto',
        refit_frequency=2,
        enable_diagnostics=True,
        n_threads=None
    )
    print(f"Model window size: {var_model.window_size}")

    # Backtest
    print("\nStarting backtest...")
    results = var_model.backtest(train, test)
    evaluation = var_model.evaluate_backtest(results)
    var_model.generate_comprehensive_report(results, evaluation)
    print("\n📊 Coverage stats:")
    for alpha in var_model.alphas:
        e = evaluation['evaluation_by_alpha'][alpha]
        print(f"{int((1-alpha)*100)}% VaR: breaches={e['exceptions']}, rate={e['exception_rate']:.2%}, zone={e['traffic_light_zone']}, p={e['LR_uc_p_value']:.4f}")

    # Plot
    print("\nPlotting results...")
    var_model.plot_results(results)

    # Save results
    try:
        df = pd.DataFrame({'Date': results['test_dates']})
        df['Return'] = results['actual_returns']
        for alpha in var_model.alphas:
            df[f'VaR_{int((1-alpha)*100)}'] = results['VaR_forecasts'][alpha]
            df[f'Breach_{int((1-alpha)*100)}'] = df['Return'] < -df[f'VaR_{int((1-alpha)*100)}']
        df['Volatility'] = results['VaR_std_forecasts']
        df.to_csv('integrated_var_backtest_results.csv', index=False)
        print("Results saved to integrated_var_backtest_results.csv")
    except Exception as err:
        print(f"Could not save: {err}")