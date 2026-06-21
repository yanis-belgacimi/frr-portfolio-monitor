import pandas as pd
import numpy as np
from data_loader import REQUIRED_COLUMNS

class PortfolioAnalytics:
    def __init__(self, portfolio : pd.Series, benchmark : pd.Series, risk_free_rate = 0.03):
         self.portfolio, self.benchmark = portfolio.align(benchmark, join = "inner")
         self.rf = risk_free_rate

    def returns(self):
        returns1 = self.portfolio.pct_change().dropna() 
        returns2 = self.benchmark.pct_change().dropna() 
        return pd.DataFrame({"Portfolio" : returns1, "Benchmark" : returns2})

    def twr(self):
        returns = self.returns().copy()
        return ((1 + returns).cumprod() - 1).iloc[-1]
    
    def annualized_return(self):
        returns = self.returns()
        return (1 + returns.mean())**252 - 1
    
    def volatility(self):
        returns = self.returns().copy()
        return returns.std() * np.sqrt(252)
    
    def sharpe(self):
        ann_returns = self.annualized_return()
        vol = self.volatility()
        sharpe_ratio = (ann_returns - self.rf) / vol
        return sharpe_ratio
    
    def sortino(self):
        ann_returns = self.annualized_return()
        returns = self.returns().copy()

        downside = returns.clip(upper=0)
        vol_down = np.sqrt((downside**2).mean()) * np.sqrt(252)

        sortino_ratio = (ann_returns - self.rf) / vol_down
        return sortino_ratio
    
    def beta(self):
        cov = self.benchmark.cov(self.portfolio)
        var_benchmark = self.benchmark.var()
        beta = cov / var_benchmark
        return pd.Series([beta, 1], index=["Portfolio", "Benchmark"])
    
    def alpha(self):
        beta = self.beta()["Portfolio"]
        port_ann_return = self.annualized_return()["Portfolio"]
        bench_ann_return = self.annualized_return()["Benchmark"]
        rf = self.rf
        # Alpha
        alpha = port_ann_return - (rf + beta * (bench_ann_return - rf))
        return pd.Series([alpha, 0.0], index=["Portfolio", "Benchmark"])
    
    def tracking_error(self):
        port_returns = self.returns().copy()["Portfolio"]
        bench_returns = self.returns().copy()["Benchmark"]
        te = (port_returns - bench_returns).std() * np.sqrt(252)
        return pd.Series([te, np.nan], index = ["Portfolio", "Benchmark"])
    
    def information_ratio(self):
        alpha = self.alpha()["Portfolio"]
        te = self.tracking_error()["Portfolio"]
        ir = alpha / te
        return pd.Series([ir, np.nan], index = ["Portfolio", "Benchmark"])
    
    def max_drawdown(self):
        # Retrieve price series
        port = self.portfolio.copy()
        bench = self.benchmark.copy()

        # Drawdown
        port_drawdown = (self.portfolio - port.cummax())/ port.cummax()
        bench_drawdown = (self.benchmark - bench.cummax()) /bench.cummax()

        # Return max drawdown
        df_drawdown = pd.Series([port_drawdown.min(), bench_drawdown.min()], index = ["Portfolio", "Benchmark"])
        return df_drawdown

    def summary(self):
        rows = {"TWR" : self.twr(), "Annualized Return" : self.annualized_return(),
                "Sharpe" : self.sharpe(), "Sortino" : self.sortino(),
                "Beta" : self.beta(), "Alpha" : self.alpha(), "TE" : self.tracking_error(),
                "IR" : self.information_ratio(), "Max DD" : self.max_drawdown()}

        df_summary = pd.DataFrame(rows).T
                                               
        return df_summary
