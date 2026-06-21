import pandas as pd
import matplotlib.pyplot as plt

class PortfolioMonitor():
    def __init__(self, target_weights : dict, tolerance : float | dict):
        self.target_weights = pd.Series(target_weights, dtype=float)

        self.tolerance = (
                    pd.Series(tolerance, index = self.target_weights.index) 
                    if isinstance(tolerance, (float, int))
                    else pd.Series(tolerance, dtype = float)
        )
        
        self.target_weights, self.tolerance = self.target_weights.align(
                    self.tolerance,
                    join = "outer"
        )

        if self.target_weights.isna().any() or self.tolerance.isna().any():
            raise ValueError("Target Weights and Tolerance must cover the same asset classes")
    
    def _validate_alignement(self, data):
        """
        Validates alignement between data columns/index and target_weights.
        """
        cols = data.columns if hasattr(data, "columns") else data.index
        missing_in_data = self.target_weights.index.difference(cols)
        missing_in_target = pd.Index(cols).difference(self.target_weights.index)
        if (not missing_in_data.empty) or (not  missing_in_target.empty):
            raise ValueError(
                f"Mismatch in data :"
                f"missing from data : {missing_in_data}"
                f"missing from target weight : {missing_in_target}"
                ) 

    def compute_drift(self, actual_weights):
        # Check index/columns alignement
        self._validate_alignement(actual_weights)      
        target_weights = self.target_weights                                                   
        return actual_weights - target_weights
    
    def decompose_drift(self, df_weights_history, df_returns_history):
        # Check index/columns alignement
        self._validate_alignement(df_weights_history)  
        self._validate_alignement(df_returns_history)  

        numerator = df_weights_history.shift(1) * (1+ df_returns_history)
        denominator = numerator.sum(axis = 1)
        weights_meca = (numerator.div(denominator, axis = 0))

        drift_meca = weights_meca - df_weights_history.shift(1)
        drift_res = df_weights_history - weights_meca
        return (drift_meca, drift_res)
    
    def check_alerts(self, actual_weights):
        # Check index/columns alignement
        self._validate_alignement(actual_weights)

        drift = self.compute_drift(actual_weights)
        alerts = drift[abs(drift) > self.tolerance]
        return alerts
        
    def track_history(self, weights_history_df):
        # Check index/columns alignement
        self._validate_alignement(weights_history_df)

        drift_history_df =  weights_history_df - self.target_weights
        return drift_history_df
    
    def plot_drift(self, weights_history_df):
        # Check index/columns alignement
        self._validate_alignement(weights_history_df)

        df_drift_history = self.track_history(weights_history_df)
        # Plot
        plt.figure(figsize=(10, 6))
        for col in df_drift_history.columns:
            line, = plt.plot(df_drift_history.index, df_drift_history[col], label = col)
            color = line.get_color()
            # Horizontal line
            plt.axhline(self.tolerance[col], linestyle='--', color=color, alpha=0.5)
            plt.axhline(-self.tolerance[col], linestyle='--', color=color, alpha = 0.5)


        plt.xticks(df_drift_history.index[::7], rotation = 45)

        # Title and legend
        plt.xlabel("Date")
        plt.ylabel("Drift (% point)")
        plt.title("Drift History by sector")
        plt.legend()
        plt.show()
        
        return
        
