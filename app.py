import streamlit as st
from pathlib import Path
import pandas as pd

from src.portfolio_analytics import PortfolioAnalytics
from src.portfolio_monitor import PortfolioMonitor

from src.data_loader import load_portfolio_data

# Set title
st.title("Portfolio Monitor Dashboard")

# Condition on "Load Portfolio button"
if "portfolio_loaded" not in st.session_state:
    st.session_state["portfolio_loaded"] = False


if st.button("Load Portfolio"):
    
    st.session_state["portfolio_loaded"] = True

    # Test Data
    st.session_state["df_returns_history"] = load_portfolio_data(Path(__file__).parent / "data" / "returns_history.csv")
    st.session_state["df_weights_history"] = load_portfolio_data(Path(__file__).parent / "data" / "weights_history.csv")

    st.session_state["target_weights"] = {
        'Government_Bonds': 0.35,
        'IG_Bonds': 0.20,
        'Equities': 0.30,
        'High_Yield_Bonds': 0.10,
        'Emerging_Markets': 0.05
    }

    st.session_state["tolerance_thresholds"] = {
        "Government_Bonds": 0.01,
        "IG_Bonds": 0.01,
        "Equities": 0.05,
        "High_Yield_Bonds": 0.01,
        "Emerging_Markets": 0.002
    }

    # Aggregate a convert returns into prices
    portfolio_returns = (
        (
        st.session_state["df_returns_history"] * 
        st.session_state["df_weights_history"].shift(1)).sum(axis=1)
    )
    benchmark_returns = (
        (
        st.session_state["df_returns_history"] * 
        pd.Series(st.session_state["target_weights"])).sum(axis=1)
    )

    # Convert into index
    st.session_state["portfolio_index"] = (100 * (1 + portfolio_returns).cumprod()).rename("Portfolio")
    st.session_state["benchmark_index"] = (100 * (1 + benchmark_returns).cumprod()).rename("Benchmark")
    st.session_state["portfolio_index"].iloc[0] = 100
    st.session_state["benchmark_index"].iloc[0] = 100

    # Retrieve actual weight to the shock date
    st.session_state["actual_weights"] = st.session_state["df_weights_history"].iloc[30]


    # Initialize classes Monitor and Analytics
    st.session_state["PortfolioMonitor"] = PortfolioMonitor(
                    st.session_state["target_weights"], 
                    st.session_state["tolerance_thresholds"]
            )

    st.session_state["PortfolioAnalytics"] = PortfolioAnalytics(
                    st.session_state["portfolio_index"],
                    st.session_state["benchmark_index"]
            )
    
if st.session_state["portfolio_loaded"]:
    
    st.success("Portfolio loaded")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Vue d'ensemble", "Allocations & Dérives", "Détail des métriques"])

    # Fill tabs
    # Tab 1 contains a summary
    with tab1:
        # Create 4 columns to stock metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("TWR", f"{st.session_state['PortfolioAnalytics'].twr()['Portfolio']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{st.session_state['PortfolioAnalytics'].sharpe()['Portfolio']:.2f}")
        with col3:
            st.metric("Volatility",f"{st.session_state['PortfolioAnalytics'].volatility()['Portfolio']:.2%}")
        with col4:
            st.metric("Max Drawdown",f"{st.session_state['PortfolioAnalytics'].max_drawdown()['Portfolio']:.2%}")
        
        st.line_chart(
            pd.concat([
                st.session_state["portfolio_index"],
                st.session_state["benchmark_index"]], axis = 1
                    )
            )
    # Tab2 contains a drift monitor
    with tab2:
        st.dataframe({
            "Réel" : st.session_state["actual_weights"], 
            "Cible" : st.session_state["PortfolioMonitor"].target_weights,
            "Ecart" : st.session_state["PortfolioMonitor"].compute_drift(st.session_state["actual_weights"])})

        alerte = st.session_state["PortfolioMonitor"].check_alerts(st.session_state["actual_weights"])
        if not (alerte.empty) :
            st.write("Alerte : ")
            alerte
        
    # Tab 3 contains a metrics table
    with tab3:
        df_summary = (st.session_state['PortfolioAnalytics'].summary().T).style.format({
            "TWR" : "{:.2%}",
            "Sharpe" : "{:.2f}",
            "Annualized Return" : "{:.2%}",
            "Sortino" : "{:.2f}",
            "Beta" : "{:.2f}",
            "Alpha Jensen" : "{:.4%}",
            "TE" : "{:.2%}",
            "IR": "{:.2f}",
            "Max DD" : "{:.2%}"
            })
        
        st.dataframe(df_summary)

else:
    st.info("No loaded portfolio")


