import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8e8;
  }
  .stApp { background-color: #0a0a0f; }

  h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

  .metric-card {
    background: linear-gradient(135deg, #12121a 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
  }
  .metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6060a0;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #a0d4ff;
    margin-top: 0.2rem;
  }
  .metric-value.positive { color: #5effa0; }
  .metric-value.negative { color: #ff6060; }

  .section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4040a0;
    border-bottom: 1px solid #1a1a3a;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }
  div[data-testid="stSidebar"] {
    background-color: #0d0d18;
    border-right: 1px solid #1a1a3a;
  }
  .stButton > button {
    background: linear-gradient(135deg, #2020a0, #4040ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #3030c0, #5050ff);
    transform: translateY(-1px);
  }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    prices = prices.dropna(how="all")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def portfolio_stats(weights: np.ndarray, mean_ret: np.ndarray, cov: np.ndarray, trading_days: int = 252):
    w = np.array(weights)
    ret = np.dot(w, mean_ret) * trading_days
    vol = np.sqrt(np.dot(w.T, np.dot(cov * trading_days, w)))
    sharpe = ret / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def run_monte_carlo(mean_ret, cov, n_assets, n_sim=5000, trading_days=252):
    results = np.zeros((3, n_sim))
    weights_all = np.zeros((n_assets, n_sim))
    for i in range(n_sim):
        w = np.random.dirichlet(np.ones(n_assets))
        r, v, s = portfolio_stats(w, mean_ret, cov, trading_days)
        results[0, i] = v
        results[1, i] = r
        results[2, i] = s
        weights_all[:, i] = w
    return results, weights_all


def efficient_frontier(mean_ret, cov, n_assets, n_points=80, trading_days=252):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n_assets
    target_returns = np.linspace(mean_ret.min() * trading_days, mean_ret.max() * trading_days, n_points)
    frontier_vol, frontier_ret = [], []
    for target in target_returns:
        cons = constraints + [{"type": "eq", "fun": lambda w, t=target: portfolio_stats(w, mean_ret, cov, trading_days)[0] - t}]
        res = minimize(
            lambda w: portfolio_stats(w, mean_ret, cov, trading_days)[1],
            x0=np.ones(n_assets) / n_assets,
            method="SLSQP", bounds=bounds, constraints=cons,
            options={"ftol": 1e-9, "maxiter": 1000}
        )
        if res.success:
            frontier_vol.append(res.fun)
            frontier_ret.append(target)
    return np.array(frontier_vol), np.array(frontier_ret)


def max_sharpe_portfolio(mean_ret, cov, n_assets, trading_days=252):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n_assets
    res = minimize(
        lambda w: -portfolio_stats(w, mean_ret, cov, trading_days)[2],
        x0=np.ones(n_assets) / n_assets,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000}
    )
    return res.x if res.success else np.ones(n_assets) / n_assets


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⟁ Portfolio Optimizer")
    st.markdown('<div class="section-title">Asset Universe</div>', unsafe_allow_html=True)

    default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA"
    ticker_input = st.text_input("Tickers (comma-separated)", value=default_tickers)
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    st.markdown('<div class="section-title">Period</div>', unsafe_allow_html=True)
    end_date = datetime.today()
    start_options = {
        "1 Year": end_date - timedelta(days=365),
        "2 Years": end_date - timedelta(days=730),
        "3 Years": end_date - timedelta(days=1095),
        "5 Years": end_date - timedelta(days=1825),
    }
    period_label = st.selectbox("Lookback Period", list(start_options.keys()), index=1)
    start_date = start_options[period_label]

    st.markdown('<div class="section-title">Simulation</div>', unsafe_allow_html=True)
    n_sim = st.slider("Monte Carlo Iterations", min_value=1000, max_value=20000, value=5000, step=1000)

    run_btn = st.button("▶ Run Optimization")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("# Portfolio Optimizer")
st.markdown("**Markowitz Efficient Frontier** × **Monte Carlo Simulation**")
st.markdown("---")

if run_btn:
    if len(tickers) < 2:
        st.error("2銘柄以上を入力してください。")
        st.stop()

    with st.spinner("Fetching market data..."):
        prices = fetch_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    # 取得できた銘柄だけ使う
    valid_tickers = [t for t in tickers if t in prices.columns]
    if len(valid_tickers) < 2:
        st.error(f"有効な銘柄が2つ未満です。取得できた銘柄: {valid_tickers}")
        st.stop()
    prices = prices[valid_tickers]
    n_assets = len(valid_tickers)

    returns = compute_returns(prices)
    mean_ret = returns.mean().values
    cov = returns.cov().values

    with st.spinner(f"Running {n_sim:,} Monte Carlo simulations..."):
        mc_results, mc_weights = run_monte_carlo(mean_ret, cov, n_assets, n_sim)

    with st.spinner("Computing efficient frontier..."):
        ef_vol, ef_ret = efficient_frontier(mean_ret, cov, n_assets)

    opt_weights = max_sharpe_portfolio(mean_ret, cov, n_assets)
    opt_ret, opt_vol, opt_sharpe = portfolio_stats(opt_weights, mean_ret, cov)

    # ── Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, "Expected Return", f"{opt_ret*100:.2f}%", opt_ret > 0),
        (col2, "Annualized Risk", f"{opt_vol*100:.2f}%", None),
        (col3, "Sharpe Ratio", f"{opt_sharpe:.3f}", opt_sharpe > 1),
        (col4, "Assets", f"{n_assets}", None),
    ]
    for col, label, value, positive in metrics:
        with col:
            cls = "positive" if positive is True else ("negative" if positive is False else "")
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value {cls}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Chart ──
    sharpe_vals = mc_results[2]
    sharpe_norm = (sharpe_vals - sharpe_vals.min()) / (sharpe_vals.max() - sharpe_vals.min() + 1e-9)

    fig = go.Figure()

    # Monte Carlo scatter
    fig.add_trace(go.Scatter(
        x=mc_results[0] * 100,
        y=mc_results[1] * 100,
        mode="markers",
        marker=dict(
            size=3,
            color=sharpe_norm,
            colorscale=[[0, "#1a1a4a"], [0.4, "#3030c0"], [0.7, "#60a0ff"], [1.0, "#a0ffcc"]],
            opacity=0.6,
            colorbar=dict(
                title=dict(text="Sharpe Ratio", font=dict(color="#6060a0", size=11)),
                tickfont=dict(color="#6060a0", size=10),
                thickness=10,
            ),
        ),
        name="Monte Carlo Portfolios",
        hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))

    # Efficient Frontier
    if len(ef_vol) > 0:
        fig.add_trace(go.Scatter(
            x=ef_vol * 100,
            y=ef_ret * 100,
            mode="lines",
            line=dict(color="#ffffff", width=2.5, dash="solid"),
            name="Efficient Frontier",
            hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra>Efficient Frontier</extra>",
        ))

    # Optimal portfolio
    fig.add_trace(go.Scatter(
        x=[opt_vol * 100],
        y=[opt_ret * 100],
        mode="markers",
        marker=dict(
            size=16, color="#ffffff",
            symbol="star",
            line=dict(color="#a0ffcc", width=2),
        ),
        name=f"Max Sharpe ({opt_sharpe:.2f})",
        hovertemplate=f"Max Sharpe Portfolio<br>Risk: {opt_vol*100:.2f}%<br>Return: {opt_ret*100:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0d0d18",
        font=dict(family="DM Mono, monospace", color="#8080c0"),
        title=dict(
            text=f"Efficient Frontier  ·  {n_sim:,} simulations  ·  {period_label}",
            font=dict(size=14, color="#6060a0"),
        ),
        xaxis=dict(
            title="Annualized Risk (%)",
            gridcolor="#1a1a3a", zerolinecolor="#2a2a4a",
        ),
        yaxis=dict(
            title="Annualized Return (%)",
            gridcolor="#1a1a3a", zerolinecolor="#2a2a4a",
        ),
        legend=dict(
            bgcolor="#0d0d18", bordercolor="#2a2a4a", borderwidth=1,
            font=dict(size=11),
        ),
        height=520,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Optimal Weights ──
    st.markdown("---")
    st.markdown('<div class="section-title">Optimal Weights — Max Sharpe Portfolio</div>', unsafe_allow_html=True)

    weight_df = pd.DataFrame({
        "Ticker": valid_tickers,
        "Weight (%)": (opt_weights * 100).round(2),
    }).sort_values("Weight (%)", ascending=False).reset_index(drop=True)

    col_w, col_chart = st.columns([1, 1])
    with col_w:
        st.dataframe(
            weight_df.style.background_gradient(subset=["Weight (%)"], cmap="Blues"),
            hide_index=True,
            use_container_width=True,
        )
    with col_chart:
        fig_pie = go.Figure(go.Pie(
            labels=weight_df["Ticker"],
            values=weight_df["Weight (%)"],
            hole=0.55,
            marker=dict(colors=["#2020a0", "#3030c0", "#4040e0", "#6060ff", "#8080ff", "#a0a0ff"][:n_assets]),
            textfont=dict(family="DM Mono, monospace", size=12),
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0a0a0f",
            font=dict(color="#8080c0"),
            showlegend=True,
            legend=dict(bgcolor="#0a0a0f", font=dict(size=11)),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Correlation Heatmap ──
    st.markdown("---")
    st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
    corr = returns.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, "#0a0a2a"], [0.5, "#2020a0"], [1, "#a0d4ff"]],
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(family="DM Mono, monospace", size=11),
    ))
    fig_corr.update_layout(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0d0d18",
        font=dict(family="DM Mono, monospace", color="#8080c0"),
        height=360,
        margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.markdown("""
    <div style="
      text-align:center; padding: 4rem 2rem;
      color: #3030a0; font-family: 'DM Mono', monospace;
      font-size: 0.9rem; letter-spacing: 0.1em;
    ">
      ← Configure assets and click <strong style="color:#6060ff">▶ Run Optimization</strong> to begin
    </div>
    """, unsafe_allow_html=True)
