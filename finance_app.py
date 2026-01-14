import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, desc
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timedelta
import yfinance as yf

import json
import re
import requests

# --- 1. DATABASE & MODEL SETUP (Local Storage) ---
Base = declarative_base()
engine = create_engine('sqlite:///personal_finance_v2.db', echo=False)


class FinancialRecord(Base):
    __tablename__ = 'financial_records'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.now)
    year = Column(Integer)
    amount = Column(Float)  # Cash value invested at time T

    # NEW COLUMNS
    allocation_type = Column(String)  # Broader Category: 'Fund', 'Savings', 'Debt'
    isin = Column(String, nullable=True)  # The Link: e.g., 'NO0010000000'
    entry_price = Column(Float, nullable=True)  # Price of fund at moment of purchase
    units_owned = Column(Float, nullable=True)  # Calculated: amount / entry_price

    interest_rate_at_time = Column(Float)
    risk_score = Column(Integer)


class MarketData(Base):
    __tablename__ = 'market_data'
    id = Column(Integer, primary_key=True)
    year = Column(Integer, unique=True)
    debt_rate = Column(Float)  # e.g., 5.06
    savings_rate = Column(Float)  # e.g., 4.8
    market_return_est = Column(Float)  # e.g., 7.0

# --- Add this to Section 1: DATABASE & MODEL SETUP ---

class FundPriceHistory(Base):
    __tablename__ = 'fund_price_history'
    id = Column(Integer, primary_key=True)
    isin = Column(String, index=True) # Indexed for fast joins
    fund_name = Column(String)
    date = Column(DateTime)
    close_price = Column(Float)
    overall_rating = Column(Float)
    risk_rating = Column(Float)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


# --- 2. CONFIGURATION & MAPPING ---

# Mapping Strategy Names -> Real Yahoo Finance Tickers
FUND_MAP = {
    'Global Index Fund': {'isin': '0P00018V9L', 'name': 'KLP AksjeGlobal P'},
    'Nordic Fund': {'isin': '0P0001CTL0', 'name': 'DNB Norden Indeks A'},
    'European Fund': {'isin': '0P00016TML', 'name': 'KLP AksjeEuropa Indeks P'},
    'Norwegian Fund': {'isin': '0P0001LR0Y', 'name': 'Norne Aksje Norge'},

    # Safe Assets (Fixed Price Proxy)
    'Savings (Buffer)': {'isin': 'BANK_SAVINGS', 'name': 'Fana Bufferkonto', 'fixed_price': 1.0},
    'Debt Paydown': {'isin': 'DEBT_PAYMENT', 'name': 'Mortgage Payment', 'fixed_price': 1.0},
}


# get funds
def fetch_fund_history(isin, fund_name):
    """Uses yfinance to get 3 years of data."""
    ticker_symbol = f"{isin}.IR"  # Irish exchange suffix often used for funds on Yahoo
    ticker = yf.Ticker(ticker_symbol)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)

    try:
        # Get Info
        info = ticker.info
        overall_Rating = info.get("morningStarOverallRating", 0.0)
        risk_Rating = info.get("morningStarRiskRating", 0.0)
        ratings = (overall_Rating, risk_Rating)

        # Get History
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            # Fallback: Try without suffix if needed, or return empty
            return [], ratings

        history = []
        for date, row in df.iterrows():
            history.append({
                "isin": isin,
                "fund_name": fund_name,
                "date": date,
                "close_price": row['Close']
            })
        return history, ratings
    except Exception as e:
        print(f"Yahoo Finance error for {fund_name}: {e}")
        return [], (0.0, 0.0)


def update_database_with_history(session_obj):
    """Orchestrator: Updates DB with fresh data."""
    # List of funds to check (extracted from FUND_MAP)
    funds_to_check = {v['name']: v['isin'] for k, v in FUND_MAP.items() if 'fixed_price' not in v}

    count = 0
    status_text = ""

    for name, isin in funds_to_check.items():
        # Check if we have recent data (last 7 days)
        last_entry = session_obj.query(FundPriceHistory).filter_by(isin=isin).order_by(
            desc(FundPriceHistory.date)).first()

        if not last_entry or last_entry.date < datetime.now() - timedelta(days=7):
            new_data, ratings = fetch_fund_history(isin, name)
            overall_val, risk_val = ratings

            added_this_round = 0
            for entry in new_data:
                # Avoid duplicates
                exists = session_obj.query(FundPriceHistory).filter_by(isin=isin, date=entry['date']).first()
                if not exists:
                    record = FundPriceHistory(
                        overall_rating=overall_val,
                        risk_rating=risk_val,
                        **entry
                    )
                    session_obj.add(record)
                    added_this_round += 1

            if added_this_round > 0:
                count += added_this_round
                status_text += f"Updated {name}: {added_this_round} records.\n"

    session_obj.commit()
    return count, status_text
# END get funds


# --- 4. CORE LOGIC ---

def get_latest_price(isin, default_val=1.0):
    """Fetch most recent price from DB."""
    record = session.query(FundPriceHistory).filter_by(isin=isin).order_by(desc(FundPriceHistory.date)).first()
    if record:
        return record.close_price

    # Check if it is a fixed asset (Savings/Debt)
    for k, v in FUND_MAP.items():
        if v.get('isin') == isin and 'fixed_price' in v:
            return v['fixed_price']

    return default_val


def get_fund_rating(isin):
    """Fetch Morningstar Rating."""
    record = session.query(FundPriceHistory).filter_by(isin=isin).order_by(desc(FundPriceHistory.date)).first()
    if record:
        return record.risk_rating
    return 0


def log_investment(amount, strategy_key, risk, year, debt_rate):
    """Log investment using Real Market Price."""
    details = FUND_MAP.get(strategy_key, {})
    isin = details.get('isin', 'UNKNOWN')

    current_price = get_latest_price(isin)

    if 'Fund' in strategy_key:
        units = amount / current_price
    else:
        units = amount  # 1:1 for cash
        current_price = 1.0

    rec = FinancialRecord(
        year=year,
        amount=amount,
        allocation_type=strategy_key,
        isin=isin,
        entry_price=current_price,
        units_owned=units,
        interest_rate_at_time=debt_rate if 'Debt' in strategy_key else 0,
        risk_score=risk
    )
    session.add(rec)
    session.commit()


def get_portfolio_df():
    records = session.query(FinancialRecord).all()
    data = []

    for r in records:
        current_price = get_latest_price(r.isin, default_val=r.entry_price or 1.0)

        # Calculate Value
        if r.units_owned:
            cur_val = r.units_owned * current_price
        else:
            cur_val = r.amount  # Fallback

        profit = cur_val - r.amount

        # Get Rating
        m_risk = get_fund_rating(r.isin) if 'Fund' in r.allocation_type else 0

        data.append({
            "Allocation": r.allocation_type,
            "ISIN": r.isin,
            "Date": r.date.date(),
            "Invested": r.amount,
            "Entry Price": r.entry_price,
            "Current Price": current_price,
            "Units": r.units_owned,
            "Current Value": cur_val,
            "Profit (NOK)": profit,
            "Morningstar Risk": m_risk
        })
    return pd.DataFrame(data)

# --- 2. THE STRATEGY ENGINE (Logic) ---



def recommend_allocation(amount, current_savings, debt_rate, savings_rate, market_rate):
    """
    The 'Brain' of the model. Decides where money should go based on the analysis.
    """
    strategy = {}
    remaining = amount

    # 1. Emergency Fund Rule (Must have 10  0k buffer)
    if current_savings < 100000:
        needed = 100000 - current_savings
        to_savings = min(remaining, needed)
        strategy['Savings (Buffer)'] = to_savings
        remaining -= to_savings

    if remaining <= 0:
        return strategy, 1  # Low Risk

    # 2. The Debt vs Market Logic
    # Tax adjusted comparison (assuming 22% tax deduction on debt, 37% on gain)
    real_debt_cost = debt_rate * 0.78
    real_market_return = market_rate * 0.62

    if real_debt_cost > real_market_return:
        strategy['Debt Paydown'] = remaining
        risk_level = 0
    else:
        # 3. Diversification Strategy (70% Global, 20% Nordic, 10% Cash/Sectorstre)
        strategy['Global Index Fund'] = remaining * 0.75
        strategy['Nordic/Sector Fund'] = remaining * 0.25
        risk_level = 4  # Moderate-High Risk

    return strategy, risk_level


def calculate_opportunity_cost(records):
    """
    Calculates the Net Profit vs Paying Debt for all historical investments.
    """
    total_benefit = 0
    for record in records:
        years_held = datetime.now().year - record.year
        if years_held < 0: years_held = 0

        # Current Value of this specific chunk of money
        if record.allocation_type == 'Debt Paydown':
            # Benefit is the interest saved
            val = record.amount * ((1 + record.interest_rate_at_time / 100 * 0.78) ** years_held)
            baseline = record.amount * ((1 + record.interest_rate_at_time / 100 * 0.78) ** years_held)
        else:
            # Asset growth (simplified assumption: it grew at the rate recorded)
            # In a real app, you'd fetch live stock prices here.
            # We assume the 'interest_rate_at_time' for funds was the EXPECTED return (7%)
            growth_rate = record.interest_rate_at_time / 100
            if 'Fund' in record.allocation_type:
                growth_rate = growth_rate * 0.62  # Tax adjusted
            else:
                growth_rate = growth_rate * 0.78  # Interest Tax

            val = record.amount * ((1 + growth_rate) ** years_held)

            # Baseline: Had we used this money to pay debt?
            # We fetch the debt rate for that year
            hist_rates = session.query(MarketData).filter_by(year=record.year).first()
            d_rate = hist_rates.debt_rate if hist_rates else 5.06
            baseline = record.amount * ((1 + d_rate / 100 * 0.78) ** years_held)

        total_benefit += (val - baseline)
    return total_benefit


def plot_fund_performance_with_entries(session, selected_isin):
    # 1. Fetch Price History
    history = session.query(FundPriceHistory).filter_by(isin=selected_isin).order_by(FundPriceHistory.date).all()
    if not history:
        return None

    df_hist = pd.DataFrame([{"Date": h.date, "Price": h.close_price, "Fund": h.fund_name} for h in history])
    fund_name = df_hist['Fund'].iloc[0]

    # 2. Fetch Investments for this fund
    investments = session.query(FinancialRecord).filter_by(isin=selected_isin).all()
    df_inv = pd.DataFrame([{"Date": i.date, "Entry Price": i.entry_price, "Amount": i.amount} for i in investments])

    # 3. Create the Plot
    fig = px.line(df_hist, x="Date", y="Price", title=f"Historical Price: {fund_name}")

    if not df_inv.empty:
        # Add a scatter layer for the entry points
        fig.add_trace(go.Scatter(
            x=df_inv["Date"],
            y=df_inv["Entry Price"],
            mode='markers+text',
            name='My Investments',
            marker=dict(size=12, color='Gold', symbol='star', line=dict(width=2, color='DarkSlateGrey')),
            text=[f"Investert: {amt:,.0f} NOK" for amt in df_inv["Amount"]],
            textposition="top center"
        ))

    fig.update_layout(hovermode="x unified", template="plotly_white")
    return fig


# --- 5. STREAMLIT FRONTEND ---

st.set_page_config(page_title="FinStrat: Real-Time", layout="wide")
st.title("💰 FinStrat: Real-Time Wealth Manager")

# SIDEBAR: DATA SYNC
st.sidebar.header("📡 Market Data")
if st.sidebar.button("🔄 Sync Yahoo Finance Data"):
    with st.spinner("Connecting to Yahoo Finance..."):
        count, log = update_database_with_history(session)
    if count > 0:
        st.sidebar.success(f"Success! Added {count} new data points.")
        with st.sidebar.expander("View Log"):
            st.text(log)
    else:
        st.sidebar.info("Data is up to date.")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
curr_year = datetime.now().year
d_rate = st.sidebar.number_input("Debt Rate (%)", 5.06)
m_rate = st.sidebar.number_input("Est. Market Return (%)", 7.00)

# TABS
tab1, tab2 = st.tabs(["🚀 Strategy & Allocation", "📊 Live Portfolio"])

with tab1:
    st.subheader("🚀 Investment & Simulation")

    # Toggle between Live and Historical
    mode = st.radio("Select Mode", ["Live Investment (Today)", "Historical Simulation (Backdate)"], horizontal=True)

    col_a, col_b = st.columns(2)
    with col_a:
        new_cash = st.number_input("Amount (NOK)", min_value=0, step=5000)
    with col_b:
        if mode == "Historical Simulation (Backdate)":
            # Allow user to pick a date from the last 3 years
            sim_date = st.date_input("Simulation Date",
                                     value=datetime.now() - timedelta(days=365),
                                     min_value=datetime.now() - timedelta(days=3 * 365),
                                     max_value=datetime.now())
        else:
            sim_date = datetime.now()

    if new_cash > 0:
        # 1. Fetch the price for the chosen date
        st.markdown("---")
        st.info(f"Analyzing strategy for **{sim_date.strftime('%d %B %Y')}**...")

        # Simple Strategy Logic
        real_debt = d_rate * 0.78
        real_mkt = m_rate * 0.62
        rec_map = {'Global Index Fund': new_cash * 0.75, 'Nordic Fund': new_cash * 0.25} if real_mkt > real_debt else {
            'Debt Paydown': new_cash}

        for key, val in rec_map.items():
            details = FUND_MAP.get(key, {})
            isin_ref = details.get('isin')

            # GET HISTORICAL PRICE FOR THAT DATE
            if 'fixed_price' in details:
                hist_price = 1.0
            else:
                # Find the closest price in our database to the sim_date
                hist_record = session.query(FundPriceHistory).filter(
                    FundPriceHistory.isin == isin_ref,
                    FundPriceHistory.date <= sim_date
                ).order_by(desc(FundPriceHistory.date)).first()

                hist_price = hist_record.close_price if hist_record else get_latest_price(isin_ref)

            # Display Preview
            st.write(f"**{key}**")
            c1, c2, c3 = st.columns(3)
            c1.caption(f"Price on {sim_date.strftime('%Y-%m-%d')}")
            c1.write(f"{hist_price:.2f} NOK")

            units = val / hist_price
            c2.caption("Units Acquired")
            c2.write(f"{units:.2f}")

            current_price = get_latest_price(isin_ref)
            growth = ((current_price / hist_price) - 1) * 100
            c3.caption("Growth Since Then")
            c3.write(f"{growth:+.2f}%")

        if st.button("Confirm & Add to Portfolio"):
            for k, v in rec_map.items():
                details = FUND_MAP.get(k, {})
                isin_ref = details.get('isin')

                # Fetch price again for saving
                if 'fixed_price' in details:
                    price_at_entry = 1.0
                else:
                    hist_record = session.query(FundPriceHistory).filter(
                        FundPriceHistory.isin == isin_ref,
                        FundPriceHistory.date <= sim_date
                    ).order_by(desc(FundPriceHistory.date)).first()
                    price_at_entry = hist_record.close_price if hist_record else get_latest_price(isin_ref)

                # LOG THE RECORD
                rec = FinancialRecord(
                    date=datetime.combine(sim_date, datetime.min.time()),
                    year=sim_date.year,
                    amount=v,
                    allocation_type=k,
                    isin=isin_ref,
                    entry_price=price_at_entry,
                    units_owned=v / price_at_entry,
                    interest_rate_at_time=d_rate,
                    risk_score=4 if 'Fund' in k else 1
                )
                session.add(rec)
            session.commit()
            st.success(f"Successfully simulated and added investment from {sim_date.strftime('%Y-%m-%d')}!")
            st.rerun()

with tab2:
    st.subheader("Live Portfolio Valuation")
    df = get_portfolio_df()

    if not df.empty:
        # Summary Metrics
        total_inv = df['Invested'].sum()
        total_val = df['Current Value'].sum()
        total_pl = total_val - total_inv
        roi = (total_pl / total_inv) * 100 if total_inv > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Invested Capital", f"{total_inv:,.0f} NOK")
        m2.metric("Current Value", f"{total_val:,.0f} NOK", delta=f"{total_pl:,.0f} NOK")
        m3.metric("Total ROI", f"{roi:.2f}%")

        # Weighted Risk Score
        avg_risk = df[df['Morningstar Risk'] > 0]['Morningstar Risk'].mean()
        m4.metric("Avg Fund Risk Rating", f"{avg_risk:.1f}" if pd.notna(avg_risk) else "N/A")

        st.markdown("---")

        # --- NEW: HISTORICAL CHART WITH ENTRY MARKS ---
        st.markdown("### 📈 Historical Context & Entry Points")

        # Filter funds that have an ISIN
        available_funds = df[df['ISIN'].str.len() >= 10]['Allocation'].unique()

        if len(available_funds) > 0:
            selected_fund_name = st.selectbox("Select fund to view history", available_funds)
            target_isin = FUND_MAP[selected_fund_name]['isin']

            # Generate the specific plot
            import plotly.graph_objects as go  # Ensure this is imported at top

            fig_hist = plot_fund_performance_with_entries(session, target_isin)

            if fig_hist:
                st.plotly_chart(fig_hist, use_container_width=True)
                st.caption("The stars indicate the exact date and price at which you deployed capital.")

        st.markdown("---")

        # Detailed Table
        st.dataframe(df.style.format({
            "Invested": "{:,.0f}",
            "Current Value": "{:,.0f}",
            "Profit (NOK)": "{:,.0f}",
            "Return (%)": "{:.2f}",
            "Current Price": "{:.2f}"
        }))

        # Visuals
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Allocation Breakdown")
            fig_pie = px.pie(df, values='Current Value', names='Allocation', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.markdown("#### Performance by Asset")
            fig_bar = px.bar(df, x='Allocation', y=['Invested', 'Current Value'], barmode='group')
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("No active investments found. Go to the 'Strategy' tab to add funds.")