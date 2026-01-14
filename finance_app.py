import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timedelta
import yfinance as yf

import json
import re
import requests

# --- 1. DATABASE & MODEL SETUP (Local Storage) ---
Base = declarative_base()
engine = create_engine('sqlite:///personal_finance.db', echo=False)


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


# get funds
def fetch_fund_history(isin, fund_name):
    """
    Uses the official yfinance library to get 3 years of data.
    """
    ticker_symbol = f"{isin}.IR"  # .OL is the suffix for Oslo-listed assets (not correct on yahoo finance)
                                  # BUT needed to switch to .IR to find the correct references on yahoo finance
    ticker = yf.Ticker(ticker_symbol)

    # Fetch 3 years of history
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)

    try:
        df = ticker.history(start=start_date, end=end_date)

        info = ticker.info
        overall_Rating = info.get("morningStarOverallRating", 0.0)
        risk_Rating = info.get("morningStarRiskRating", 0.0)
        ratings = (overall_Rating, risk_Rating)

        if df.empty:
            # Some funds might not have the .OL suffix, try without
            df = yf.Ticker(isin).history(start=start_date, end=end_date)

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
        return []


def update_database_with_history(session):
    """
    Orchestrator: Runs the fetcher for all funds and updates DB.
    """
    funds = {
        "DNB Norden Indeks A": "0P0001CTL0",
        "KLP AksjeGlobal P": "0P00018V9L",
        "KLP AksjeEuropa Indeks P": "0P00016TML",
        "Norne Kombi 50": "0P0001LRWQ",
        "Norne Aksje Norge": "0P0001LR0Y"
    }

    count = 0
    for name, isin in funds.items():
        # Check if we already have recent data to avoid spamming
        last_entry = session.query(FundPriceHistory).filter_by(isin=isin).order_by(FundPriceHistory.date.desc()).first()

        # If no data or data is old (> 7 days), fetch update
        if not last_entry or last_entry.date < datetime.now() - timedelta(days=7):
            print(f"Updating history for {name}...")
            new_data, ratings = fetch_fund_history(isin, name)
            overall_val, risk_val = ratings
            for entry in new_data:
                # Optional: Check if date exists to avoid duplicates
                exists = session.query(FundPriceHistory).filter_by(isin=isin, date=entry['date']).first()
                if not exists:
                    record = FundPriceHistory(
                        overall_rating = overall_val,
                        risk_rating = risk_val,
                        **entry
                        )
                    session.add(record)
                    count += 1
            session.commit()

    return count
# END get funds


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()



def log_investment(session, amount, allocation_type, isin=None):
    """
    Logs a new investment and automatically links it to the latest fund price
    to calculate units purchased.
    """
    new_record = FinancialRecord(
        date=datetime.now(),
        year=datetime.now().year,
        amount=amount,
        allocation_type=allocation_type,
        isin=isin
    )

    if allocation_type == 'Fund' and isin:
        # 1. FIND THE PRICE
        # Get the most recent price available for this ISIN
        latest_price_row = session.query(FundPriceHistory) \
            .filter_by(isin=isin) \
            .order_by(desc(FundPriceHistory.date)) \
            .first()

        if latest_price_row:
            new_record.entry_price = latest_price_row.close_price
            # 2. CALCULATE UNITS
            # If you invest 10,000 NOK and price is 100 NOK, you own 100 units.
            new_record.units_owned = amount / latest_price_row.close_price
            new_record.fund_name = latest_price_row.fund_name  # Optional: Cache the name
        else:
            # Fallback if no history exists yet (e.g. manual entry required later)
            print(f"Warning: No price history found for {isin}")
            new_record.units_owned = 0

    session.add(new_record)
    session.commit()

# --- 2. THE STRATEGY ENGINE (Logic) ---

def get_current_rates(year):
    """Fetch rates for the given year, or return defaults if not set."""
    data = session.query(MarketData).filter_by(year=year).first()
    if data:
        return data.debt_rate, data.savings_rate, data.market_return_est
    return 5.06, 4.8, 7.0  # Defaults based on your context


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

# --- 3. STREAMLIT DASHBOARD (Frontend) ---

st.set_page_config(page_title="FinStrat: Cradle to Grave", layout="wide")

# Styling
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .stAlert {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("💰 FinStrat: Strategic Wealth Manager")
st.markdown("### *Cradle to Grave Strategy Engine*")

# --- SIDEBAR: Configuration & New Cash ---
st.sidebar.header("1. Market Conditions (Yearly)")
curr_year = datetime.now().year
sel_year = st.sidebar.number_input("Year", value=curr_year)
d_rate = st.sidebar.number_input("Debt Interest Rate (%)", value=5.06)
s_rate = st.sidebar.number_input("Safe Savings Rate (%)", value=4.80)
m_rate = st.sidebar.number_input("Est. Market Return (%)", value=7.00)

if st.sidebar.button("Update Market Data"):
    existing = session.query(MarketData).filter_by(year=sel_year).first()
    if not existing:
        new_data = MarketData(year=sel_year, debt_rate=d_rate, savings_rate=s_rate, market_return_est=m_rate)
        session.add(new_data)
    else:
        existing.debt_rate = d_rate
        existing.savings_rate = s_rate
        existing.market_return_est = m_rate
    session.commit()
    st.sidebar.success(f"Rates for {sel_year} updated!")

st.sidebar.markdown("---")
st.sidebar.header("2. Receive Funds")
new_cash = st.sidebar.number_input("Incoming Cash Amount (NOK)", min_value=0, step=1000)

# --- SIDEBAR ADDITION ---
st.sidebar.markdown("---")
st.sidebar.header("3. Data Sync")

if st.sidebar.button("Sync 3Y Fund History"):
    with st.spinner("Fetching 3 years of daily data for all funds..."):
        # Make sure table exists
        Base.metadata.create_all(engine)

        # Run scraper
        rows_added = update_database_with_history(session)
        st.sidebar.success(f"Database updated! {rows_added} new data points added.")

# --- MAIN DASHBOARD ---

# Load Data
records = pd.read_sql(session.query(FinancialRecord).statement, session.bind)
total_invested = records['amount'].sum() if not records.empty else 0
current_savings = records[records['allocation_type'].str.contains('Savings')][
    'amount'].sum() if not records.empty else 0

# Strategy Calculation for New Cash
if new_cash > 0:
    st.subheader("🤖 AI Strategy Recommendation")

    # Run Model
    alloc, risk = recommend_allocation(new_cash, current_savings, d_rate, s_rate, m_rate)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(
            f"Based on the current debt rate of **{d_rate}%** and market outlook of **{m_rate}%**, here is the optimal split:")

        # Display Recommendation
        for asset, amt in alloc.items():
            st.info(f"**{asset}:** {amt:,.0f} NOK")

        # Diversification Warning
        if 'Global Index Fund' in alloc and (alloc['Global Index Fund'] / new_cash) > 0.8:
            st.warning("⚠️ Warning: Concentration Risk. Ensure you aren't over-exposed to one asset class.")

    with col2:
        st.write("**Risk Score (0-5)**")
        st.progress(risk / 5)
        st.caption(f"Strategy Risk Level: {risk}/5")

    if st.button("Confirm & Log Investment"):
        for asset, amt in alloc.items():
            rate_used = m_rate if 'Fund' in asset else (s_rate if 'Savings' in asset else d_rate)
            rec = FinancialRecord(year=sel_year, amount=amt, allocation_type=asset, interest_rate_at_time=rate_used,
                                  risk_score=risk)
            session.add(rec)
        session.commit()
        st.rerun()

st.markdown("---")

# --- MAIN DASHBOARD VISUALIZATION ---

# Fetch data for plotting
history_df = pd.read_sql(session.query(FundPriceHistory).statement, session.bind)
# Convert ratings to Star strings for a better look (Optional but recommended)
history_df['stars_display'] = history_df['overall_rating'].apply(lambda x: "⭐" * int(x) if x else "N/A")

if not history_df.empty:
    st.subheader("📈 Fund Performance (Last 3 Years)")

    # Create a clean line chart
    fig = px.line(
        history_df,
        x="date",
        y="close_price",
        color="fund_name",
        custom_data=["stars_display", "risk_rating"]  # Extra data for the template
    )

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{fullData.name}</b>",
            "Date: %{x|%Y-%m-%d}",
            "Price: %{y:.2f} NOK",
            "Rating: %{customdata[0]}",
            "Risk: %{customdata[1]}/5",
            "<extra></extra>"  # This removes the 'trace name' box on the right
        ])
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No historical data found. Click 'Sync 3Y Fund History' in the sidebar.")

# --- PORTFOLIO OVERVIEW ---

if not records.empty:
    st.subheader("📊 Portfolio Dashboard")

    # Metrics
    net_vs_debt = calculate_opportunity_cost(session.query(FinancialRecord).all())

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Capital Deployed", f"{total_invested:,.0f} NOK")
    m2.metric("Portfolio Risk Score", f"{records['risk_score'].mean():.1f} / 5.0")
    m3.metric("Net Profit vs Paying Debt", f"{net_vs_debt:,.0f} NOK", delta_color="normal")

    # Visuals
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Asset Allocation")
        fig_pie = px.pie(records, values='amount', names='allocation_type', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### Investment History")
        fig_bar = px.bar(records, x='year', y='amount', color='allocation_type', title="Yearly Deployments")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed Table
    with st.expander("View Detailed History"):
        st.dataframe(records)
else:
    st.info("No investments recorded yet. Use the sidebar to add your first cash inflow.")