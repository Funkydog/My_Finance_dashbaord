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

# --- 1. DATABASE SETUP ---
Base = declarative_base()
engine = create_engine('sqlite:///personal_finance_v4.db', echo=False)

class FinancialRecord(Base):
    __tablename__ = 'financial_records'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.now)
    year = Column(Integer)
    amount = Column(Float)
    allocation_type = Column(String)
    isin = Column(String, nullable=True)
    entry_price = Column(Float, nullable=True)
    units_owned = Column(Float, nullable=True)
    interest_rate_at_time = Column(Float)
    risk_score = Column(Integer)

class FundPriceHistory(Base):
    __tablename__ = 'fund_price_history'
    id = Column(Integer, primary_key=True)
    isin = Column(String, index=True)
    fund_name = Column(String)
    date = Column(DateTime)
    close_price = Column(Float)
    overall_rating = Column(Float, default=0.0)
    risk_rating = Column(Float, default=0.0)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


# --- 2. CONFIGURATION & MAPPING ---

FUND_MAP = {
    'Global Index Fund': {
        'isin': '0P00018V9L',
        'name': 'KLP AksjeGlobal P',
        'desc': 'Exposure to 1,500+ world companies. Heavily weighted in US Tech.',
        'holdings': ['Apple', 'Microsoft', 'NVIDIA', 'Amazon'],
        'sectors': ['Technology (24%)', 'Finance (14%)', 'Health (12%)']
    },
    'Nordic Fund': {
        'isin': '0P0001CTL0',
        'name': 'DNB Norden Indeks A',
        'desc': 'Nordic stability. Strong on Pharma, Energy, and Industrials.',
        'holdings': ['Novo Nordisk (DK)', 'Equinor (NO)', 'Atlas Copco (SE)', 'Investor AB'],
        'sectors': ['Healthcare (35%)', 'Industrials (25%)', 'Energy (10%)']
    },
    'European Fund': {
        'isin': '0P00016TML',
        'name': 'KLP AksjeEuropa Indeks P',
        'desc': 'Eurozone giants. Luxury, Semi-conductors, and Banking.',
        'holdings': ['ASML (Tech)', 'LVMH (Luxury)', 'Nestle', 'SAP'],
        'sectors': ['Consumer (20%)', 'Financials (18%)', 'Tech (8%)']
    },
    'Norwegian Fund': {
        'isin': '0P0001LR0Y',
        'name': 'Norne Aksje Norge',
        'desc': 'Pure Norway exposure. Oil, Fish, and Aluminum.',
        'holdings': ['Equinor', 'DNB', 'Norsk Hydro', 'Mowi'],
        'sectors': ['Energy (30%)', 'Seafood (15%)', 'Materials (10%)']
    },
    'Savings (Buffer)': {'isin': 'BANK_SAVINGS', 'name': 'Fana Bufferkonto', 'fixed_price': 1.0, 'desc': 'Risk-free cash.'},
    'Debt Paydown': {'isin': 'DEBT_PAYMENT', 'name': 'Mortgage Payment', 'fixed_price': 1.0, 'desc': 'Guaranteed 5.06% return.'},
}


def get_exchange_rates():
    """Fetches current NOK to EUR and USD rates."""
    try:
        tickers = yf.Tickers('EURNOK=X USDNOK=X')
        rates = {
            'EUR': tickers.tickers['EURNOK=X'].fast_info['last_price'],
            'USD': tickers.tickers['USDNOK=X'].fast_info['last_price']
        }
        return rates
    except Exception:
        return {'EUR': 11.50, 'USD': 10.50} # Fallback

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
    record = session.query(FundPriceHistory).filter_by(isin=isin).order_by(desc(FundPriceHistory.date)).first()
    if record: return record.close_price
    for v in FUND_MAP.values():
        if v.get('isin') == isin and 'fixed_price' in v: return v['fixed_price']
    return default_val


def get_fund_rating(isin):
    """Fetch Morningstar Rating."""
    record = session.query(FundPriceHistory).filter_by(isin=isin).order_by(desc(FundPriceHistory.date)).first()
    if record:
        return record.risk_rating
    return 0


def log_investment(amount, strategy_key, risk, year, debt_rate, date_override=None):
    details = FUND_MAP.get(strategy_key, {})
    isin = details.get('isin', 'UNKNOWN')

    # Handle Simulations
    if date_override:
        rec_date = datetime.combine(date_override, datetime.min.time())
        if 'fixed_price' in details:
            price = 1.0
        else:
            hist = session.query(FundPriceHistory).filter(FundPriceHistory.isin == isin,
                                                          FundPriceHistory.date <= date_override).order_by(
                desc(FundPriceHistory.date)).first()
            price = hist.close_price if hist else get_latest_price(isin)
    else:
        rec_date = datetime.now()
        price = get_latest_price(isin)

    if 'Fund' in strategy_key:
        units = amount / price
    else:
        units = amount
        price = 1.0

    rec = FinancialRecord(
        date=rec_date, year=rec_date.year, amount=amount, allocation_type=strategy_key,
        isin=isin, entry_price=price, units_owned=units,
        interest_rate_at_time=debt_rate if 'Debt' in strategy_key else 0, risk_score=risk
    )
    session.add(rec)
    session.commit()


# --- 3. DATAFRAME GENERATION ---

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

        # FIX: Calculate Return % safely
        if r.amount > 0:
            ret_pct = (profit / r.amount) * 100
        else:
            ret_pct = 0.0

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
            "Profit": profit,
            "Return (%)": ret_pct,
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


def plot_fund_performance_with_entries(session, selected_isin, currency_symbol, rate):
    # 1. Fetch History
    history = session.query(FundPriceHistory).filter_by(isin=selected_isin).order_by(FundPriceHistory.date).all()
    if not history: return None

    # Convert Historical Prices using CURRENT rate
    df_hist = pd.DataFrame([{"Date": h.date, "Price": h.close_price / rate, "Fund": h.fund_name} for h in history])

    # 2. Fetch Investments
    investments = session.query(FinancialRecord).filter_by(isin=selected_isin).all()
    df_inv = pd.DataFrame([{
        "Date": i.date,
        "Entry Price": i.entry_price / rate,
        "Amount": i.amount / rate
    } for i in investments])

    # 3. Build Plot
    fig = px.line(df_hist, x="Date", y="Price",
                  title=f"Historical Price: {df_hist['Fund'].iloc[0]} ({currency_symbol})")

    if not df_inv.empty:
        # Add Stars with Text Labels
        fig.add_trace(go.Scatter(
            x=df_inv["Date"],
            y=df_inv["Entry Price"],
            mode='markers+text',  # <--- TEXT ENABLED
            name='My Investments',
            marker=dict(size=14, color='Gold', symbol='star', line=dict(width=2, color='DarkSlateGrey')),
            text=[f"Invested {amt:,.0f} {currency_symbol}" for amt in df_inv["Amount"]],  # <--- AMOUNT LABEL
            textposition="top center",
            textfont=dict(color='black', size=11)
        ))

        # --- THE FIX IS ON THIS LINE ---
        # We add template="plotly_white" to force a light background
        fig.update_layout(
            yaxis_title=f"Price ({currency_symbol})",
            hovermode="x unified",
            template="plotly_white"
        )

    return fig


# --- 4. DASHBOARD ---

st.set_page_config(page_title="FinStrat: Global", layout="wide")
st.title("💰 FinStrat: Global Wealth Manager")

# SIDEBAR
if st.sidebar.button("🔄 Sync Market Data"):
    with st.spinner("Fetching Yahoo Finance Data..."):
        c, l = update_database_with_history(session)
    st.sidebar.success(f"Synced {c} records.")


st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
# DEFINING THE RATES HERE SO TAB 1 CAN SEE THEM
d_rate = st.sidebar.number_input("Debt Rate (%)", 5.06)
m_rate = st.sidebar.number_input("Est. Market Return (%)", 7.00)

st.sidebar.markdown("---")
st.sidebar.header("🌍 Currency Settings")
rates = get_exchange_rates()
view_currency = st.sidebar.radio("Display Currency:", ["NOK (Local)", "EUR (Home)", "USD (Global)"])

# Currency Logic
if view_currency == "NOK (Local)":
    rate_div = 1.0
    curr_sym = "kr"
elif view_currency == "EUR (Home)":
    rate_div = rates['EUR']
    curr_sym = "€"
else:
    rate_div = rates['USD']
    curr_sym = "$"

st.sidebar.caption(f"Exchange Rate: 1 {curr_sym} = {rate_div:.2f} NOK")

# TABS
tab1, tab2 = st.tabs(["🚀 Strategy Engine", "📊 Portfolio Overview"])


# --- TAB 1: NEW INVESTMENT (Enhanced) ---
with tab1:
    st.subheader("New Cash Allocation")

    # 1. SETUP
    mode = st.radio("Mode", ["Live (Today)", "Simulation (Backtest)"], horizontal=True)

    col_a, col_b = st.columns(2)
    with col_a:
        inv_amount_nok = st.number_input("Total Amount (NOK)", min_value=0.0, step=5000.0, value=0.0)
    with col_b:
        sim_date = datetime.now()
        if mode == "Simulation (Backtest)":
            sim_date = st.date_input("Simulation Date", value=datetime.now() - timedelta(days=365))

    st.markdown("---")

    if inv_amount_nok > 0:
        # 2. AI STRATEGY ENGINE
        real_debt = d_rate * 0.78
        real_mkt = m_rate * 0.62

        rec_map = {}
        strategy_reason = ""

        if real_debt > real_mkt:
            rec_map = {'Debt Paydown': inv_amount_nok}
            strategy_reason = f"📉 **Defensive Strategy:** The after-tax cost of debt ({real_debt:.2f}%) exceeds market expectations. We recommend paying down your loan."
        else:
            rec_map = {
                'Global Index Fund': inv_amount_nok * 0.70,
                'Nordic Fund': inv_amount_nok * 0.20,
                'European Fund': inv_amount_nok * 0.10
            }
            strategy_reason = f"📈 **Growth Strategy:** Market returns ({real_mkt:.2f}%) are superior to debt cost. We recommend a diversified portfolio."

        st.info(strategy_reason)

        # --- NEW: FUND INTELLIGENCE DISPLAY ---
        if 'Fund' in list(rec_map.keys())[0]:
            st.markdown("### 🔍 Proposed Fund Details")
            cols = st.columns(len(rec_map))

            for idx, (key, amount) in enumerate(rec_map.items()):
                data = FUND_MAP.get(key, {})

                with cols[idx]:
                    # Create a "Card" for each fund
                    st.markdown(f"**{key}**")
                    st.caption(f"{data.get('name')}")

                    # Show allocation amount
                    st.metric("Allocation", f"{amount:,.0f} NOK",
                              help=f"{(amount / inv_amount_nok) * 100:.0f}% of total")

                    # Deep Dive Info
                    with st.expander("See Holdings & Sectors", expanded=True):
                        st.markdown(f"*{data.get('desc')}*")

                        if 'holdings' in data:
                            st.markdown("**🏆 Top Holdings:**")
                            for h in data['holdings']:
                                st.text(f"• {h}")

                        if 'sectors' in data:
                            st.markdown("**🏭 Key Sectors:**")
                            for s in data['sectors']:
                                st.text(f"• {s}")

        # 3. USER CUSTOMIZATION
        st.markdown("---")
        st.subheader("🛠 Customize & Confirm")
        with st.form("allocation_form"):
            user_allocs = {}
            c1, c2 = st.columns(2)
            keys = list(FUND_MAP.keys())

            for i, key in enumerate(keys):
                rec_val = rec_map.get(key, 0.0)
                with (c1 if i % 2 == 0 else c2):
                    # Show the specific fund name in the label for clarity
                    fund_real_name = FUND_MAP[key].get('name')
                    user_allocs[key] = st.number_input(
                        f"{key} ({fund_real_name})",
                        min_value=0.0,
                        value=float(rec_val),
                        step=1000.0
                    )

            st.markdown("---")
            total_input = sum(user_allocs.values())
            diff = inv_amount_nok - total_input

            if abs(diff) > 1.0:
                st.warning(f"⚠️ Mismatch: Allocated {total_input:,.0f} vs Total {inv_amount_nok:,.0f}.")
                ready = False
            else:
                st.success("✅ Allocation matches Total.")
                ready = True

            if st.form_submit_button("Confirm Investment"):
                if ready:
                    for key, amount in user_allocs.items():
                        if amount > 0:
                            risk = 4 if 'Fund' in key else 1
                            log_investment(amount, key, risk, sim_date.year, d_rate,
                                           sim_date if mode == "Simulation (Backtest)" else None)
                    st.toast("Saved!", icon="🎉")
                    st.rerun()
                else:
                    st.error("Fix totals first.")

with tab2:
    st.subheader(f"Portfolio Valuation ({view_currency})")
    df = get_portfolio_df()

    if not df.empty:
        # CONVERT DATA FOR DISPLAY
        df_disp = df.copy()
        cols_money = ['Invested', 'Current Value', 'Profit', 'Entry Price', 'Current Price']
        for c in cols_money:
            df_disp[c] = df_disp[c] / rate_div

        # Metrics
        tot_inv = df_disp['Invested'].sum()
        tot_val = df_disp['Current Value'].sum()
        tot_pl = tot_val - tot_inv
        roi = (tot_pl / tot_inv) * 100 if tot_inv else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Invested Capital", f"{curr_sym} {tot_inv:,.0f}")
        m2.metric("Current Value", f"{curr_sym} {tot_val:,.0f}", delta=f"{tot_pl:,.0f}")
        m3.metric("Total ROI", f"{roi:.2f}%")
        m4.metric("Avg Risk", f"{df_disp[df_disp['Morningstar Risk'] > 0]['Morningstar Risk'].mean():.1f}")

        st.markdown("---")

        # PLOT 1: HISTORICAL (WITH STARS & LABELS)
        funds = df[df['ISIN'].str.len() > 8]['Allocation'].unique()
        if len(funds) > 0:
            sel_fund = st.selectbox("Select Asset History", funds)
            target_isin = FUND_MAP[sel_fund]['isin']
            # Calls the UPDATED function with text labels
            fig_hist = plot_fund_performance_with_entries(session, target_isin, curr_sym, rate_div)
            if fig_hist: st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # PLOT 2 & 3: PIE CHART AND BAR CHART
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Asset Allocation")
            # --- RESTORED PIE CHART ---
            fig_pie = px.pie(df_disp, values='Current Value', names='Allocation', hole=0.4,
                             color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("#### Performance by Asset")
            fig_bar = px.bar(df_disp, x='Allocation', y=['Invested', 'Current Value'], barmode='group',
                             labels={'value': f'Amount ({curr_sym})', 'variable': 'Metric'})
            fig_bar.update_layout(yaxis_title=f"Value ({curr_sym})", legend_title_text='')
            st.plotly_chart(fig_bar, use_container_width=True)

        # DETAILED TABLE
        st.markdown("#### Holding Details")
        st.dataframe(df_disp[['Allocation', 'Date', 'Invested', 'Current Value', 'Profit', 'Return (%)']]
        .style.format({
            'Invested': f"{curr_sym} {{:,.0f}}",
            'Current Value': f"{curr_sym} {{:,.0f}}",
            'Profit': f"{curr_sym} {{:,.0f}}",
            'Return (%)': "{:.2f}%"
        }))

    else:
        st.info("No investments found.")