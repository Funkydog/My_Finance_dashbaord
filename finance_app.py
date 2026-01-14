import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import json
import hashlib
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, desc, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timedelta

# --- 1. DATABASE SETUP ---
Base = declarative_base()
engine = create_engine('sqlite:///personal_finance_v7.db', echo=False)


# NEW: User Table for Authentication
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)


class FinancialRecord(Base):
    __tablename__ = 'financial_records'
    id = Column(Integer, primary_key=True)

    # NEW: Link to specific user
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    date = Column(DateTime, default=datetime.now)
    year = Column(Integer)
    amount = Column(Float)
    allocation_type = Column(String)
    isin = Column(String, nullable=True)
    entry_price = Column(Float, nullable=True)
    units_owned = Column(Float, nullable=True)
    interest_rate_at_time = Column(Float)
    risk_score = Column(Integer)


class FundProfile(Base):
    __tablename__ = 'fund_profiles'
    isin = Column(String, primary_key=True)
    ticker = Column(String)
    name = Column(String)
    description = Column(Text)
    top_holdings = Column(Text)
    sector_info = Column(Text)
    category = Column(String)


class FundPriceHistory(Base):
    __tablename__ = 'fund_price_history'
    id = Column(Integer, primary_key=True)
    isin = Column(String, index=True)
    date = Column(DateTime)
    close_price = Column(Float)
    risk_rating = Column(Float, default=0.0)


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


# --- 2. AUTHENTICATION HELPERS ---

def make_hash(password):
    """Create a secure SHA256 hash of the password."""
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_login(username, password):
    """Verify credentials against database."""
    pwd_hash = make_hash(password)
    return session.query(User).filter_by(username=username, password_hash=pwd_hash).first()


def register_user(username, password):
    """Create a new user if username doesn't exist."""
    if session.query(User).filter_by(username=username).first():
        return False, "Username already exists."

    new_user = User(username=username, password_hash=make_hash(password))
    session.add(new_user)
    session.commit()
    return True, "User created successfully! Please login."


# --- 3. CONFIGURATION & DATA LOGIC ---
SUGGESTED_FUNDS = [
    {"ticker": "EQQQ.PA", "cat": "Tech", "name": "Invesco Nasdaq-100 ETF"},
    {"ticker": "SPY", "cat": "Global", "name": "SPDR S&P 500 ETF"},
    {"ticker": "ICLN", "cat": "Energy", "name": "iShares Global Clean Energy"},
    {"ticker": "EQNR.OL", "cat": "Norwegian", "name": "Equinor ASA (Energy)"},
    {"ticker": "DNB.OL", "cat": "Norwegian", "name": "DNB Bank ASA"},
    {"ticker": "MOWI.OL", "cat": "Norwegian", "name": "Mowi ASA (Seafood)"},
    {"ticker": "EUNL.DE", "cat": "Global", "name": "iShares Core MSCI World"}
]


def fetch_and_save_fund_profile(ticker_symbol, category_tag="Custom"):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        isin = info.get('isin', ticker_symbol)
        name = info.get('longName', info.get('shortName', ticker_symbol))
        desc = info.get('longBusinessSummary', 'No description available.')
        holdings, sectors = [], []
        if 'sectorWeightings' in info:
            for s in info['sectorWeightings'][:3]:
                for k, v in s.items(): sectors.append(f"{k}: {v * 100:.1f}%")
        profile = FundProfile(isin=isin, ticker=ticker_symbol, name=name, description=desc,
                              top_holdings=json.dumps(holdings), sector_info=json.dumps(sectors), category=category_tag)
        session.merge(profile)
        session.commit()
        return isin, name
    except Exception as e:
        return None, str(e)


def update_price_history(isin):
    profile = session.query(FundProfile).filter_by(isin=isin).first()
    if not profile: return
    ticker = yf.Ticker(profile.ticker)
    end = datetime.now()
    start = end - timedelta(days=3 * 365)
    hist = ticker.history(start=start, end=end)
    risk = ticker.info.get('morningStarRiskRating', 0.0)
    for date, row in hist.iterrows():
        if not session.query(FundPriceHistory).filter_by(isin=isin, date=date).first():
            session.add(FundPriceHistory(isin=isin, date=date, close_price=row['Close'], risk_rating=risk))
    session.commit()


def init_defaults():
    if not session.query(FundProfile).first():
        defaults = [("0P00018V9L.IR", "Global"), ("0P0001CTL0.IR", "Nordic"), ("0P00016TML.IR", "European")]
        for t, c in defaults:
            isin, _ = fetch_and_save_fund_profile(t, c)
            if isin: update_price_history(isin)


init_defaults()


def get_exchange_rates():
    try:
        t = yf.Tickers('EURNOK=X USDNOK=X')
        return {'EUR': t.tickers['EURNOK=X'].fast_info['last_price'],
                'USD': t.tickers['USDNOK=X'].fast_info['last_price']}
    except:
        return {'EUR': 11.5, 'USD': 10.5}


def get_latest_price(isin):
    rec = session.query(FundPriceHistory).filter_by(isin=isin).order_by(desc(FundPriceHistory.date)).first()
    return rec.close_price if rec else 1.0


# UPDATED: NOW REQUIRES user_id
def log_investment(user_id, amount, isin, alloc_type, risk, date_override=None, d_rate=0):
    rec_date = datetime.now()
    if date_override: rec_date = datetime.combine(date_override, datetime.min.time())

    price = 1.0
    if isin not in ['BANK', 'DEBT']:
        hist = session.query(FundPriceHistory).filter(FundPriceHistory.isin == isin,
                                                      FundPriceHistory.date <= rec_date).order_by(
            desc(FundPriceHistory.date)).first()
        price = hist.close_price if hist else get_latest_price(isin)
        if price == 1.0:
            update_price_history(isin)
            price = get_latest_price(isin)

    units = amount / price
    rec = FinancialRecord(
        user_id=user_id,  # Link record to user
        date=rec_date, year=rec_date.year, amount=amount, allocation_type=alloc_type,
        isin=isin, entry_price=price, units_owned=units,
        interest_rate_at_time=d_rate, risk_score=risk
    )
    session.add(rec)
    session.commit()


# UPDATED: Filters by user_id
def get_portfolio_df(user_id):
    records = session.query(FinancialRecord).filter_by(user_id=user_id).all()
    data = []
    for r in records:
        curr_price = 1.0
        if r.isin not in ['BANK', 'DEBT']: curr_price = get_latest_price(r.isin)
        cur_val = r.units_owned * curr_price
        prof = cur_val - r.amount
        ret = (prof / r.amount) * 100 if r.amount > 0 else 0
        name = r.allocation_type
        if r.isin not in ['BANK', 'DEBT']:
            prof_db = session.query(FundProfile).filter_by(isin=r.isin).first()
            if prof_db: name = prof_db.name
        data.append({
            "Allocation": name, "ISIN": r.isin, "Date": r.date.date(),
            "Invested": r.amount, "Current Value": cur_val, "Profit": prof, "Return (%)": ret,
            "Entry Price": r.entry_price
        })
    return pd.DataFrame(data)


# UPDATED: Filters by user_id
def plot_history(session, user_id, isin, currency_sym, rate):
    hist = session.query(FundPriceHistory).filter_by(isin=isin).order_by(FundPriceHistory.date).all()
    if not hist: return None
    prof = session.query(FundProfile).filter_by(isin=isin).first()
    name = prof.name if prof else isin
    df_h = pd.DataFrame([{"Date": h.date, "Price": h.close_price / rate} for h in hist])

    # Filter only THIS user's investments
    investments = session.query(FinancialRecord).filter_by(isin=isin, user_id=user_id).all()
    df_inv = pd.DataFrame(
        [{"Date": i.date, "Price": i.entry_price / rate, "Amt": i.amount / rate} for i in investments])

    fig = px.line(df_h, x="Date", y="Price", title=f"{name} ({currency_sym})")
    if not df_inv.empty:
        fig.add_trace(go.Scatter(
            x=df_inv["Date"], y=df_inv["Price"], mode='markers+text',
            marker=dict(size=14, color='Gold', symbol='star', line=dict(width=2, color='DarkSlateGrey')),
            text=[f"{a:,.0f}{currency_sym}" for a in df_inv["Amt"]], textposition="top center",
            textfont=dict(color='black')
        ))
    fig.update_layout(template="plotly_white")
    return fig


# --- 4. MAIN APP LOGIC ---

st.set_page_config(page_title="FinStrat Pro", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

# ==========================================
# 🔐 AUTHENTICATION PAGE (Show if not logged in)
# ==========================================
if st.session_state.user_id is None:
    st.title("🔐 FinStrat: Secure Login")

    tab_login, tab_reg = st.tabs(["Login", "Register New User"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                user = check_login(username, password)
                if user:
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.success(f"Welcome back, {user.username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with tab_reg:
        with st.form("reg_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            submit_reg = st.form_submit_button("Create Account")

            if submit_reg:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 4:
                    st.error("Password must be at least 4 characters.")
                else:
                    success, msg = register_user(new_user, new_pass)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

    # Database Reset for Auth Upgrade
    st.divider()
    with st.expander("🛠 Troubleshooting: Database Error?"):
        st.caption("If you see an error about 'no such column: users', your database is outdated.")
        if st.button("RESET DATABASE (Deletes All Data)"):
            Base.metadata.drop_all(engine)
            Base.metadata.create_all(engine)
            st.success("Database reset. You can now register a new user.")

# ==========================================
# 📊 MAIN DASHBOARD (Show ONLY if logged in)
# ==========================================
else:
    # Sidebar: User Info & Logout
    st.sidebar.title(f"👤 {st.session_state.username}")
    if st.sidebar.button("Log Out"):
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()

    st.sidebar.markdown("---")

    # ------------------------------------
    # (PASTE ALL PREVIOUS DASHBOARD CODE HERE)
    # The code below is the adapted version of your previous dashboard
    # ------------------------------------

    st.title("💰 FinStrat: Dynamic Wealth Manager")

    # SIDEBAR SETTINGS
    st.sidebar.header("🌍 Settings")
    rates = get_exchange_rates()
    curr_opt = st.sidebar.radio("Currency", ["NOK", "EUR", "USD"])
    if curr_opt == "NOK":
        rate = 1.0; sym = "kr"
    elif curr_opt == "EUR":
        rate = rates['EUR']; sym = "€"
    else:
        rate = rates['USD']; sym = "$"

    st.sidebar.markdown("---")
    st.sidebar.header("📥 Fund Discovery")
    import_mode = st.sidebar.selectbox("Add Funds via:", ["Quick Recommendations", "Manual Ticker Search"])
    if import_mode == "Quick Recommendations":
        for fund in SUGGESTED_FUNDS:
            exists = session.query(FundProfile).filter_by(ticker=fund['ticker']).first()
            col1, col2 = st.sidebar.columns([3, 1])
            col1.text(fund['name'])
            if not exists:
                if col2.button("Add", key=fund['ticker']):
                    with st.spinner("Fetching..."):
                        isin, res = fetch_and_save_fund_profile(fund['ticker'], fund['cat'])
                        if isin:
                            update_price_history(isin)
                            st.sidebar.success("Added!")
                            st.rerun()
            else:
                col2.caption("✅")
    else:
        with st.sidebar.form("import_form"):
            new_ticker = st.text_input("Ticker Symbol")
            new_cat = st.selectbox("Category", ["Global", "Nordic", "Tech", "Energy", "Custom"])
            if st.form_submit_button("Fetch & Add"):
                with st.spinner("Scraping..."):
                    isin, res = fetch_and_save_fund_profile(new_ticker, new_cat)
                    if isin:
                        update_price_history(isin)
                        st.success(f"Added: {res}")

    # TABS
    tab1, tab2 = st.tabs(["🚀 Allocation Strategy", "📊 Portfolio Dashboard"])

    with tab1:
        st.subheader("New Cash Allocation")
        all_funds = session.query(FundProfile).all()

        with st.expander("📂 Your Active Fund Universe", expanded=True):
            if not all_funds:
                st.warning("No funds found! Add via Sidebar.")
            else:
                u_cols = st.columns(3)
                for i, f in enumerate(all_funds):
                    latest_p = get_latest_price(f.isin)
                    with u_cols[i % 3]:
                        st.markdown(f"**{f.name}**")
                        st.caption(f"Price: {latest_p:.2f}")

        st.markdown("---")
        mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)
        amount = st.number_input("Amount to Invest (NOK)", step=5000.0)
        sim_date = st.date_input("Date") if mode == "Backtest" else datetime.now()

        if amount > 0:
            global_f = next((f for f in all_funds if f.category == 'Global'), all_funds[0] if all_funds else None)
            nordic_f = next((f for f in all_funds if f.category == 'Nordic'), None)
            rec_map = {}
            if global_f: rec_map[global_f.isin] = amount * 0.7
            if nordic_f: rec_map[nordic_f.isin] = amount * 0.2
            buffer = amount - sum(rec_map.values())
            if buffer > 0: rec_map['BANK'] = buffer

            st.info("💡 **AI Suggestion:** Customize below.")

            if 'form_id' not in st.session_state: st.session_state.form_id = 0

            with st.form(key=f"alloc_form_{st.session_state.form_id}"):
                allocs = {}
                st.subheader("Customize Your Split")
                c1, c2 = st.columns(2)
                allocs['BANK'] = c1.number_input("Savings / Buffer", value=float(rec_map.get('BANK', 0.0)),
                                                 key=f"bank_{st.session_state.form_id}")
                allocs['DEBT'] = c2.number_input("Debt Paydown (5.06%)", value=0.0,
                                                 key=f"debt_{st.session_state.form_id}")

                st.markdown("#### Market Funds")
                cols = st.columns(2)
                for i, fund in enumerate(all_funds):
                    default_val = float(rec_map.get(fund.isin, 0.0))
                    with cols[i % 2]:
                        st.markdown(f"**{fund.name}**")
                        allocs[fund.isin] = st.number_input(f"Invest in {fund.ticker}", value=default_val,
                                                            key=f"inv_{fund.isin}_{st.session_state.form_id}")

                st.markdown("---")
                total = sum(allocs.values())

                if abs(total - amount) < 1.0:
                    st.success(f"Total: {total:,.0f} NOK matches.")
                    ready = True
                else:
                    st.warning(f"Allocated: {total:,.0f} / {amount:,.0f} NOK")
                    ready = False

                submitted = st.form_submit_button("Confirm Investment")
                if submitted:
                    if ready:
                        for k, v in allocs.items():
                            if v > 0:
                                n = "Savings" if k == 'BANK' else ("Debt" if k == 'DEBT' else "Fund")
                                # Pass User ID here
                                log_investment(st.session_state.user_id, v, k, n, 4,
                                               sim_date if mode == "Backtest" else None)

                        st.success("Investment Logged!")
                        st.session_state.form_id += 1
                        st.rerun()

    with tab2:
        st.subheader(f"Portfolio ({curr_opt})")
        # Fetch ONLY this user's data
        df = get_portfolio_df(st.session_state.user_id)

        if not df.empty:
            df_d = df.copy()
            for c in ['Invested', 'Current Value', 'Profit', 'Entry Price']:
                df_d[c] = df_d[c] / rate

            df_grouped = df_d.groupby('Allocation', as_index=False).agg({
                'Invested': 'sum', 'Current Value': 'sum', 'Profit': 'sum', 'ISIN': 'first'
            })
            df_grouped['Return (%)'] = ((df_grouped['Current Value'] - df_grouped['Invested']) / df_grouped[
                'Invested']) * 100

            tot_inv = df_grouped['Invested'].sum()
            tot_val = df_grouped['Current Value'].sum()
            roi = ((tot_val - tot_inv) / tot_inv) * 100 if tot_inv else 0

            m1, m2 = st.columns(2)
            m1.metric("Total Value", f"{sym} {tot_val:,.0f}")
            m2.metric("Total ROI", f"{roi:.2f}%", delta=f"{tot_val - tot_inv:,.0f} {sym}")

            fund_isins = [x for x in df['ISIN'].unique() if x not in ['BANK', 'DEBT']]
            if fund_isins:
                sel = st.selectbox("Select History", fund_isins,
                                   format_func=lambda x: session.query(FundProfile).filter_by(isin=x).first().name)
                # Pass User ID here too
                fig = plot_history(session, st.session_state.user_id, sel, sym, rate)
                if fig: st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Holdings Summary")
            st.dataframe(df_grouped[['Allocation', 'Invested', 'Current Value', 'Profit', 'Return (%)']].style.format({
                'Invested': f"{sym} {{:,.0f}}", 'Current Value': f"{sym} {{:,.0f}}",
                'Profit': f"{sym} {{:,.0f}}", 'Return (%)': "{:+.2f}%"
            }), use_container_width=True)

            with st.expander("View Transaction Log"):
                st.dataframe(df_d[['Date', 'Allocation', 'Invested', 'Entry Price']].style.format({
                    'Date': '{:%Y-%m-%d}', 'Invested': f"{sym} {{:,.0f}}", 'Entry Price': f"{sym} {{:,.2f}}"
                }))
        else:
            st.info("No investments found for this user.")