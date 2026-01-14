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
engine = create_engine('sqlite:///personal_finance_v8.db', echo=False)


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    tax_residences = Column(String, default="Norway")


class FundProfile(Base):
    __tablename__ = 'fund_profiles'
    isin = Column(String, primary_key=True)
    ticker = Column(String)
    name = Column(String)
    description = Column(Text)
    top_holdings = Column(Text)
    sector_info = Column(Text)
    category = Column(String)
    region = Column(String, default="Global")


class UserFundSelection(Base):
    __tablename__ = 'user_fund_selections'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    isin = Column(String, ForeignKey('fund_profiles.isin'))


class FinancialRecord(Base):
    __tablename__ = 'financial_records'
    id = Column(Integer, primary_key=True)
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

# --- 2. FUND DATABASE ---
SUGGESTED_FUNDS = [
    {"ticker": "SPY", "cat": "Global", "name": "SPDR S&P 500 ETF (US)", "countries": ["USA", "Global"]},
    {"ticker": "QQQ", "cat": "Tech", "name": "Invesco QQQ (Nasdaq-100)", "countries": ["USA", "Global"]},
    {"ticker": "VT", "cat": "Global", "name": "Vanguard Total World Stock", "countries": ["USA", "Global"]},
    {"ticker": "DNB.OL", "cat": "Norwegian", "name": "DNB Bank ASA", "countries": ["Norway"]},
    {"ticker": "EQNR.OL", "cat": "Energy", "name": "Equinor ASA", "countries": ["Norway"]},
    {"ticker": "MOWI.OL", "cat": "Seafood", "name": "Mowi ASA", "countries": ["Norway"]},
    {"ticker": "YAR.OL", "cat": "Materials", "name": "Yara International", "countries": ["Norway"]},
    {"ticker": "KOG.OL", "cat": "Defense", "name": "Kongsberg Gruppen", "countries": ["Norway"]},
    {"ticker": "ORK.OL", "cat": "Consumer", "name": "Orkla ASA", "countries": ["Norway"]},
    {"ticker": "CW8.PA", "cat": "Global", "name": "Amundi MSCI World (PEA)", "countries": ["France"]},
    {"ticker": "ESE.PA", "cat": "Global", "name": "BNP Paribas S&P 500 (PEA)", "countries": ["France"]},
    {"ticker": "TTE.PA", "cat": "Energy", "name": "TotalEnergies SE", "countries": ["France"]},
    {"ticker": "MC.PA", "cat": "Luxury", "name": "LVMH Moët Hennessy", "countries": ["France"]},
    {"ticker": "AI.PA", "cat": "Materials", "name": "Air Liquide", "countries": ["France"]},
    {"ticker": "OR.PA", "cat": "Beauty", "name": "L'Oréal", "countries": ["France"]},
    {"ticker": "EUNL.DE", "cat": "Global", "name": "iShares Core MSCI World (UCITS)",
     "countries": ["Germany", "Norway", "Sweden", "Denmark"]},
    {"ticker": "SAP.DE", "cat": "Tech", "name": "SAP SE", "countries": ["Germany"]},
    {"ticker": "SIE.DE", "cat": "Industrial", "name": "Siemens AG", "countries": ["Germany"]},
    {"ticker": "INVE-B.ST", "cat": "Finance", "name": "Investor AB", "countries": ["Sweden"]},
    {"ticker": "VOLV-B.ST", "cat": "Industrial", "name": "Volvo AB", "countries": ["Sweden"]},
    {"ticker": "ATCO-A.ST", "cat": "Industrial", "name": "Atlas Copco", "countries": ["Sweden"]},
    {"ticker": "NOVO-B.CO", "cat": "Health", "name": "Novo Nordisk", "countries": ["Denmark", "Global"]},
    {"ticker": "DSV.CO", "cat": "Logistics", "name": "DSV A/S", "countries": ["Denmark"]},
    {"ticker": "HSBA.L", "cat": "Finance", "name": "HSBC Holdings", "countries": ["UK"]},
    {"ticker": "AZN.L", "cat": "Health", "name": "AstraZeneca", "countries": ["UK"]},
]


# --- 3. AUTH & LOGIC ---

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_login(username, password):
    pwd_hash = make_hash(password)
    return session.query(User).filter_by(username=username, password_hash=pwd_hash).first()


def register_user(username, password, countries):
    if session.query(User).filter_by(username=username).first(): return False, "Username exists."
    country_str = ",".join(countries)
    new_user = User(username=username, password_hash=make_hash(password), tax_residences=country_str)
    session.add(new_user)
    session.commit()
    defaults = ["0P00018V9L.IR"]
    for ticker in defaults:
        prof = session.query(FundProfile).filter_by(ticker=ticker).first()
        if prof: add_fund_to_user_universe(new_user.id, prof.isin)
    return True, "User created! Please login."


def add_fund_to_user_universe(user_id, isin):
    exists = session.query(UserFundSelection).filter_by(user_id=user_id, isin=isin).first()
    if not exists:
        link = UserFundSelection(user_id=user_id, isin=isin)
        session.add(link)
        session.commit()


def fetch_and_save_fund_profile(ticker_symbol, category_tag="Custom", user_id_to_link=None):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        isin = info.get('isin', ticker_symbol)
        name = info.get('longName', info.get('shortName', ticker_symbol))
        desc = info.get('longBusinessSummary', 'No description.')

        region = "Global"
        if any(x in ticker_symbol for x in [".OL", ".PA", ".DE", ".ST", ".CO"]): region = "EEA"
        if "US" in isin or "SPY" in ticker_symbol: region = "US"

        holdings, sectors = [], []
        if 'sectorWeightings' in info:
            for s in info['sectorWeightings'][:3]:
                for k, v in s.items(): sectors.append(f"{k}: {v * 100:.1f}%")

        profile = FundProfile(
            isin=isin, ticker=ticker_symbol, name=name, description=desc,
            top_holdings=json.dumps(holdings), sector_info=json.dumps(sectors),
            category=category_tag, region=region
        )
        session.merge(profile)
        session.commit()

        if user_id_to_link: add_fund_to_user_universe(user_id_to_link, isin)
        return isin, name
    except Exception as e:
        return None, str(e)


def update_price_history(isin):
    """Fetches latest 3 years of data from Yahoo."""
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
        defaults = [("0P00018V9L.IR", "Global"), ("0P0001CTL0.IR", "Nordic")]
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


# --- UPDATED LOG_INVESTMENT with AUTO-UPDATE ---
def log_investment(user_id, amount, isin, alloc_type, risk, date_override=None, d_rate=0):
    rec_date = datetime.now()
    if date_override: rec_date = datetime.combine(date_override, datetime.min.time())

    price = 1.0
    if isin not in ['BANK', 'DEBT']:
        # 1. AUTO-UPDATE CHECK (Only in Live Mode)
        if not date_override:
            # Check if we have recent data (last 24 hours)
            latest_rec = session.query(FundPriceHistory).filter_by(isin=isin).order_by(
                desc(FundPriceHistory.date)).first()
            is_stale = False
            if not latest_rec:
                is_stale = True
            elif latest_rec.date < datetime.now() - timedelta(hours=24):
                is_stale = True

            # If stale, update immediately
            if is_stale:
                # We use a placeholder print or spinner in the UI, but here we just run logic
                update_price_history(isin)

        # 2. GET PRICE
        hist = session.query(FundPriceHistory).filter(FundPriceHistory.isin == isin,
                                                      FundPriceHistory.date <= rec_date).order_by(
            desc(FundPriceHistory.date)).first()
        price = hist.close_price if hist else get_latest_price(isin)

        # Safety net: if still 1.0 (no history found), try one last force update
        if price == 1.0 and not date_override:
            update_price_history(isin)
            price = get_latest_price(isin)

    units = amount / price
    rec = FinancialRecord(
        user_id=user_id, date=rec_date, year=rec_date.year, amount=amount, allocation_type=alloc_type,
        isin=isin, entry_price=price, units_owned=units,
        interest_rate_at_time=d_rate, risk_score=risk
    )
    session.add(rec)
    session.commit()


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


def plot_history(session, user_id, isin, currency_sym, rate):
    hist = session.query(FundPriceHistory).filter_by(isin=isin).order_by(FundPriceHistory.date).all()
    if not hist: return None
    prof = session.query(FundProfile).filter_by(isin=isin).first()
    name = prof.name if prof else isin
    df_h = pd.DataFrame([{"Date": h.date, "Price": h.close_price / rate} for h in hist])
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


# --- 4. MAIN APP ---

st.set_page_config(page_title="FinStrat Pro", layout="wide")

if 'user_id' not in st.session_state: st.session_state.user_id = None
if 'username' not in st.session_state: st.session_state.username = None
if 'residences' not in st.session_state: st.session_state.residences = []

if st.session_state.user_id is None:
    st.title("🔐 FinStrat: Secure Login")
    t1, t2 = st.tabs(["Login", "Register"])
    with t1:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                user = check_login(u, p)
                if user:
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.session_state.residences = user.tax_residences.split(',')
                    st.success("Success!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    with t2:
        with st.form("reg"):
            nu = st.text_input("New Username")
            np = st.text_input("New Password", type="password")
            countries = st.multiselect("Tax Residence(s)",
                                       ["Norway", "France", "USA", "UK", "Germany", "Sweden", "Denmark"],
                                       default=["Norway"])
            if st.form_submit_button("Create Account"):
                if len(np) < 4:
                    st.error("Password too short.")
                else:
                    ok, msg = register_user(nu, np, countries)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    st.divider()
    with st.expander("🛠 Troubleshooting"):
        if st.button("RESET DATABASE"):
            Base.metadata.drop_all(engine)
            Base.metadata.create_all(engine)
            st.success("Database Reset.")
else:
    st.sidebar.title(f"👤 {st.session_state.username}")
    st.sidebar.caption(f"📍 {', '.join(st.session_state.residences)}")
    if st.sidebar.button("Log Out"):
        st.session_state.user_id = None
        st.rerun()

    st.sidebar.markdown("---")


    def get_tax_badge(fund_region):
        badges = []
        user_countries = st.session_state.residences
        if "Norway" in user_countries and fund_region == "EEA": badges.append("✅ ASK")
        if "France" in user_countries and fund_region == "EEA": badges.append("🇫🇷 PEA")
        if "USA" in user_countries and fund_region != "US": badges.append("⚠️ PFIC")
        return " | ".join(badges)


    # --- PRICE UPDATES ---
    st.sidebar.header("🔄 Price Updates")
    user_links = session.query(UserFundSelection).filter_by(user_id=st.session_state.user_id).all()
    user_fund_isins = [l.isin for l in user_links]

    if user_fund_isins:
        fund_map = {f.isin: f.name for f in
                    session.query(FundProfile).filter(FundProfile.isin.in_(user_fund_isins)).all()}
        selected_update = st.sidebar.multiselect("Select Funds:", user_fund_isins,
                                                 format_func=lambda x: fund_map.get(x, x), default=user_fund_isins)
        if st.sidebar.button("Update Selected Prices"):
            if not selected_update:
                st.sidebar.error("None selected.")
            else:
                progress = st.sidebar.progress(0)
                for i, isin_code in enumerate(selected_update):
                    update_price_history(isin_code)
                    progress.progress((i + 1) / len(selected_update))
                st.sidebar.success("Prices Updated!")
                st.rerun()
    else:
        st.sidebar.info("Add funds to enable updates.")

    # --- SMART DISCOVERY ---
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Smart Discovery")
    mode = st.sidebar.selectbox("Add Funds via:", ["Local Recommendations", "Manual Search"])
    if mode == "Local Recommendations":
        st.sidebar.caption("Curated for you:")
        relevant_funds = []
        user_res = st.session_state.residences
        for fund in SUGGESTED_FUNDS:
            if "Global" in fund['countries'] or any(c in user_res for c in fund['countries']):
                relevant_funds.append(fund)
        for fund in relevant_funds:
            prof = session.query(FundProfile).filter_by(ticker=fund['ticker']).first()
            is_linked = False
            if prof:
                link = session.query(UserFundSelection).filter_by(user_id=st.session_state.user_id,
                                                                  isin=prof.isin).first()
                if link: is_linked = True
            c1, c2 = st.sidebar.columns([3, 1])
            flag = ""
            if "Norway" in fund['countries']:
                flag = "🇳🇴 "
            elif "France" in fund['countries']:
                flag = "🇫🇷 "
            elif "USA" in fund['countries']:
                flag = "🇺🇸 "
            c1.text(f"{flag}{fund['name']}")
            if not is_linked:
                if c2.button("Add", key=f"add_{fund['ticker']}"):
                    with st.spinner("Adding..."):
                        isin, res = fetch_and_save_fund_profile(fund['ticker'], fund['cat'], st.session_state.user_id)
                        if isin:
                            update_price_history(isin)
                            st.rerun()
            else:
                c2.caption("✅")
    else:
        with st.sidebar.form("import"):
            nt = st.text_input("Ticker (e.g. EQNR.OL)")
            nc = st.selectbox("Cat", ["Global", "Nordic", "Tech"])
            if st.form_submit_button("Add"):
                with st.spinner("Fetching..."):
                    isin, res = fetch_and_save_fund_profile(nt, nc, st.session_state.user_id)
                    if isin:
                        update_price_history(isin)
                        st.success(f"Added {res}")

    # DISPLAY SETTINGS
    st.sidebar.markdown("---")
    st.sidebar.header("🌍 Display")
    rates = get_exchange_rates()
    curr_opt = st.sidebar.radio("Currency", ["NOK", "EUR", "USD"])
    if curr_opt == "NOK":
        rate = 1.0; sym = "kr"
    elif curr_opt == "EUR":
        rate = rates['EUR']; sym = "€"
    else:
        rate = rates['USD']; sym = "$"

    tab1, tab2 = st.tabs(["🚀 Allocation", "📊 Portfolio"])

    with tab1:
        st.subheader("New Cash Allocation")
        # Refresh user links
        user_links = session.query(UserFundSelection).filter_by(user_id=st.session_state.user_id).all()
        user_isins = [l.isin for l in user_links]
        my_funds = session.query(FundProfile).filter(FundProfile.isin.in_(user_isins)).all()

        with st.expander("📂 My Fund Universe", expanded=True):
            if not my_funds:
                st.warning("Universe empty. Add funds from sidebar.")
            else:
                u_cols = st.columns(2)
                for i, f in enumerate(my_funds):
                    with u_cols[i % 2]:
                        st.markdown(f"**{f.name}**")
                        tx = get_tax_badge(f.region)
                        if tx: st.caption(tx)

        st.markdown("---")
        mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)
        amount = st.number_input("Amount (NOK)", step=5000.0)
        sim_date = st.date_input("Date") if mode == "Backtest" else datetime.now()

        if amount > 0:
            rec_map = {}
            funds_avail = [f.isin for f in my_funds]
            if len(funds_avail) >= 1: rec_map[funds_avail[0]] = amount * 0.7
            if len(funds_avail) >= 2: rec_map[funds_avail[1]] = amount * 0.2
            rem = amount - sum(rec_map.values())
            if rem > 0: rec_map['BANK'] = rem

            st.info("💡 **Strategy:** Using your personalized universe.")
            if 'form_id' not in st.session_state: st.session_state.form_id = 0

            with st.form(key=f"alloc_{st.session_state.form_id}"):
                allocs = {}
                c1, c2 = st.columns(2)
                allocs['BANK'] = c1.number_input("Savings", value=float(rec_map.get('BANK', 0.0)),
                                                 key=f"b_{st.session_state.form_id}")
                allocs['DEBT'] = c2.number_input("Debt Paydown", value=0.0, key=f"d_{st.session_state.form_id}")
                st.markdown("#### Market Funds")
                cols = st.columns(2)
                for i, fund in enumerate(my_funds):
                    def_val = float(rec_map.get(fund.isin, 0.0))
                    with cols[i % 2]:
                        st.markdown(f"**{fund.name}**")
                        allocs[fund.isin] = st.number_input("Invest", value=def_val,
                                                            key=f"i_{fund.isin}_{st.session_state.form_id}")

                if st.form_submit_button("Confirm"):
                    # TRIGGER AUTO-UPDATE MESSAGE
                    if mode == "Live":
                        with st.spinner("Refreshing fund prices from market..."):
                            pass  # The actual update happens inside log_investment

                    for k, v in allocs.items():
                        if v > 0:
                            n = "Savings" if k == 'BANK' else ("Debt" if k == 'DEBT' else "Fund")
                            log_investment(st.session_state.user_id, v, k, n, 4,
                                           sim_date if mode == "Backtest" else None)
                    st.success("Logged with latest prices!")
                    st.session_state.form_id += 1
                    st.rerun()

    with tab2:
        st.subheader(f"Portfolio ({curr_opt})")
        df = get_portfolio_df(st.session_state.user_id)
        if not df.empty:
            df_d = df.copy()
            for c in ['Invested', 'Current Value', 'Profit', 'Entry Price']: df_d[c] = df_d[c] / rate
            df_g = df_d.groupby('Allocation', as_index=False).agg(
                {'Invested': 'sum', 'Current Value': 'sum', 'Profit': 'sum', 'ISIN': 'first'})
            df_g['Return (%)'] = ((df_g['Current Value'] - df_g['Invested']) / df_g['Invested']) * 100
            m1, m2 = st.columns(2)
            m1.metric("Total Value", f"{sym} {df_g['Current Value'].sum():,.0f}")
            roi = ((df_g['Current Value'].sum() - df_g['Invested'].sum()) / df_g['Invested'].sum()) * 100
            m2.metric("Total ROI", f"{roi:.2f}%")

            user_owned_isins = [x for x in df['ISIN'].unique() if x not in ['BANK', 'DEBT']]
            if user_owned_isins:
                sel = st.selectbox("History", user_owned_isins,
                                   format_func=lambda x: session.query(FundProfile).filter_by(isin=x).first().name)
                fig = plot_history(session, st.session_state.user_id, sel, sym, rate)
                if fig: st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_g[['Allocation', 'Invested', 'Current Value', 'Return (%)']].style.format({
                'Invested': f"{sym} {{:,.0f}}", 'Current Value': f"{sym} {{:,.0f}}", 'Return (%)': "{:+.2f}%"
            }), use_container_width=True)
        else:
            st.info("No investments found.")