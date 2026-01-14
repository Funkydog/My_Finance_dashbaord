import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import json
import hashlib
import google.generativeai as genai  # NEW LIBRARY
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, desc, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timedelta

# ==========================================
# 🧠 AI AGENT CONFIGURATION
# ==========================================
SYSTEM_PROMPT_TEMPLATE = """
You are a CFA-certified Chief Investment Officer (CIO) specializing in global funds and de-resking strategies.

**User Profile:**
- **Tax Residence:** {tax_residence}
- **Investment Amount:** {amount} {currency}
- **Risk Profile:** {risk_profile}

**Tax Rules to Apply:**
1. **Norway:** Prioritize 'Aksjesparekonto' (ASK) eligible funds (EEA/EU domiciled).
2. **France:** Prioritize 'PEA' eligible funds (European equities).
3. **USA:** Avoid PFIC (Foreign Mutual Funds), favor US-domiciled ETFs.

**Available Funds (User's Universe):**
{fund_list_json}

**Your Mission:**
Allocate the cash into the "Available Funds" to create an optimal portfolio based on the tax rules and risk profile.
- You must ONLY use the funds provided in the list.
- If the universe is poor, suggest the best available option.

**Output Format:**
Respond with a strict JSON object (no markdown):
{{
  "allocations": {{ "ISIN_CODE_1": AMOUNT_NUMBER, "ISIN_CODE_2": AMOUNT_NUMBER }},
  "reasoning": "One sentence explaining the tax/risk logic used."
}}
"""

# ==========================================
# 1. DATABASE SETUP
# ==========================================
Base = declarative_base()
engine = create_engine('sqlite:///personal_finance_v9_agent.db', echo=False)


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


# ==========================================
# 2. STRATEGY AGENT (UPDATED FOR GEMINI)
# ==========================================
class StrategyAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_allocation(self, user_profile, funds, amount):
        # 1. Prepare Data Context
        funds_context = []
        for f in funds:
            funds_context.append({
                "isin": f.isin, "name": f.name, "region": f.region,
                "category": f.category
            })

        # 2. Build Prompt
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tax_residence=user_profile['tax_residence'],
            amount=amount,
            currency="NOK",
            risk_profile=user_profile.get('risk', 'Growth'),
            fund_list_json=json.dumps(funds_context, indent=2)
        )

        # 3. CALL GEMINI
        if not self.api_key:
            return {}, "⚠️ No API Key provided. Please enter your Google Gemini Key in the sidebar."

        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')

            # Request response
            response = model.generate_content(prompt)

            # Clean response (Gemini often wraps JSON in ```json blocks)
            raw_text = response.text
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()

            # Parse JSON
            result = json.loads(clean_text)

            return result.get('allocations', {}), result.get('reasoning', "AI provided no reasoning.")

        except Exception as e:
            return {}, f"AI Error: {str(e)}"


# ==========================================
# 3. CORE LOGIC & HELPERS
# ==========================================
SUGGESTED_FUNDS = [
    {"ticker": "SPY", "cat": "Global", "name": "SPDR S&P 500 (US)", "countries": ["USA", "Global"]},
    {"ticker": "EQNR.OL", "cat": "Energy", "name": "Equinor ASA", "countries": ["Norway"]},
    {"ticker": "DNB.OL", "cat": "Finance", "name": "DNB Bank", "countries": ["Norway"]},
    {"ticker": "CW8.PA", "cat": "Global", "name": "Amundi MSCI World (PEA)", "countries": ["France"]},
    {"ticker": "EUNL.DE", "cat": "Global", "name": "iShares Core MSCI World", "countries": ["Germany", "Norway"]},
    {"ticker": "NOVO-B.CO", "cat": "Health", "name": "Novo Nordisk", "countries": ["Denmark", "Global"]},
]


def make_hash(p): return hashlib.sha256(str.encode(p)).hexdigest()


def check_login(u, p): return session.query(User).filter_by(username=u, password_hash=make_hash(p)).first()


def register_user(u, p, c):
    if session.query(User).filter_by(username=u).first(): return False, "User exists."
    session.add(User(username=u, password_hash=make_hash(p), tax_residences=",".join(c)))
    session.commit()
    # Default fund (KLP Global)
    fetch_and_save_fund_profile("0P00018V9L.IR", "Global")
    return True, "Created."


def fetch_and_save_fund_profile(ticker, cat="Custom", user_id=None):
    try:
        t = yf.Ticker(ticker)
        i = t.info
        isin = i.get('isin', ticker)
        name = i.get('longName', ticker)
        reg = "Global"
        if any(x in ticker for x in [".OL", ".PA", ".DE", ".CO"]): reg = "EEA"
        if "US" in isin: reg = "US"

        prof = FundProfile(isin=isin, ticker=ticker, name=name, category=cat, region=reg)
        session.merge(prof)
        session.commit()
        if user_id:
            if not session.query(UserFundSelection).filter_by(user_id=user_id, isin=isin).first():
                session.add(UserFundSelection(user_id=user_id, isin=isin))
                session.commit()
        return isin, name
    except:
        return None, "Error"


def update_price_history(isin):
    p = session.query(FundProfile).filter_by(isin=isin).first()
    if not p: return
    try:
        hist = yf.Ticker(p.ticker).history(period="1mo")
        if not hist.empty:
            for date, row in hist.iterrows():
                if not session.query(FundPriceHistory).filter_by(isin=isin, date=date).first():
                    session.add(FundPriceHistory(isin=isin, date=date, close_price=row['Close']))
            session.commit()
    except:
        pass


def get_latest_price(isin):
    rec = session.query(FundPriceHistory).filter_by(isin=isin).order_by(desc(FundPriceHistory.date)).first()
    return rec.close_price if rec else 1.0


def log_investment(user_id, amount, isin, alloc_type, risk, date_override=None):
    d = datetime.combine(date_override, datetime.min.time()) if date_override else datetime.now()
    p = 1.0
    if isin not in ['BANK', 'DEBT']:
        if not date_override: update_price_history(isin)
        p = get_latest_price(isin)

    session.add(
        FinancialRecord(user_id=user_id, date=d, year=d.year, amount=amount, allocation_type=alloc_type, isin=isin,
                        entry_price=p, units_owned=amount / p, risk_score=risk))
    session.commit()


def get_portfolio_df(user_id):
    recs = session.query(FinancialRecord).filter_by(user_id=user_id).all()
    data = []
    for r in recs:
        p = 1.0
        name = r.allocation_type
        if r.isin not in ['BANK', 'DEBT']:
            p = get_latest_price(r.isin)
            prof = session.query(FundProfile).filter_by(isin=r.isin).first()
            if prof: name = prof.name

        data.append({
            "Allocation": name, "ISIN": r.isin, "Date": r.date.date(),
            "Invested": r.amount, "Current Value": r.units_owned * p,
            "Profit": (r.units_owned * p) - r.amount, "Entry Price": r.entry_price
        })
    return pd.DataFrame(data)


def get_exchange_rates():
    try:
        t = yf.Tickers('EURNOK=X USDNOK=X')
        return {'EUR': t.tickers['EURNOK=X'].fast_info['last_price'],
                'USD': t.tickers['USDNOK=X'].fast_info['last_price']}
    except:
        return {'EUR': 11.5, 'USD': 10.5}


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


# ==========================================
# 4. MAIN UI
# ==========================================
st.set_page_config(page_title="FinStrat AI", layout="wide")
if 'user_id' not in st.session_state: st.session_state.user_id = None

# --- AUTH SCREEN ---
if not st.session_state.user_id:
    st.title("🤖 FinStrat AI: Agent Login")
    t1, t2 = st.tabs(["Login", "Register"])
    with t1:
        with st.form("l"):
            u, p = st.text_input("User"), st.text_input("Pass", type="password")
            if st.form_submit_button("Go"):
                user = check_login(u, p)
                if user:
                    st.session_state.user_id = user.id
                    st.session_state.residences = user.tax_residences
                    st.session_state.username = user.username
                    st.rerun()
    with t2:
        with st.form("r"):
            nu, np = st.text_input("User"), st.text_input("Pass", type="password")
            ct = st.multiselect("Tax", ["Norway", "France", "USA", "Germany", "Sweden"])
            if st.form_submit_button("Create"):
                register_user(nu, np, ct)
                st.success("Done")

    st.divider()
    if st.button("RESET DATABASE"):
        Base.metadata.drop_all(engine);
        Base.metadata.create_all(engine);
        st.success("Reset!");
        st.rerun()

else:
    # --- DASHBOARD ---
    st.sidebar.title(f"👤 {st.session_state.username}")
    st.sidebar.info(f"Tax: {st.session_state.residences}")

    # API KEY INPUT
    st.sidebar.markdown("---")
    gemini_key = st.sidebar.text_input("🔑 Gemini API Key", type="password", help="Get key from aistudio.google.com")

    if st.sidebar.button("Logout"): st.session_state.user_id = None; st.rerun()

    # 1. ADD FUNDS
    st.sidebar.markdown("---")
    st.sidebar.header("Universe Builder")

    user_res = st.session_state.residences
    filtered_suggestions = [f for f in SUGGESTED_FUNDS if
                            "Global" in f['countries'] or any(c in user_res for c in f['countries'])]

    for f in filtered_suggestions:
        c1, c2 = st.sidebar.columns([3, 1])
        c1.text(f['name'])
        if c2.button("Add", key=f['ticker']):
            fetch_and_save_fund_profile(f['ticker'], f['cat'], st.session_state.user_id)
            update_price_history(session.query(FundProfile).filter_by(ticker=f['ticker']).first().isin)

    # 2. MAIN TABS
    tab1, tab2 = st.tabs(["🧠 AI Strategy Agent", "📊 Portfolio"])

    # --- TAB 1: AI AGENT ---
    with tab1:
        st.subheader("AI Portfolio Architect")

        # Get User Universe
        links = session.query(UserFundSelection).filter_by(user_id=st.session_state.user_id).all()
        my_funds = session.query(FundProfile).filter(FundProfile.isin.in_([l.isin for l in links])).all()

        if not my_funds:
            st.warning("Your universe is empty. Add funds from the sidebar first.")
        else:
            with st.expander("📂 My Active Fund Universe"):
                for f in my_funds: st.text(f"• {f.name} ({f.region})")

            st.markdown("---")
            c_a, c_b = st.columns(2)
            amount = c_a.number_input("Cash to Deploy (NOK)", step=5000.0)
            risk = c_b.select_slider("Risk Profile", options=["Conservative", "Moderate", "Growth", "Aggressive"],
                                     value="Growth")

            if amount > 0:
                # --- CALL THE AGENT ---
                agent = StrategyAgent(api_key=gemini_key)
                user_prof = {'tax_residence': st.session_state.residences, 'risk': risk}

                # If key is missing, warn but don't crash
                if not gemini_key:
                    st.warning("⚠️ Enter Gemini API Key in sidebar to get AI recommendations.")

                else:
                    with st.spinner("🤖 Consulting Gemini Pro Strategy Agent..."):
                        rec_map, reasoning = agent.get_allocation(user_prof, my_funds, amount)

                    st.markdown("### 🤖 Agent Recommendation")
                    if "Error" in reasoning:
                        st.error(reasoning)
                    else:
                        st.info(f"**Strategy Reasoning:** {reasoning}")

                        with st.form("agent_exec"):
                            allocs = {}
                            cols = st.columns(2)
                            # Pre-fill form with AI suggestions
                            for i, (isin, val) in enumerate(rec_map.items()):
                                # Match ISIN to Name
                                fname = next((f.name for f in my_funds if f.isin == isin), isin)
                                with cols[i % 2]:
                                    allocs[isin] = st.number_input(f"{fname}", value=float(val))

                            # Handle funds AI didn't pick (set to 0)
                            for f in my_funds:
                                if f.isin not in allocs:
                                    # Hidden logic to include them in the form just in case
                                    pass

                                    # Remaining for Buffer/Debt
                            rem = amount - sum(allocs.values())
                            if rem > 0:
                                st.caption(f"Remaining: {rem:,.0f} NOK")
                                c1, c2 = st.columns(2)
                                allocs['BANK'] = c1.number_input("Savings", value=rem)
                                allocs['DEBT'] = c2.number_input("Debt Paydown", value=0.0)

                            if st.form_submit_button("Execute Strategy"):
                                for isin, amt in allocs.items():
                                    if amt > 0:
                                        n = "Savings" if isin == 'BANK' else ("Debt" if isin == 'DEBT' else "Fund")
                                        log_investment(st.session_state.user_id, amt, isin, n, 4)
                                st.success("Executed!")
                                st.rerun()

    # --- TAB 2: PORTFOLIO ---
    with tab2:
        rates = get_exchange_rates()
        curr_opt = st.radio("Display Currency", ["NOK", "EUR", "USD"], horizontal=True)
        if curr_opt == "NOK":
            rate = 1.0; sym = "kr"
        elif curr_opt == "EUR":
            rate = rates['EUR']; sym = "€"
        else:
            rate = rates['USD']; sym = "$"

        df = get_portfolio_df(st.session_state.user_id)
        if not df.empty:
            df_d = df.copy()
            for c in ['Invested', 'Current Value', 'Profit', 'Entry Price']: df_d[c] = df_d[c] / rate

            df_g = df_d.groupby('Allocation', as_index=False).agg({
                'Invested': 'sum', 'Current Value': 'sum', 'Profit': 'sum', 'ISIN': 'first'
            })
            df_g['Return (%)'] = ((df_g['Current Value'] - df_g['Invested']) / df_g['Invested']) * 100

            tot_val = df_g['Current Value'].sum()
            roi = ((df_g['Current Value'].sum() - df_g['Invested'].sum()) / df_g['Invested'].sum()) * 100

            m1, m2 = st.columns(2)
            m1.metric("Total Value", f"{sym} {tot_val:,.0f}")
            m2.metric("Total ROI", f"{roi:.2f}%")

            fund_isins = [x for x in df['ISIN'].unique() if x not in ['BANK', 'DEBT']]
            if fund_isins:
                sel = st.selectbox("History", fund_isins,
                                   format_func=lambda x: session.query(FundProfile).filter_by(isin=x).first().name)
                fig = plot_history(session, st.session_state.user_id, sel, sym, rate)
                if fig: st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_g[['Allocation', 'Invested', 'Current Value', 'Return (%)']].style.format({
                'Invested': f"{sym} {{:,.0f}}", 'Current Value': f"{sym} {{:,.0f}}", 'Return (%)': "{:+.2f}%"
            }), use_container_width=True)
        else:
            st.info("No investments found.")