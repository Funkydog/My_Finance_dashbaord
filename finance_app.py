import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import json
import hashlib
import google.generativeai as genai
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, desc, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timedelta

# ==========================================
# 🧠 AI AGENT CONFIGURATION
# ==========================================
SYSTEM_PROMPT_TEMPLATE = """
You are a CFA-certified Chief Investment Officer (CIO) specializing in tax-efficient portfolio construction.

**User Profile:**
- **Tax Residence:** {tax_residence}
- **Investment Amount:** {amount} {currency}
- **Risk Profile:** {risk_profile} (Aggressive, Growth, Moderate, or Conservative)
- **Current Savings Buffer:** {current_savings} {currency} (Approximate)
- **Mortgage/Debt Interest Rate:** {debt_rate}%

**Tax Rules (Strict Compliance):**
1. **Norway:** Prioritize 'Aksjesparekonto' (ASK) eligible funds (EEA/EU domiciled ISINs).
2. **France:** Prioritize 'PEA' eligible funds (European equities).
3. **USA:** Avoid PFIC (Foreign Mutual Funds); favor US-domiciled ETFs.

**Available Funds (User's Universe):**
{fund_list_json}

**Current Portfolio Holdings (Existing Exposure):**
{portfolio_json}

**Investment Strategy (The "Hierarchy of Wealth"):**
Allocate the capital following this strict priority order:

1.  **The Fortress (Safety & Liquidity):**
    - Check the `BANK` / Savings option.
    - If `{current_savings}` is low (< 50,000), allocate heavily here first.
    - If Risk Profile is "Conservative", prioritize `BANK` regardless of savings.

2.  **Debt Arbitrage (Guaranteed Return):**
    - Check the `DEBT` / Paydown option.
    - If `{debt_rate}` is > 5.0%, this is a risk-free guaranteed return. Allocate significantly here (30-50%) unless the user is "Aggressive".
    - If `{debt_rate}` is < 3.5%, ignore this (cheap leverage).

3.  **The Growth Engine (The Core Portfolio):**
    - Allocate the remainder to Market Funds based on `{risk_profile}`:
        - **Aggressive/Growth:** 80% Core (Global/US Index) + 20% Satellites (Sector/Tech/Energy).
        - **Moderate:** 60% Core + 40% Safer Assets (Bond funds or Dividends).
        - **Conservative:** Focus on Capital Preservation.
    - **Geographic Bias:** Adjust the "Core" to fit the User's Tax Residence (e.g., Use EU funds for France/Norway).

**Constraint Checklist:**
- You must **ONLY** use the funds provided in the "Available Funds" list.
- If the universe is poor (e.g., only 1 fund), allocate 100% to the best option or `BANK`.

**Output Format:**
Respond with a strict JSON object (no markdown):
{{
  "allocations": {{ "ISIN_OR_TICKER_1": AMOUNT_NUMBER, "ISIN_OR_TICKER_2": AMOUNT_NUMBER }},
  "reasoning": "Concise explanation of the split, referencing tax efficiency and the debt/risk trade-off."
}}
"""

ANALYSIS_PROMPT_TEMPLATE = """
You are a Risk Manager reviewing a user's proposed trade.

**User Context:**
- **Profile:** {risk_profile}, Residence: {tax_residence}
- **Proposed Final Trade:**
{proposed_allocation_json}

**Current Portfolio Context:**
{portfolio_json}

**Task:**
Analyze this specific allocation.
1. Does it align with their risk profile?
2. Is it tax-efficient for {tax_residence}?
3. Does it fix or worsen any portfolio concentration issues?

**Output:**
Provide a short, 3-sentence critique. Start with "✅ Approved" or "⚠️ Caution".
"""

# ==========================================
# 1. DATABASE SETUP
# ==========================================
Base = declarative_base()
engine = create_engine('sqlite:///personal_finance_v18_final.db', echo=False)


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
# 2. SUGGESTED FUNDS
# ==========================================
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


# ==========================================
# 3. STRATEGY AGENT
# ==========================================
class StrategyAgent:
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name

    def get_allocation(self, user_profile, funds, amount, current_portfolio):
        funds_context = [{
            "isin": f.isin, "name": f.name, "region": f.region, "category": f.category
        } for f in funds]

        portfolio_context = json.dumps(current_portfolio, indent=2) if current_portfolio else "Empty Portfolio."

        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tax_residence=user_profile['tax_residence'],
            amount=amount,
            currency="NOK",
            risk_profile=user_profile.get('risk', 'Growth'),
            current_savings=user_profile.get('savings', 0),
            debt_rate=user_profile.get('debt_rate', 0.0),
            fund_list_json=json.dumps(funds_context, indent=2),
            portfolio_json=portfolio_context
        )
        return self._call_gemini_json(prompt)

    def analyze_user_edit(self, user_profile, proposed_allocations, funds, current_portfolio):
        readable_alloc = {}
        for isin, amt in proposed_allocations.items():
            if amt > 0:
                fname = next((f.name for f in funds if f.isin == isin), isin)
                readable_alloc[fname] = amt

        portfolio_context = json.dumps(current_portfolio, indent=2) if current_portfolio else "Empty Portfolio."

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            risk_profile=user_profile.get('risk', 'Growth'),
            tax_residence=user_profile['tax_residence'],
            proposed_allocation_json=json.dumps(readable_alloc, indent=2),
            portfolio_json=portfolio_context
        )
        return self._call_gemini_text(prompt)

    def _call_gemini_json(self, prompt):
        if not self.api_key: return {}, "⚠️ No API Key."
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            clean = response.text.replace("```json", "").replace("```", "").strip()
            res = json.loads(clean)
            return res.get('allocations', {}), res.get('reasoning', "")
        except Exception as e:
            return {}, f"AI Error: {str(e)}"

    def _call_gemini_text(self, prompt):
        if not self.api_key: return "⚠️ No API Key."
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI Error: {str(e)}"


# ==========================================
# 4. HELPERS
# ==========================================
def make_hash(p): return hashlib.sha256(str.encode(p)).hexdigest()


def check_login(u, p): return session.query(User).filter_by(username=u, password_hash=make_hash(p)).first()


def get_user_by_id(uid): return session.query(User).filter_by(id=uid).first()


def register_user(u, p, c):
    if session.query(User).filter_by(username=u).first(): return False, "User exists."
    session.add(User(username=u, password_hash=make_hash(p), tax_residences=",".join(c)))
    session.commit()
    fetch_and_save_fund_profile("0P00018V9L.IR", "Global")
    return True, "Created."


def fetch_and_save_fund_profile(ticker, cat="Custom", user_id=None, manual_isin=None):
    try:
        t = yf.Ticker(ticker)
        i = t.info
        isin = manual_isin if manual_isin else i.get('isin', ticker)
        name = i.get('longName', i.get('shortName', ticker))
        reg = "Global"
        if any(x in ticker for x in [".OL", ".PA", ".DE", ".CO", ".ST"]): reg = "EEA"
        if "US" in isin: reg = "US"
        prof = FundProfile(isin=isin, ticker=ticker, name=name, category=cat, region=reg)
        session.merge(prof)
        session.commit()
        if user_id:
            if not session.query(UserFundSelection).filter_by(user_id=user_id, isin=isin).first():
                session.add(UserFundSelection(user_id=user_id, isin=isin))
                session.commit()
        return isin, name
    except Exception as e:
        return None, str(e)


def update_price_history(isin):
    p = session.query(FundProfile).filter_by(isin=isin).first()
    if not p: return
    try:
        hist = yf.Ticker(p.ticker).history(period="5y")
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
        if date_override:
            hist = session.query(FundPriceHistory).filter(FundPriceHistory.isin == isin,
                                                          FundPriceHistory.date <= d).order_by(
                desc(FundPriceHistory.date)).first()
            p = hist.close_price if hist else get_latest_price(isin)
        else:
            update_price_history(isin)
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


def get_portfolio_context(user_id):
    df = get_portfolio_df(user_id)
    if df.empty: return []
    summary = df.groupby('Allocation', as_index=False).agg({
        'Current Value': 'sum',
        'ISIN': 'first'
    })
    context = []
    for _, row in summary.iterrows():
        isin = row['ISIN']
        category = "Unknown"
        region = "Global"
        if isin in ['BANK', 'DEBT']:
            category = "Cash/Debt"
            region = "Local"
        else:
            prof = session.query(FundProfile).filter_by(isin=isin).first()
            if prof:
                category = prof.category
                region = prof.region
        context.append({
            "name": row['Allocation'],
            "current_value_nok": round(row['Current Value'], 0),
            "category": category,
            "region": region
        })
    return context


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
            x=df_inv["Date"], y=df_inv["Price"], mode='markers+text', marker=dict(size=14, color='Gold', symbol='star'),
            text=[f"{a:,.0f}{currency_sym}" for a in df_inv["Amt"]], textposition="top center",
            textfont=dict(color='black')
        ))
    fig.update_layout(template="plotly_white")
    return fig


# ==========================================
# 5. MAIN UI
# ==========================================
st.set_page_config(page_title="FinStrat AI Pro", layout="wide")

if 'user_id' not in st.session_state: st.session_state.user_id = None
if st.session_state.user_id is None:
    params = st.query_params
    if "uid" in params:
        try:
            uid = int(params["uid"])
            user = get_user_by_id(uid)
            if user:
                st.session_state.user_id = user.id
                st.session_state.residences = user.tax_residences
                st.session_state.username = user.username
        except:
            pass

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
                    st.query_params["uid"] = str(user.id)
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
    # DASHBOARD
    st.sidebar.title(f"👤 {st.session_state.username}")
    st.sidebar.info(f"Tax: {st.session_state.residences}")
    st.sidebar.markdown("---")

    gemini_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
    ai_model = st.sidebar.text_input("🤖 AI Model", value="gemini-1.5-flash")

    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.query_params.clear()
        st.rerun()

    # UNIVERSE BUILDER
    st.sidebar.markdown("---")
    st.sidebar.header("Universe Builder")
    sb_tab1, sb_tab2 = st.sidebar.tabs(["Recommendations", "Manual ISIN"])

    with sb_tab1:
        user_res = st.session_state.residences
        filtered_suggestions = [f for f in SUGGESTED_FUNDS if
                                "Global" in f['countries'] or any(c in user_res for c in f['countries'])]
        for f in filtered_suggestions:
            prof = session.query(FundProfile).filter_by(ticker=f['ticker']).first()
            is_linked = False
            if prof and session.query(UserFundSelection).filter_by(user_id=st.session_state.user_id,
                                                                   isin=prof.isin).first():
                is_linked = True
            c1, c2 = st.sidebar.columns([3, 1])
            c1.text(f['name'])
            if not is_linked:
                if c2.button("Add", key=f['ticker']):
                    fetch_and_save_fund_profile(f['ticker'], f['cat'], st.session_state.user_id)
                    update_price_history(session.query(FundProfile).filter_by(ticker=f['ticker']).first().isin)
                    st.rerun()
            else:
                c2.caption("✅")
    with sb_tab2:
        with st.sidebar.form("manual_add"):
            m_tick = st.text_input("Yahoo Ticker")
            m_isin = st.text_input("ISIN Code")
            m_cat = st.selectbox("Category", ["Global", "Nordic", "Tech", "Energy"])
            if st.form_submit_button("Add Fund"):
                if m_tick and m_isin:
                    res_isin, res_name = fetch_and_save_fund_profile(m_tick, m_cat, st.session_state.user_id,
                                                                     manual_isin=m_isin)
                    if res_isin:
                        update_price_history(res_isin)
                        st.success(f"Added {res_name}")
                        st.rerun()

    # TABS
    tab1, tab2 = st.tabs(["🧠 AI Strategy Agent", "📊 Portfolio"])

    with tab1:
        st.subheader("AI Portfolio Architect")

        links = session.query(UserFundSelection).filter_by(user_id=st.session_state.user_id).all()
        my_funds = session.query(FundProfile).filter(FundProfile.isin.in_([l.isin for l in links])).all()

        if not my_funds:
            st.warning("Your universe is empty. Add funds from the sidebar.")
        else:
            with st.expander("📂 My Active Fund Universe"):
                for f in my_funds: st.text(f"• {f.name} (ISIN: {f.isin})")

            st.markdown("---")
            mode = st.radio("Mode", ["Live (Today)", "Backtest (Historical)"], horizontal=True)
            c_a, c_b = st.columns(2)
            amount = c_a.number_input("Cash to Deploy (NOK)", step=5000.0)
            risk = c_b.select_slider("Risk Profile", options=["Conservative", "Moderate", "Growth", "Aggressive"],
                                     value="Growth")

            with st.expander("💼 Financial Context", expanded=True):
                c1, c2 = st.columns(2)
                current_savings = c1.number_input("Savings (Approx)", value=50000.0)
                debt_rate = c2.number_input("Mortgage Rate (%)", value=5.5)

            sim_date = None
            if mode == "Backtest (Historical)":
                sim_date = st.date_input("Date", value=datetime.now() - timedelta(days=365))

            if 'agent_result' not in st.session_state: st.session_state.agent_result = None
            if 'agent_analysis' not in st.session_state: st.session_state.agent_analysis = None
            if 'selected_extras' not in st.session_state: st.session_state.selected_extras = []

            if amount > 0:
                if st.session_state.agent_result is None:
                    if st.button("⚡ Generate AI Strategy", type="primary"):
                        if not gemini_key:
                            st.warning("⚠️ Enter API Key.")
                        else:
                            agent = StrategyAgent(gemini_key, ai_model)
                            user_prof = {'tax_residence': st.session_state.residences, 'risk': risk,
                                         'savings': current_savings, 'debt_rate': debt_rate}
                            curr_port = get_portfolio_context(st.session_state.user_id)
                            with st.spinner(f"Consulting {ai_model}..."):
                                rec_map, reasoning = agent.get_allocation(user_prof, my_funds, amount, curr_port)
                                st.session_state.agent_result = {'allocs': rec_map, 'reasoning': reasoning}
                                st.rerun()

                if st.session_state.agent_result:
                    res = st.session_state.agent_result
                    st.info(f"**AI Strategy:** {res['reasoning']}")

                    if st.button("🔄 Reset Strategy"):
                        st.session_state.agent_result = None
                        st.session_state.agent_analysis = None
                        st.session_state.selected_extras = []
                        st.rerun()

                    st.markdown("#### 🛠 Customize & Execute")

                    # 1. SELECT EXTRA FUNDS (OUTSIDE FORM to trigger re-run)
                    rec_map = res['allocs']
                    existing_isins = list(rec_map.keys())
                    avail_other = [f for f in my_funds if f.isin not in existing_isins]

                    if avail_other:
                        st.caption("Select other funds from your universe to allocate cash to:")
                        st.session_state.selected_extras = st.multiselect(
                            "Add Funds:",
                            options=[f.isin for f in avail_other],
                            format_func=lambda x: next((f.name for f in avail_other if f.isin == x), x),
                            default=st.session_state.selected_extras
                        )

                    # 2. EXECUTION FORM
                    with st.form("agent_exec"):
                        final_allocs = {}

                        st.markdown("**Allocation Split**")
                        # AI Funds
                        for isin, val in rec_map.items():
                            fname = next((f.name for f in my_funds if f.isin == isin), isin)
                            final_allocs[isin] = st.number_input(f"{fname} (AI)", value=float(val), key=f"alloc_{isin}")

                        # Manual Funds
                        if st.session_state.selected_extras:
                            for extra_isin in st.session_state.selected_extras:
                                fname = next((f.name for f in avail_other if f.isin == extra_isin), extra_isin)
                                final_allocs[extra_isin] = st.number_input(f"{fname} (You)", value=0.0,
                                                                           key=f"manual_{extra_isin}")

                        # Cash/Debt
                        rem = amount - sum(final_allocs.values())
                        if abs(rem) > 1.0: st.caption(f"Unallocated / Overallocated: {rem:,.0f} NOK")

                        c1, c2 = st.columns(2)
                        final_allocs['BANK'] = c1.number_input("Savings", value=max(0.0, rem))
                        final_allocs['DEBT'] = c2.number_input("Debt Paydown", value=0.0)

                        c_sub1, c_sub2 = st.columns(2)
                        analyze_clk = c_sub1.form_submit_button("🔎 Analyze My Strategy")
                        exec_clk = c_sub2.form_submit_button("✅ Execute Investment")

                        if analyze_clk:
                            agent = StrategyAgent(gemini_key, ai_model)
                            user_prof = {'tax_residence': st.session_state.residences, 'risk': risk}
                            curr_port = get_portfolio_context(st.session_state.user_id)
                            with st.spinner("Analyzing custom allocation..."):
                                analysis = agent.analyze_user_edit(user_prof, final_allocs, my_funds, curr_port)
                                st.session_state.agent_analysis = analysis
                                st.rerun()

                        if exec_clk:
                            for isin, amt in final_allocs.items():
                                if amt > 0:
                                    n = "Savings" if isin == 'BANK' else ("Debt" if isin == 'DEBT' else "Fund")
                                    log_investment(st.session_state.user_id, amt, isin, n, 4, date_override=sim_date)
                            st.success("Executed!")
                            st.session_state.agent_result = None
                            st.session_state.agent_analysis = None
                            st.session_state.selected_extras = []
                            st.rerun()

                    if st.session_state.agent_analysis:
                        st.markdown("### 🤖 Strategy Critique")
                        st.warning(st.session_state.agent_analysis)

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
                def get_fund_name_safe(isin_code):
                    profile = session.query(FundProfile).filter_by(isin=isin_code).first()
                    return profile.name if profile else f"{isin_code} (Unknown)"


                sel = st.selectbox("History", fund_isins, format_func=get_fund_name_safe)
                fig = plot_history(session, st.session_state.user_id, sel, sym, rate)
                if fig: st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_g[['Allocation', 'Invested', 'Current Value', 'Return (%)']].style.format({
                'Invested': f"{sym} {{:,.0f}}", 'Current Value': f"{sym} {{:,.0f}}", 'Return (%)': "{:+.2f}%"
            }), use_container_width=True)
        else:
            st.info("No investments found.")