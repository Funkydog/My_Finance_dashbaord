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
You are a CFA-certified Chief Investment Officer (CIO) specializing in tax-efficient portfolio construction for retail investors.

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
- You can suggest allocations to "BANK" (savings) or "DEBT" (paydown) if it makes more economical sense.
- If the universe is poor, suggest the best available option and you can also suggest adding better funds available in that tax jurisdiction.
- Provide a diversified allocation that balances risk and tax efficiency.
- Your reasoning should reflect tax optimization and risk management with a focus on the user's profile.
- Consider the user as a newby investor with no prior holdings.
- Screen the geopolitical risk of funds based on their region and adjust allocations accordingly.

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
engine = create_engine('sqlite:///personal_finance_v10_complete.db', echo=False)


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
# 2. RESTORED: EXPANDED FUND LIST
# ==========================================
SUGGESTED_FUNDS = [
    # GLOBAL / US
    {"ticker": "SPY", "cat": "Global", "name": "SPDR S&P 500 ETF (US)", "countries": ["USA", "Global"]},
    {"ticker": "QQQ", "cat": "Tech", "name": "Invesco QQQ (Nasdaq-100)", "countries": ["USA", "Global"]},
    {"ticker": "VT", "cat": "Global", "name": "Vanguard Total World Stock", "countries": ["USA", "Global"]},
    # NORWAY
    {"ticker": "DNB.OL", "cat": "Norwegian", "name": "DNB Bank ASA", "countries": ["Norway"]},
    {"ticker": "EQNR.OL", "cat": "Energy", "name": "Equinor ASA", "countries": ["Norway"]},
    {"ticker": "MOWI.OL", "cat": "Seafood", "name": "Mowi ASA", "countries": ["Norway"]},
    {"ticker": "YAR.OL", "cat": "Materials", "name": "Yara International", "countries": ["Norway"]},
    {"ticker": "KOG.OL", "cat": "Defense", "name": "Kongsberg Gruppen", "countries": ["Norway"]},
    {"ticker": "ORK.OL", "cat": "Consumer", "name": "Orkla ASA", "countries": ["Norway"]},
    # FRANCE
    {"ticker": "CW8.PA", "cat": "Global", "name": "Amundi MSCI World (PEA)", "countries": ["France"]},
    {"ticker": "ESE.PA", "cat": "Global", "name": "BNP Paribas S&P 500 (PEA)", "countries": ["France"]},
    {"ticker": "TTE.PA", "cat": "Energy", "name": "TotalEnergies SE", "countries": ["France"]},
    {"ticker": "MC.PA", "cat": "Luxury", "name": "LVMH Moët Hennessy", "countries": ["France"]},
    {"ticker": "AI.PA", "cat": "Materials", "name": "Air Liquide", "countries": ["France"]},
    {"ticker": "OR.PA", "cat": "Beauty", "name": "L'Oréal", "countries": ["France"]},
    # GERMANY / EU
    {"ticker": "EUNL.DE", "cat": "Global", "name": "iShares Core MSCI World (UCITS)",
     "countries": ["Germany", "Norway", "Sweden", "Denmark"]},
    {"ticker": "SAP.DE", "cat": "Tech", "name": "SAP SE", "countries": ["Germany"]},
    {"ticker": "SIE.DE", "cat": "Industrial", "name": "Siemens AG", "countries": ["Germany"]},
    # SWEDEN
    {"ticker": "INVE-B.ST", "cat": "Finance", "name": "Investor AB", "countries": ["Sweden"]},
    {"ticker": "VOLV-B.ST", "cat": "Industrial", "name": "Volvo AB", "countries": ["Sweden"]},
    {"ticker": "ATCO-A.ST", "cat": "Industrial", "name": "Atlas Copco", "countries": ["Sweden"]},
    # DENMARK
    {"ticker": "NOVO-B.CO", "cat": "Health", "name": "Novo Nordisk", "countries": ["Denmark", "Global"]},
    {"ticker": "DSV.CO", "cat": "Logistics", "name": "DSV A/S", "countries": ["Denmark"]},
    # UK
    {"ticker": "HSBA.L", "cat": "Finance", "name": "HSBC Holdings", "countries": ["UK"]},
    {"ticker": "AZN.L", "cat": "Health", "name": "AstraZeneca", "countries": ["UK"]},
]


# ==========================================
# 3. STRATEGY AGENT (WITH FIX)
# ==========================================
class StrategyAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_allocation(self, user_profile, funds, amount):
        funds_context = []
        for f in funds:
            funds_context.append({
                "isin": f.isin, "name": f.name, "region": f.region,
                "category": f.category
            })

        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tax_residence=user_profile['tax_residence'],
            amount=amount,
            currency="NOK",
            risk_profile=user_profile.get('risk', 'Growth'),
            fund_list_json=json.dumps(funds_context, indent=2)
        )

        if not self.api_key:
            return {}, "⚠️ No API Key provided. Please enter your Google Gemini Key in the sidebar."

        try:
            genai.configure(api_key=self.api_key)
            # FIX: Updated model name to gemini-1.5-flash
            model = genai.GenerativeModel('gemini-2.5-flash')

            response = model.generate_content(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_text)

            return result.get('allocations', {}), result.get('reasoning', "AI provided no reasoning.")

        except Exception as e:
            return {}, f"AI Error: {str(e)}"


# ==========================================
# 4. CORE LOGIC & HELPERS
# ==========================================
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
# 5. MAIN UI
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
    st.sidebar.text("🔑Try copy this temp-key: AIzaSyCjW1wqq9tPrYGMn9ed9ZHWQ3p-mZQBS8g")
    if st.sidebar.button("Logout"): st.session_state.user_id = None; st.rerun()

    # RESTORED: PRICE UPDATES SECTION
    st.sidebar.markdown("---")
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

    # 1. ADD FUNDS (SMART DISCOVERY)
    st.sidebar.markdown("---")
    st.sidebar.header("Universe Builder")

    user_res = st.session_state.residences
    filtered_suggestions = [f for f in SUGGESTED_FUNDS if
                            "Global" in f['countries'] or any(c in user_res for c in f['countries'])]

    for f in filtered_suggestions:
        # Check if linked
        prof = session.query(FundProfile).filter_by(ticker=f['ticker']).first()
        is_linked = False
        if prof:
            if session.query(UserFundSelection).filter_by(user_id=st.session_state.user_id, isin=prof.isin).first():
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

                if not gemini_key:
                    st.warning("⚠️ Enter Gemini API Key in sidebar to get AI recommendations.")
                else:
                    with st.spinner("🤖 Consulting Gemini 1.5 Flash..."):
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
                                fname = next((f.name for f in my_funds if f.isin == isin), isin)
                                with cols[i % 2]:
                                    allocs[isin] = st.number_input(f"{fname}", value=float(val))

                            # Handle funds AI didn't pick
                            for f in my_funds:
                                if f.isin not in allocs:
                                    # Hidden input or just not allocated
                                    pass

                                    # Remaining
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