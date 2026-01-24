import streamlit as st
import pandas as pd
import numpy as np # Ensure numpy is imported
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
# 🧠 AI AGENT CONFIGURATION (UPDATED)
# ==========================================
SYSTEM_PROMPT_TEMPLATE = """
You are a CFA-certified Private Wealth Manager.

**Client Profile:**
- **Residence:** {tax_residence} | **Risk:** {risk_profile}
- **Monthly Savings Capacity:** {monthly_savings} {currency}

**Balance Sheet Context:**
{balance_sheet_json}

**Current Portfolio Exposure (REBALANCING LOGIC):**
{portfolio_exposure}
**Rebalancing Rule:**
Compare the 'Current Exposure' above to the client's Risk Profile ({risk_profile}).
- If they are **Overweight** in a category (e.g., >70% in Growth), DO NOT allocate new cash there.
- Prioritize **Underweighted** categories to bring them closer to target.
- Explicitly state: "Allocating to [Category] to rebalance your underweight exposure.

**Investment Goal:**
Allocate **{amount} {currency}** (New Cash) into the available universe.
**Timing Context:** Analysis Date: {analysis_date}.

**3. MACRO-REGIME FRAMEWORK (The Logic):**
You have access to REAL-TIME Macro Indicators. Use them to determine the regime.
**Live Macro Data:**
{macro_data}

**Regime Classification Rules:**
- **Inflationary Growth:** 10Y Yields Rising, Oil Rising, SP500 > SMA200. (Action: Overweight Value/Commodities).
- **Stagflation:** 10Y Yields Rising, GDP/Growth Slowing (or VIX High), Gold Rising. (Action: Overweight Cash/Defensives/Gold).
- **Disinflationary Growth (Goldilocks):** 10Y Yields Falling/Stable, VIX Low, SP500 > SMA200. (Action: Overweight Tech/Growth).
- **Deflationary Bust:** 10Y Yields Falling (Flight to Safety), VIX Spiking, SP500 < SMA200. (Action: Overweight Bonds/Quality).

**Strategic Rules:**
1. **Liquidity Check:** If 'Cash' assets are < 3 months of expenses (approx 60k), prioritize filling the 'Savings' bucket.
2. **Debt Management:** If Mortgage Rate ({avg_debt_rate}%) > 5%, allocate heavily to Debt Paydown.
3. **Diversification:** Consider the user's *Total* Net Worth. If they are heavy in Real Estate, prioritize liquid global equities.
4. **Tax Efficiency:** Norway (ASK), France (PEA), USA (No PFIC).

**Tactical Rules (Moving Average Logic):**
You have access to the **50-Day Simple Moving Average (SMA)** for each fund. Use **Mean Reversion Theory**:
- **✅ Good Entry (Buy):** Price is **BELOW** the SMA. The asset is "on sale" relative to its recent trend.
- **⚠️ Caution (Wait/Drip):** Price is **ABOVE** the SMA (especially > 5% above). The asset is "extended" or "expensive".

**Available Funds (Universe):**
{fund_list_json}

**Output Format:**
Respond with a strict JSON object:
{{
  "current_regime": "Name of the regime (e.g., Disinflationary Growth)",
  "allocations": {{ "ISIN_OR_TICKER": AMOUNT }},
  "reasoning": "A structured analysis: 1) Cite specific macro data (e.g. 'With 10Y Yields at X% and Oil falling...'). 2) Explain the Regime choice. 3) Fund Selection: Why these specific funds given the regime and their SMA status."
}}
"""

ANALYSIS_PROMPT_TEMPLATE = """
You are a Risk Manager.
**Context:**
- Profile: {risk_profile}, Residence: {tax_residence}
- **Total Balance Sheet:**
{balance_sheet_json}
- **Proposed Trade:**
{proposed_allocation_json}

**Task:**
Critique this trade. Does it improve the client's Net Worth diversification?
Does it address their high-interest debt (if any)?

**Output:**
3-sentence critique. Start with "✅ Approved" or "⚠️ Caution".
"""

# ==========================================
# 1. DATABASE SETUP
# ==========================================
Base = declarative_base()
engine = create_engine('sqlite:///personal_finance_v19_holistic.db', echo=False)


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    tax_residences = Column(String, default="Norway")
    monthly_savings_capacity = Column(Float, default=0.0)


class Asset(Base):
    __tablename__ = 'assets'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    name = Column(String)
    category = Column(String)
    value = Column(Float)
    interest_rate = Column(Float, default=0.0)


class Liability(Base):
    __tablename__ = 'liabilities'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    name = Column(String)
    principal = Column(Float)
    interest_rate = Column(Float)
    monthly_payment = Column(Float)


class FundProfile(Base):
    __tablename__ = 'fund_profiles'
    isin = Column(String, primary_key=True)
    ticker = Column(String)
    name = Column(String)
    description = Column(Text)
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
    risk_score = Column(Integer)


class FundPriceHistory(Base):
    __tablename__ = 'fund_price_history'
    id = Column(Integer, primary_key=True)
    isin = Column(String, index=True)
    date = Column(DateTime)
    close_price = Column(Float)


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
    {"ticker": "CW8.PA", "cat": "Global", "name": "Amundi MSCI World (PEA)", "countries": ["France"]},
    {"ticker": "EUNL.DE", "cat": "Global", "name": "iShares Core MSCI World", "countries": ["Germany", "Norway"]},
    {"ticker": "NOVO-B.CO", "cat": "Health", "name": "Novo Nordisk", "countries": ["Denmark", "Global"]},
]

# ==========================================
# 2b. REBALANCING CONFIGURATION
# ==========================================
# 1. Target Weights per Risk Profile
RISK_MODELS = {
    "Conservative": {"Defensive": 0.50, "Growth": 0.20, "Cash": 0.30},
    "Moderate":     {"Defensive": 0.35, "Growth": 0.45, "Cash": 0.20},
    "Growth":       {"Defensive": 0.15, "Growth": 0.75, "Cash": 0.10},
    "Aggressive":   {"Defensive": 0.05, "Growth": 0.90, "Cash": 0.05}
}

# 2. Map Fund Categories to Macro Buckets
# (Adjust these keys based on the categories you use in your app)
CATEGORY_MAP = {
    "Tech": "Growth", "Global": "Growth", "Seafood": "Growth", "USA": "Growth",
    "Health": "Defensive", "Energy": "Defensive", "Norwegian": "Defensive",
    "Bonds": "Defensive", "Bank": "Cash", "Cash": "Cash"
}

def get_bucket(cat_name):
    """Maps a specific fund category (e.g., 'Tech') to a Macro Bucket (e.g., 'Growth')."""
    return CATEGORY_MAP.get(cat_name, "Growth") # Default to Growth if unknown


# ==========================================
# 3. STRATEGY AGENT
# ==========================================
class StrategyAgent:
    def __init__(self, api_key, model_name="gemini-2.5-flash-lite"):
        self.api_key = api_key
        self.model_name = model_name

    def get_allocation(self, user_profile, funds, amount, balance_sheet, analysis_date=None):
        target_date = analysis_date if analysis_date else datetime.now()

        # --- NEW: FETCH MACRO DATA ---
        macro_text = "Data not available (Simulation Mode - Manual Regime Assumption Required)"
        # Only fetch live macro data if we are NOT in a deep historical backtest
        if (datetime.now() - target_date).days < 3:
            macro_text = get_macro_indicators()

        # --- NEW: CALCULATE EXPOSURE ---
        # We need the user ID to fetch the portfolio.
        # Assuming you pass 'user_id' in user_profile or we fetch it from session outside.
        # For this function, let's assume user_profile has 'user_id'
        cur_exp, tot_val = get_portfolio_exposure(user_profile.get('user_id'))

        # Format for AI
        exp_str = json.dumps(cur_exp, indent=2)

        funds_context = []
        for f in funds:
            # 1. Fetch Price AT THE TARGET DATE
            price_rec = session.query(FundPriceHistory).filter(
                FundPriceHistory.isin == f.isin,
                FundPriceHistory.date <= target_date
            ).order_by(desc(FundPriceHistory.date)).first()

            curr_p = price_rec.close_price if price_rec else 0.0

            # 2. Calculate SMA-50
            sma_hist = session.query(FundPriceHistory.close_price).filter(
                FundPriceHistory.isin == f.isin,
                FundPriceHistory.date <= target_date
            ).order_by(desc(FundPriceHistory.date)).limit(50).all()

            trend = "No Data"
            if len(sma_hist) >= 10:
                sma = sum([h[0] for h in sma_hist]) / len(sma_hist)
                if curr_p > 0:
                    diff = ((curr_p - sma) / sma) * 100
                    trend = f"{diff:+.1f}% vs SMA50 (on {target_date.strftime('%Y-%m-%d')})"

            funds_context.append({
                "isin": f.isin, "name": f.name, "cat": f.category,
                "price": curr_p, "trend": trend
            })

        bs_json = json.dumps(balance_sheet, indent=2)
        debts = balance_sheet.get('liabilities', [])
        avg_rate = 0.0
        if debts:
            total_debt = sum(d['principal'] for d in debts)
            if total_debt > 0: avg_rate = sum(d['principal'] * d['interest_rate'] for d in debts) / total_debt

        # --- UPDATED PROMPT INJECTION ---
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tax_residence=user_profile['tax_residence'],
            amount=amount,
            currency="NOK",
            risk_profile=user_profile.get('risk', 'Growth'),
            monthly_savings=user_profile.get('monthly_savings', 0),
            balance_sheet_json=bs_json,
            portfolio_exposure=exp_str,  # <--- INJECT EXPOSURE HERE
            avg_debt_rate=round(avg_rate, 2),
            fund_list_json=json.dumps(funds_context, indent=2),
            analysis_date=target_date.strftime('%Y-%m-%d'),
            macro_data=macro_text
        )
        return self._call_gemini_json(prompt)

    def analyze_user_edit(self, user_profile, proposed_allocations, funds, balance_sheet):
        readable_alloc = {}
        for isin, amt in proposed_allocations.items():
            if amt > 0:
                fname = next((f.name for f in funds if f.isin == isin), isin)
                readable_alloc[fname] = amt

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            risk_profile=user_profile.get('risk', 'Growth'),
            tax_residence=user_profile['tax_residence'],
            proposed_allocation_json=json.dumps(readable_alloc, indent=2),
            balance_sheet_json=json.dumps(balance_sheet, indent=2)
        )
        return self._call_gemini_text(prompt)

    def _call_gemini_json(self, prompt):
        if not self.api_key: return {}, {}, "⚠️ No API Key."
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            clean = response.text.replace("```json", "").replace("```", "").strip()
            res = json.loads(clean)
            return res.get('allocations', {}), res.get('current_regime', "Unknown"), res.get('reasoning', "")
        except Exception as e:
            return {}, "Error", f"AI Error: {str(e)}"

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
    if session.query(User).filter_by(username=u).first(): return False, "Exists."
    session.add(User(username=u, password_hash=make_hash(p), tax_residences=",".join(c)))
    session.commit()
    fetch_and_save_fund_profile("0P00018V9L.IR", "Global")
    return True, "Created."


# --- IMPROVED: INTEGRITY & PRICE CHECK ---
# --- STRICT TICKER-FIRST LOGIC ---
def fetch_and_save_fund_profile(ticker, cat="Custom", user_id=None, manual_isin=None):
    try:
        t = yf.Ticker(ticker)

        # 1. PRICE HEALTH CHECK
        hist = t.history(period="1mo")
        if hist.empty:
            return None, f"❌ Yahoo has no data for ticker '{ticker}'. Cannot add fund."

        i = t.info

        # 2. STRICT ISIN COLLECTION
        # Primary: Get ISIN from Yahoo (The Source of Truth)
        yahoo_isin = i.get('isin')
        yahoo_name = i.get('longName', i.get('shortName', ticker))

        final_isin = None

        if yahoo_isin and yahoo_isin.strip():
            # SUCCESS: Yahoo provided an ISIN. We use it.
            final_isin = yahoo_isin.strip()
            # If user provided a manual ISIN that contradicts Yahoo, we ignore the manual one
            # to ensure the Price (from Yahoo) matches the Asset (ISIN) in our DB.
        else:
            # FALLBACK: Yahoo has price data but NO ISIN (rare/obscure funds)
            # Only in this specific case do we trust the user's manual input.
            if manual_isin and manual_isin.strip():
                final_isin = manual_isin.strip()
            else:
                # Last Resort: Use Ticker as pseudo-ISIN
                final_isin = ticker.strip()

        # 3. SAVE TO DB
        reg = "Global"
        if any(x in ticker for x in [".OL", ".PA", ".DE", ".CO", ".ST"]): reg = "EEA"
        if "US" in final_isin: reg = "US"

        prof = FundProfile(isin=final_isin, ticker=ticker, name=yahoo_name, category=cat, region=reg)
        session.merge(prof)
        session.commit()

        if user_id:
            if not session.query(UserFundSelection).filter_by(user_id=user_id, isin=final_isin).first():
                session.add(UserFundSelection(user_id=user_id, isin=final_isin))
                session.commit()

        return final_isin, yahoo_name
    except Exception as e:
        return None, f"System Error: {str(e)}"


# --- UPDATED: Ensure 3 Months of History for Proper SMA ---
def ensure_historical_data_for_sma(isin, target_date):
    """Fetches enough historical data (3 months back) to calculate SMA-50 for the target date."""

    # 1. Check if we already have the specific target date price
    existing = session.query(FundPriceHistory).filter_by(isin=isin, date=target_date).first()

    # 2. Check if we have the 50 days prior (approx)
    start_window = target_date - timedelta(days=90)  # 3 months buffer
    prior_count = session.query(FundPriceHistory).filter(
        FundPriceHistory.isin == isin,
        FundPriceHistory.date >= start_window,
        FundPriceHistory.date <= target_date
    ).count()

    # If we have the price AND enough context, we are good
    if existing and prior_count >= 40:
        return existing.close_price

    # Otherwise, fetch from Yahoo
    prof = session.query(FundProfile).filter_by(isin=isin).first()
    if not prof: return 1.0

    try:
        t = yf.Ticker(prof.ticker)
        # Fetch 3 months before to 1 day after
        end_d = target_date + timedelta(days=1)
        hist = t.history(start=start_window, end=end_d)

        if not hist.empty:
            for d, row in hist.iterrows():
                if not session.query(FundPriceHistory).filter_by(isin=isin, date=d).first():
                    session.add(FundPriceHistory(isin=isin, date=d, close_price=row['Close']))
            session.commit()

            # Return the specific price for target date (or closest before it)
            closest_idx = hist.index.get_indexer([target_date], method='pad')[0]
            if closest_idx != -1:
                return hist.iloc[closest_idx]['Close']

    except:
        pass
    return 1.0


# --- NEW: Specific Historical Fetcher ---
# --- SMART UPDATE FUNCTION (OPTIMIZED) ---
def update_price_history(isin):
    p = session.query(FundProfile).filter_by(isin=isin).first()
    if not p: return

    # 1. Check the LATEST date we already have in the database
    last_rec = session.query(FundPriceHistory).filter_by(isin=isin).order_by(desc(FundPriceHistory.date)).first()

    try:
        t = yf.Ticker(p.ticker)
        hist = pd.DataFrame()

        # 2. Determine Fetch Strategy
        if not last_rec:
            # Case A: No data at all -> Fetch full history (5 years)
            # st.write("Fetching 5 years")
            hist = t.history(period="5y")
        else:
            # Case B: Data exists -> Check if it's stale
            # st.write("Fetching since {} until today".format(last_rec.date))
            last_date_in_db = last_rec.date
            today = datetime.now().date()

            # If last data is older than today, fetch only the missing delta
            if last_date_in_db.date() < today:
                # Start from the day AFTER the last record
                start_fetch = last_date_in_db + timedelta(days=1)
                hist = t.history(start=start_fetch)
            else:
                # Case C: Data is fresh (today) -> Do nothing (Efficient!)
                # st.write("Data up to date")
                return

                # 3. Save new data if any
        if not hist.empty:
            for date, row in hist.iterrows():
                if not session.query(FundPriceHistory).filter_by(isin=isin, date=date).first():
                    session.add(FundPriceHistory(isin=isin, date=date, close_price=row['Close']))
            session.commit()
    except:
        pass

#TODO: make a warning message if the fund is not updated as of today (due to lack of data from yahoo)

# --- NEW: AUTO-UPDATE ON LOGIN ---
def refresh_user_prices(user_id):
    """Updates prices for all funds in the user's universe."""
    links = session.query(UserFundSelection).filter_by(user_id=user_id).all()
    for link in links:
        update_price_history(link.isin)

def get_latest_price(isin):
    rec = session.query(FundPriceHistory).filter_by(isin=isin).order_by(desc(FundPriceHistory.date)).first()
    return rec.close_price if rec else 1.0


# --- UPDATED: Smart Date Alignment ---
def log_investment(user_id, amount, isin, alloc_type, risk, date_override=None, manual_price_override=None):
    transaction_date = datetime.now()
    final_price = 1.0

    if isin not in ['BANK', 'DEBT']:
        # Case A: Backtest (User set a date)
        if date_override:
            transaction_date = datetime.combine(date_override, datetime.min.time())
            if manual_price_override and manual_price_override > 0:
                final_price = manual_price_override
            else:
                final_price = ensure_historical_data_for_sma(isin, transaction_date)
                if final_price == 1.0: final_price = get_latest_price(isin)

        # Case B: Live (Snap to latest market data)
        else:
            # 1. Ensure latest data
            update_price_history(isin)
            # 2. Get latest record from DB
            latest_rec = session.query(FundPriceHistory).filter_by(isin=isin).order_by(
                desc(FundPriceHistory.date)).first()
            if latest_rec:
                transaction_date = latest_rec.date  # SNAP TO MARKET DATE
                final_price = latest_rec.close_price
            else:
                final_price = 1.0

            # Allow manual override even in Live mode
            if manual_price_override and manual_price_override > 0:
                final_price = manual_price_override

    session.add(FinancialRecord(user_id=user_id, date=transaction_date, year=transaction_date.year, amount=amount,
                                allocation_type=alloc_type, isin=isin, entry_price=final_price,
                                units_owned=amount / final_price if final_price else 0, risk_score=risk))
    session.commit()


def delete_transactions(record_ids):
    if not record_ids: return
    session.query(FinancialRecord).filter(FinancialRecord.id.in_(record_ids)).delete(synchronize_session=False)
    session.commit()


def get_balance_sheet(user_id):
    assets = session.query(Asset).filter_by(user_id=user_id).all()
    asset_list = [{"name": a.name, "category": a.category, "value": a.value, "yield": a.interest_rate} for a in assets]
    liabilities = session.query(Liability).filter_by(user_id=user_id).all()
    liab_list = [{"name": l.name, "principal": l.principal, "interest_rate": l.interest_rate,
                  "monthly_payment": l.monthly_payment} for l in liabilities]

    recs = session.query(FinancialRecord).filter_by(user_id=user_id).all()
    market_val = 0
    portfolio_breakdown = []
    for r in recs:
        p = 1.0
        if r.isin not in ['BANK', 'DEBT']:
            p = get_latest_price(r.isin)
        curr = r.units_owned * p
        market_val += curr
        portfolio_breakdown.append({"isin": r.isin, "value": curr})

    return {
        "assets": asset_list,
        "liabilities": liab_list,
        "market_portfolio_total": market_val,
        "market_holdings": portfolio_breakdown
    }


def get_portfolio_df(user_id, risk_period="1y"):
    recs = session.query(FinancialRecord).filter_by(user_id=user_id).order_by(desc(FinancialRecord.date)).all()
    data = []

    for r in recs:
        p = 1.0
        name = r.allocation_type
        vol = 0.0
        dd = 0.0

        # Only calculate risk for actual Funds (skip Bank/Debt)
        if r.isin not in ['BANK', 'DEBT']:
            p = get_latest_price(r.isin)
            prof = session.query(FundProfile).filter_by(isin=r.isin).first()
            if prof: name = prof.name

            # --- CALL RISK CALCULATOR ---
            # Pass the selected period to the calculator
            vol, dd = calculate_risk_metrics(r.isin, period=risk_period)

        curr_val = r.units_owned * p
        profit = curr_val - r.amount

        data.append({
            "ID": r.id,
            "Delete?": False,
            "Allocation": name,
            "ISIN": r.isin,
            "Date": r.date.date(),
            "Invested": r.amount,
            "Current Value": curr_val,
            "Profit": profit,
            "Volatility": vol / 100,  # Store as float for formatting (e.g. 0.15 for 15%)
            "Max Drawdown": dd / 100  # Store as float (e.g. -0.20 for -20%)
        })

    return pd.DataFrame(data)


def get_exchange_rates():
    try:
        t = yf.Tickers('EURNOK=X USDNOK=X')
        return {'EUR': t.tickers['EURNOK=X'].fast_info['last_price'],
                'USD': t.tickers['USDNOK=X'].fast_info['last_price']}
    except:
        return {'EUR': 11.5, 'USD': 10.5}

def get_safe_fund_name(isin_code):
    """Global helper prevents widget reset."""
    p = session.query(FundProfile).filter_by(isin=isin_code).first()
    return p.name if p else f"{isin_code} (Unknown)"


# --- NEW: MACRO DATA FETCHER ---
def get_macro_indicators():
    """Fetches key economic indicators for AI context."""
    tickers = {
        "US 10Y Yield": "^TNX",
        "Crude Oil": "CL=F",
        "Gold": "GC=F",
        "VIX (Volatility)": "^VIX",
        "S&P 500": "^GSPC"
    }

    data_summary = []

    try:
        # Fetch current data for all tickers
        for name, ticker in tickers.items():
            t = yf.Ticker(ticker)

            # fast_info is often faster/more reliable for "last price" than history
            try:
                price = t.fast_info['last_price']
            except:
                hist = t.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                else:
                    price = 0.0

            # Specific logic for SP500 trend (200 Day SMA)
            trend_txt = ""
            if name == "S&P 500":
                try:
                    hist_long = t.history(period="1y")
                    if len(hist_long) > 200:
                        sma200 = hist_long['Close'].rolling(window=200).mean().iloc[-1]
                        if price > sma200:
                            trend_txt = f"(Bullish: Above 200SMA {sma200:.0f})"
                        else:
                            trend_txt = f"(Bearish: Below 200SMA {sma200:.0f})"
                except:
                    pass

            # Formatting
            if name == "US 10Y Yield":
                val_str = f"{price:.2f}%"
            else:
                val_str = f"{price:,.2f}"

            data_summary.append(f"- {name}: {val_str} {trend_txt}")

        return "\n".join(data_summary)

    except Exception as e:
        return f"Error fetching macro data: {str(e)}"


def calculate_risk_metrics(isin, period="1y"):
    """
    Calculates Max Drawdown and Annualized Volatility over a dynamic period.
    Options: '1y', '6mo', '1mo', '1wk'
    """
    end_date = datetime.now()

    # Determine lookback window
    if period == "1wk":
        start_date = end_date - timedelta(days=7)
    elif period == "1mo":
        start_date = end_date - timedelta(days=30)
    elif period == "6mo":
        start_date = end_date - timedelta(days=180)
    else:  # Default 1y
        start_date = end_date - timedelta(days=365)

    # Fetch Data
    hist = session.query(FundPriceHistory.close_price).filter(
        FundPriceHistory.isin == isin,
        FundPriceHistory.date >= start_date
    ).order_by(FundPriceHistory.date).all()

    prices = [h[0] for h in hist]

    # Data Integrity Check
    min_points = 2  # Need at least 2 points to calc return
    if len(prices) < min_points:
        return 0.0, 0.0

    price_series = pd.Series(prices)

    # 1. Volatility (Annualized)
    # We always annualize volatility (sqrt(252)) to make it comparable across timeframes
    returns = price_series.pct_change().dropna()
    if returns.empty: return 0.0, 0.0
    volatility = returns.std() * np.sqrt(252) * 100

    # 2. Max Drawdown (Specific to the selected period)
    rolling_max = price_series.cummax()
    drawdown = (price_series - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    return round(volatility, 1), round(max_dd, 1)


def get_portfolio_exposure(user_id):
    """Calculates the current % allocation of the user's total net worth."""
    # 1. Get Balance Sheet
    bs = get_balance_sheet(user_id)
    total_val = bs['market_portfolio_total'] + sum(a['value'] for a in bs['assets'])

    if total_val == 0: return {}, 0

    # 2. Group Market Holdings
    current_counts = {"Growth": 0.0, "Defensive": 0.0, "Cash": 0.0}

    # A. Funds in Portfolio
    recs = session.query(FinancialRecord).filter_by(user_id=user_id).all()
    for r in recs:
        if r.isin in ['BANK', 'DEBT']: continue  # Handled in assets usually, or skip

        p = get_latest_price(r.isin)
        val = r.units_owned * p

        prof = session.query(FundProfile).filter_by(isin=r.isin).first()
        cat = prof.category if prof else "Global"
        bucket = get_bucket(cat)
        current_counts[bucket] += val

    # B. Cash/Assets (from Asset table)
    for a in bs['assets']:
        bucket = get_bucket(a['category'])  # e.g. "Real Estate" -> Growth or "Cash" -> Cash
        if bucket in current_counts:
            current_counts[bucket] += a['value']
        else:
            # Fallback for unmapped assets
            current_counts["Cash"] += a['value']

    # 3. Calculate %
    exposure_pct = {k: round(v / total_val, 2) for k, v in current_counts.items()}
    return exposure_pct, total_val


def get_portfolio_evolution(user_id, savings_rate_annual):
    """
    Reconstructs the daily value of the portfolio vs. a savings account.
    Fixes 'Zero Price' bug by falling back to entry price if history is missing.
    """
    # 1. Fetch Transactions
    recs = session.query(FinancialRecord).filter_by(user_id=user_id).order_by(FinancialRecord.date).all()
    if not recs: return pd.DataFrame()

    # 2. Setup Timeline
    start_date = pd.Timestamp(recs[0].date.date())
    end_date = pd.Timestamp(datetime.now().date())
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # 3. Initialize Dataframes
    unique_isins = list(set([r.isin for r in recs if r.isin not in ['BANK', 'DEBT']]))
    df_units = pd.DataFrame(0.0, index=date_range, columns=unique_isins)
    df_invested = pd.Series(0.0, index=date_range)

    # Track Entry Prices for Fallback
    fallback_prices = {}

    # 4. Process Transactions
    for r in recs:
        d = pd.Timestamp(r.date.date())
        if d < start_date: d = start_date

        # Capture the entry price for this ISIN in case we miss history later
        if r.isin not in fallback_prices and r.entry_price > 0:
            fallback_prices[r.isin] = r.entry_price

        # Cumulative Invested
        if d in df_invested.index:
            df_invested.loc[d:] += r.amount

        # Cumulative Units
        if r.isin in unique_isins:
            if d in df_units.index:
                units = r.units_owned if r.units_owned else 0
                df_units.loc[d:, r.isin] += units

    # 5. Fetch Prices & Calculate Portfolio Value
    df_prices = pd.DataFrame(index=date_range, columns=unique_isins)

    for isin in unique_isins:
        hist = session.query(FundPriceHistory).filter(
            FundPriceHistory.isin == isin,
            FundPriceHistory.date >= start_date
        ).all()

        # If we found history, map it
        if hist:
            temp_prices = {pd.Timestamp(h.date.date()): h.close_price for h in hist}
            df_prices[isin] = df_prices.index.map(temp_prices)
        else:
            # SAFETY NET: No history found? Use the fallback entry price
            # This prevents the line from dropping to 0
            safe_price = fallback_prices.get(isin, 1.0)
            df_prices[isin] = safe_price

    # Clean the Data:
    # 1. forward fill (cover weekends)
    # 2. back fill (cover days before history started but after purchase)
    # 3. fillna with fallback (absolute worst case)
    df_prices = df_prices.ffill().bfill()

    # Final check for any remaining NaNs (if no history AND no fallback)
    df_prices = df_prices.fillna(1.0)

    # Calculate Value
    df_portfolio_val = (df_units * df_prices).sum(axis=1)

    # 6. Calculate Savings Line
    daily_rate = (1 + (savings_rate_annual / 100.0)) ** (1 / 365.0) - 1
    savings_curve = []
    curr_bal = 0.0

    daily_flows = pd.Series(0.0, index=date_range)
    for r in recs:
        d = pd.Timestamp(r.date.date())
        if d >= start_date and d <= end_date:
            daily_flows.loc[d] += r.amount

    for date in date_range:
        flow = daily_flows.at[date]
        curr_bal = (curr_bal * (1 + daily_rate)) + flow
        savings_curve.append(curr_bal)

    return pd.DataFrame({
        "Invested": df_invested,
        "Savings": savings_curve,
        "Portfolio": df_portfolio_val
    }, index=date_range)

# --- UPDATED PLOT WITH ENVELOPES ---
def plot_history(session, user_id, isin, currency_sym, rate):
    # 1. Fetch History
    hist = session.query(FundPriceHistory).filter_by(isin=isin).order_by(FundPriceHistory.date).all()
    if not hist: return None
    prof = session.query(FundProfile).filter_by(isin=isin).first()
    name = prof.name if prof else isin

    # Create DataFrame
    df_h = pd.DataFrame([{"Date": h.date, "Price": h.close_price / rate} for h in hist])
    if df_h.empty: return None

    # --- CALC INDICATORS ---
    # Moving Average (Simple, Period 50)
    df_h['SMA_50'] = df_h['Price'].rolling(window=50).mean()
    # Envelope (Percent, Shift 5%)
    df_h['Upper_Env'] = df_h['SMA_50'] * 1.05
    df_h['Lower_Env'] = df_h['SMA_50'] * 0.95

    # 2. Fetch User Investments
    investments = session.query(FinancialRecord).filter_by(isin=isin, user_id=user_id).all()
    df_inv = pd.DataFrame(
        [{"Date": i.date, "Price": i.entry_price / rate, "Amt": i.amount / rate} for i in investments])

    # 3. Create Plot
    fig = px.line(df_h, x="Date", y="Price", title=f"{name} ({currency_sym})")

    # Add SMA Trace
    fig.add_trace(go.Scatter(x=df_h["Date"], y=df_h["SMA_50"], name="SMA (50)", line=dict(color='orange', width=1.5)))

    # Add Upper/Lower Envelope Traces
    # Upper
    fig.add_trace(go.Scatter(x=df_h["Date"], y=df_h["Upper_Env"], name="Upper Env (+5%)",
                             line=dict(color='gray', width=1, dash='dash')))
    # Lower (with Fill to Upper for visual envelope)
    fig.add_trace(go.Scatter(
        x=df_h["Date"], y=df_h["Lower_Env"], name="Lower Env (-5%)",
        line=dict(color='gray', width=1, dash='dash'),
        fill='tonexty',  # Fills to the previous trace (Upper Env)
        fillcolor='rgba(200, 200, 200, 0.1)'  # Light gray transparent fill
    ))

    # Add Investment Markers
    if not df_inv.empty:
        fig.add_trace(go.Scatter(
            x=df_inv["Date"],
            y=df_inv["Price"],
            mode='markers+text',
            marker=dict(size=14, color='Gold', symbol='star', line=dict(width=1, color='black')),
            text=[f"{a:,.0f}{currency_sym}" for a in df_inv["Amt"]],
            textposition="top center",
            textfont=dict(color='white'),
            name="Your Buy"
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
    if 'prices_updated' not in st.session_state:
        with st.spinner("🔄 Checking for new price data..."):
            refresh_user_prices(st.session_state.user_id)
        st.session_state.prices_updated = True

    st.sidebar.title(f"👤 {st.session_state.username}")
    try:
        gemini_key = st.secrets["api_keys"]["gemini_agent"]
    except:
        gemini_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")

    ai_model = st.sidebar.text_input("🤖 AI Model", value="gemini-2.5-flash-lite")
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.query_params.clear()
        del st.session_state['prices_updated']
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("Universe Builder")
    sb_tab1, sb_tab2 = st.sidebar.tabs(["Recs", "Manual"])
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
                    res_isin, res_msg = fetch_and_save_fund_profile(f['ticker'], f['cat'], st.session_state.user_id)
                    if res_isin:
                        update_price_history(res_isin)
                        st.rerun()
                    else:
                        st.error(res_msg)
            else:
                c2.caption("✅")
    with sb_tab2:
        with st.sidebar.form("manual_add"):
            st.info("Yahoo Ticker is the Source of Truth.")
            m_tick = st.text_input("Yahoo Ticker")
            m_isin = st.text_input("ISIN (Optional)")
            m_cat = st.selectbox("Category", ["Global", "Nordic", "Tech"])
            if st.form_submit_button("Add"):
                res_isin, res_msg = fetch_and_save_fund_profile(m_tick, m_cat, st.session_state.user_id,
                                                                manual_isin=m_isin)
                if res_isin:
                    update_price_history(res_isin)
                    st.success(f"Added {res_msg}")
                    st.rerun()
                else:
                    st.error(res_msg)

    # --- NAVIGATION ---
    TAB_NAMES = ["👤 Profile", "🧠 AI Strategy", "📊 Portfolio"]

    default_index = 0
    if "tab" in st.query_params:
        try:
            url_tab_name = st.query_params["tab"]
            if url_tab_name in TAB_NAMES:
                default_index = TAB_NAMES.index(url_tab_name)
        except:
            pass

    # Custom Tabs Style
    st.markdown("""
        <style>
            div.row-widget.stRadio > div{flex-direction:row;}
            div.row-widget.stRadio > div > label{
                background-color: #f0f2f6; padding: 10px 20px; border-radius: 5px 5px 0 0; 
                margin-right: 2px; border: 1px solid #ddd; border-bottom: none; cursor: pointer;
            }
            div.row-widget.stRadio > div > label[data-baseweb="radio"] {
                background-color: white; border-bottom: 2px solid #ff4b4b;
            }
        </style>
    """, unsafe_allow_html=True)

    selected_tab = st.radio("Main Navigation", TAB_NAMES, index=default_index, horizontal=True,
                            label_visibility="collapsed", key="main_nav_radio")

    # Sync URL if changed by click
    if selected_tab != st.query_params.get("tab", TAB_NAMES[0]):
        st.query_params["tab"] = selected_tab
        st.rerun()

    st.markdown("---")

    if selected_tab == "👤 Profile":
        st.header("Financial Identity")
        col_a, col_b = st.columns(2)
        with col_a:
            with st.form("profile_update"):
                st.subheader("1. Savings Capacity")
                curr_user = session.query(User).filter_by(id=st.session_state.user_id).first()
                new_savings = st.number_input("Monthly Savings (NOK)", value=float(curr_user.monthly_savings_capacity))
                if st.form_submit_button("Update"):
                    curr_user.monthly_savings_capacity = new_savings
                    session.commit()
                    st.success("Updated")
            with st.form("add_asset"):
                st.subheader("2. Add Asset")
                a_name = st.text_input("Name")
                a_cat = st.selectbox("Type", ["Cash", "Real Estate", "Crypto", "Other Fund"])
                a_val = st.number_input("Value", step=1000.0)
                a_int = st.number_input("Yield (%)", step=0.1)
                if st.form_submit_button("Add Asset"):
                    session.add(Asset(user_id=st.session_state.user_id, name=a_name, category=a_cat, value=a_val,
                                      interest_rate=a_int))
                    session.commit()
                    st.rerun()
        with col_b:
            with st.form("add_liability"):
                st.subheader("3. Add Liability")
                l_name = st.text_input("Name")
                l_princ = st.number_input("Principal", step=10000.0)
                l_rate = st.number_input("Rate (%)", step=0.1)
                l_pay = st.number_input("Payment", step=100.0)
                if st.form_submit_button("Add Liability"):
                    session.add(Liability(user_id=st.session_state.user_id, name=l_name, principal=l_princ,
                                          interest_rate=l_rate, monthly_payment=l_pay))
                    session.commit()
                    st.rerun()
        st.markdown("---")
        st.subheader("Current Balance Sheet")
        assets = session.query(Asset).filter_by(user_id=st.session_state.user_id).all()
        liabs = session.query(Liability).filter_by(user_id=st.session_state.user_id).all()
        c1, c2 = st.columns(2)
        with c1:
            if assets:
                st.dataframe(pd.DataFrame([{"Name": a.name, "Value": a.value} for a in assets]))
                if st.button("Clear Assets"):
                    session.query(Asset).filter_by(user_id=st.session_state.user_id).delete();
                    session.commit();
                    st.rerun()
        with c2:
            if liabs:
                st.dataframe(pd.DataFrame([{"Name": l.name, "Principal": l.principal} for l in liabs]))
                if st.button("Clear Liabilities"):
                    session.query(Liability).filter_by(user_id=st.session_state.user_id).delete();
                    session.commit();
                    st.rerun()

    elif selected_tab == "🧠 AI Strategy":
        st.subheader("AI Portfolio Architect")
        links = session.query(UserFundSelection).filter_by(user_id=st.session_state.user_id).all()
        my_funds = session.query(FundProfile).filter(FundProfile.isin.in_([l.isin for l in links])).all()

        mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)
        c_a, c_b = st.columns(2)
        amount = c_a.number_input("New Cash (NOK)", step=5000.0)
        risk = c_b.select_slider("Risk", options=["Conservative", "Moderate", "Growth", "Aggressive"], value="Growth")

        sim_date = None
        if mode == "Backtest":
            sim_date = st.date_input("Date", value=datetime.now() - timedelta(days=365))

        if 'agent_result' not in st.session_state: st.session_state.agent_result = None
        if 'selected_extras' not in st.session_state: st.session_state.selected_extras = []

        if amount > 0:
            if st.session_state.agent_result is None:
                if st.button("⚡ Generate AI Strategy", type="primary"):
                    if not gemini_key:
                        st.warning("⚠️ Enter API Key.")
                    else:
                        agent = StrategyAgent(gemini_key, ai_model)
                        user_prof = {'user_id': st.session_state.user_id,
                                     'tax_residence': st.session_state.residences,
                                     'risk': risk,
                                     'monthly_savings': session.query(User).get(
                                         st.session_state.user_id).monthly_savings_capacity}
                        bs = get_balance_sheet(st.session_state.user_id)
                        analysis_d = datetime.combine(sim_date, datetime.min.time()) if sim_date else datetime.now()
                        with st.spinner("Analyzing Macro Regime..."):
                            rec_map, regime, reasoning = agent.get_allocation(user_prof, my_funds, amount, bs,
                                                                              analysis_date=analysis_d)
                            st.session_state.agent_result = {'allocs': rec_map, 'regime': regime,
                                                             'reasoning': reasoning}
                            st.rerun()

            if st.session_state.agent_result:
                res = st.session_state.agent_result
                st.success(f"**Current Regime:** {res['regime']}")
                st.info(f"**Analysis:** {res['reasoning']}")
                if st.button("🔄 Reset"): st.session_state.agent_result = None; st.rerun()

                rec_map = res['allocs']
                avail_other = [f for f in my_funds if f.isin not in rec_map.keys()]
                if avail_other:
                    st.session_state.selected_extras = st.multiselect("Add Funds:", [f.isin for f in avail_other],
                                                                      format_func=lambda x: next(
                                                                          (f.name for f in avail_other if f.isin == x),
                                                                          x), default=st.session_state.selected_extras)

                with st.form("agent_exec"):
                    final_allocs = {}
                    final_prices = {}
                    st.markdown("**Allocation**")
                    for isin, val in rec_map.items():
                        fname = next((f.name for f in my_funds if f.isin == isin), isin)
                        c1, c2 = st.columns([2, 1])
                        final_allocs[isin] = c1.number_input(f"{fname} (AI)", value=float(val), key=f"a_{isin}")
                        display_p = 1.0
                        d_context = datetime.combine(sim_date, datetime.min.time()) if sim_date else datetime.now()
                        if sim_date:
                            display_p = ensure_historical_data_for_sma(isin, d_context)
                        else:
                            display_p = get_latest_price(isin)
                        final_prices[isin] = c2.number_input(f"Price {isin}", value=float(display_p), key=f"p_{isin}")

                    for extra_isin in st.session_state.selected_extras:
                        fname = next((f.name for f in avail_other if f.isin == extra_isin), extra_isin)
                        c1, c2 = st.columns([2, 1])
                        final_allocs[extra_isin] = c1.number_input(f"{fname} (You)", value=0.0, key=f"ma_{extra_isin}")
                        display_p = 1.0
                        d_context = datetime.combine(sim_date, datetime.min.time()) if sim_date else datetime.now()
                        if sim_date:
                            display_p = ensure_historical_data_for_sma(extra_isin, d_context)
                        else:
                            display_p = get_latest_price(extra_isin)
                        final_prices[extra_isin] = c2.number_input(f"Price {extra_isin}", value=float(display_p),
                                                                   key=f"mp_{extra_isin}")

                    rem = amount - sum(final_allocs.values())
                    st.caption(f"Unallocated: {rem:,.0f}")
                    c1, c2 = st.columns(2)
                    final_allocs['BANK'] = c1.number_input("Savings", value=max(0.0, rem))
                    final_allocs['DEBT'] = c2.number_input("Debt Paydown", value=0.0)

                    c_sub1, c_sub2 = st.columns(2)
                    if c_sub1.form_submit_button("🔎 Analyze"):
                        agent = StrategyAgent(gemini_key, ai_model)
                        bs = get_balance_sheet(st.session_state.user_id)
                        analysis = agent.analyze_user_edit({'risk': risk, 'tax_residence': st.session_state.residences},
                                                           final_allocs, my_funds, bs)
                        st.warning(analysis)

                    if c_sub2.form_submit_button("✅ Execute"):
                        for isin, amt in final_allocs.items():
                            if amt > 0:
                                n = "Savings" if isin == 'BANK' else ("Debt" if isin == 'DEBT' else "Fund")
                                override_p = final_prices.get(isin, 1.0)
                                log_investment(st.session_state.user_id, amt, isin, n, 4, date_override=sim_date,
                                               manual_price_override=override_p)
                        st.success("Executed!")
                        st.session_state.agent_result = None
                        st.session_state.selected_extras = []

                        # --- REDIRECTION FIX ---
                        # st.session_state.main_nav_radio = "📊 Portfolio"  # Force radio state
                        st.query_params["tab"] = "📊 Portfolio"  # Sync URL
                        st.rerun()


        ###
        # 1. Calculate & Display Rebalancing Gap
        st.markdown("### ⚖️ Portfolio Rebalancing")

        # --- DEFINITIONS FOR TOOLTIPS ---
        BUCKET_DESCRIPTIONS = {
            "Growth": "<b>High Risk / High Reward</b><br>Focus: Capital Appreciation.<br>Includes: Tech (QQQ), Global Indexes (SPY, MSCI World), Emerging Markets.",
            "Defensive": "<b>Moderate Risk / Stability</b><br>Focus: Dividends & Low Volatility.<br>Includes: Healthcare, Energy (Equinor), Staples (Mowi/Seafood), Gold.",
            "Cash": "<b>Zero Risk / Liquidity</b><br>Focus: Capital Preservation.<br>Includes: Bank Accounts, Money Market Funds, Short-term Bonds."
        }

        curr_exp, _ = get_portfolio_exposure(st.session_state.user_id)
        target_model = RISK_MODELS.get(risk, RISK_MODELS["Growth"])

        # Prepare Data for Chart
        reb_data = []
        for cat in ["Growth", "Defensive", "Cash"]:
            c_val = curr_exp.get(cat, 0.0) * 100
            t_val = target_model.get(cat, 0.0) * 100

            # We add the 'Description' field here so Plotly can see it
            reb_data.append({"Category": cat, "Type": "Current", "% Allocation": c_val, "Description": desc})
            reb_data.append({"Category": cat, "Type": "Target", "% Allocation": t_val, "Description": desc})


        # Prepare Data for Chart
        reb_data = []
        for cat in ["Growth", "Defensive", "Cash"]:
            c_val = curr_exp.get(cat, 0.0) * 100
            t_val = target_model.get(cat, 0.0) * 100
            desc = BUCKET_DESCRIPTIONS.get(cat, "")

            # We add the 'Description' field here so Plotly can see it
            reb_data.append({"Category": cat, "Type": "Current", "% Allocation": c_val, "Description": desc})
            reb_data.append({"Category": cat, "Type": "Target", "% Allocation": t_val, "Description": desc})

        df_reb = pd.DataFrame(reb_data)

        # Render Grouped Bar Chart with Custom Hover
        fig_reb = px.bar(
            df_reb,
            x="Category",
            y="% Allocation",
            color="Type",
            barmode="group",
            color_discrete_map={"Current": "#EF553B", "Target": "#636EFA"},
            title=f"Rebalancing Check ({risk} Profile)",
            # 1. Pass the description column to custom_data
            custom_data=["Description"]
        )

        # 2. Configure the Hover Template to show the description
        fig_reb.update_traces(
            hovertemplate="<b>%{x}</b><br>Allocation: %{y:.1f}%<br><br>%{customdata[0]}<extra></extra>"
        )

        st.plotly_chart(fig_reb, use_container_width=True)

        # 2. Logic Check
        # Optional: Show a warning if deviation is high
        for cat, t_val in target_model.items():
            c_val = curr_exp.get(cat, 0.0)
            if c_val > (t_val + 0.10):
                st.warning(
                    f"⚠️ You are Overweight in {cat} ({c_val:.0%} vs {t_val:.0%}). AI will likely reduce exposure here.")
        ###



    elif selected_tab == "📊 Portfolio":
        bs = get_balance_sheet(st.session_state.user_id)
        total_assets = sum(a['value'] for a in bs['assets']) + bs['market_portfolio_total']
        total_liab = sum(l['principal'] for l in bs['liabilities'])
        net_worth = total_assets - total_liab

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Assets", f"{total_assets:,.0f} NOK")
        m2.metric("Total Liabilities", f"{total_liab:,.0f} NOK")
        m3.metric("Net Worth", f"{net_worth:,.0f} NOK")

        st.subheader("Asset Allocation")
        data_pie = []
        for a in bs['assets']: data_pie.append({"Category": a['category'], "Value": a['value']})
        if bs['market_portfolio_total'] > 0: data_pie.append(
            {"Category": "Market Funds", "Value": bs['market_portfolio_total']})
        if data_pie:
            st.plotly_chart(px.pie(pd.DataFrame(data_pie), values='Value', names='Category'))

        st.markdown("---")

        # --- CONTROL BAR ---
        col_head, col_ctrl = st.columns([3, 1])
        col_head.subheader("Market Portfolio Performance")

        # Risk Timeframe Selector
        risk_timeframe = col_ctrl.selectbox(
            "Risk Lookback",
            options=["1wk", "1mo", "6mo", "1y"],
            index=3,
            format_func=lambda x: "Last " + x.replace("wk", "Week").replace("mo", "Month").replace("y", "Year")
        )

        # Fetch Raw Data
        df_raw = get_portfolio_df(st.session_state.user_id, risk_period=risk_timeframe)

        if not df_raw.empty:
            # ==========================================
            # 🆕 BENCHMARK COMPARISON (VS CASH)
            # ==========================================
            with st.container():
                st.markdown("#### 🆚 Benchmark: Portfolio vs. Savings Account")

                # 1. Controls
                c_bench_1, c_bench_2, c_bench_3 = st.columns([1, 1, 2])

                savings_rate_input = c_bench_1.number_input(
                    "Savings Rate (%)",
                    min_value=0.0, max_value=15.0, value=4.5, step=0.5
                )

                chart_type = c_bench_2.radio(
                    "View Type",
                    ["Bar (Snapshot)", "Line (History)", "🔮 Projection (5Y)"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

                # --- CALCULATE CURRENT STATE (Required for all views) ---
                total_invested = df_raw['Invested'].sum()
                current_portfolio_value = df_raw['Current Value'].sum()
                hypothetical_cash_value = 0.0
                today = datetime.now().date()

                # Calculate Hypothetical Cash Value (Opportunity Cost)
                for index, row in df_raw.iterrows():
                    days_held = (today - row['Date']).days
                    years_held = days_held / 365.25
                    rate_decimal = savings_rate_input / 100.0
                    hypothetical_cash_value += row['Invested'] * ((1 + rate_decimal) ** years_held)

                # --- VIEW LOGIC ---
                if chart_type == "Bar (Snapshot)":
                    # [Keep your existing Bar Chart Logic here]
                    alpha = current_portfolio_value - hypothetical_cash_value
                    is_beating = alpha >= 0

                    c_bench_3.metric(
                        label=f"Value vs. {savings_rate_input}% Savings",
                        value=f"{current_portfolio_value:,.0f} NOK",
                        delta=f"{alpha:+,.0f} NOK",
                        delta_color="normal"
                    )

                    bench_data = [
                        {"Scenario": "1. Principal", "Value": total_invested, "Color": "LightGray",
                         "Desc": "Total Cash Invested"},
                        {"Scenario": f"2. Savings ({savings_rate_input}%)", "Value": hypothetical_cash_value,
                         "Color": "#636EFA", "Desc": "Hypothetical Benchmark"},
                        {"Scenario": "3. Your Portfolio", "Value": current_portfolio_value,
                         "Color": "#00CC96" if is_beating else "#EF553B", "Desc": "Actual Market Value"}
                    ]
                    fig_bench = px.bar(pd.DataFrame(bench_data), x="Value", y="Scenario", orientation='h',
                                       text="Value", color="Scenario",
                                       color_discrete_map={d["Scenario"]: d["Color"] for d in bench_data},
                                       custom_data=["Desc"])
                    fig_bench.update_traces(texttemplate='%{text:,.0f}', textposition='auto',
                                            hovertemplate="<b>%{y}</b><br>%{x:,.0f} NOK<br>%{customdata[0]}<extra></extra>")
                    fig_bench.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_bench, use_container_width=True)

                elif chart_type == "Line (History)":
                    # [Keep your existing Line Chart Logic here]
                    # (Ensure you use the 'get_portfolio_evolution' function we fixed earlier)
                    with st.spinner("Reconstructing history..."):
                        df_history = get_portfolio_evolution(st.session_state.user_id, savings_rate_input)
                    if not df_history.empty:
                        latest = df_history.iloc[-1]
                        raw_profit = latest['Portfolio'] - latest['Invested']
                        alpha = latest['Portfolio'] - latest['Savings']
                        c1, c2 = st.columns(2)
                        c1.metric("Total Profit", f"{latest['Portfolio']:,.0f} NOK", f"{raw_profit:+,.0f} NOK")
                        c2.metric("vs. Savings", f"{alpha:+,.0f} NOK", "Excess Return", delta_color="normal")

                        fig_line = px.line(df_history, y=["Invested", "Savings", "Portfolio"], title="Past Performance",
                                           color_discrete_map={"Invested": "gray", "Savings": "#636EFA",
                                                               "Portfolio": "#00CC96" if alpha > 0 else "#EF553B"})
                        fig_line.update_traces(patch={"line": {"dash": "dash"}}, selector={"name": "Invested"})
                        fig_line.update_layout(hovermode="x unified", legend_title="", height=350)
                        st.plotly_chart(fig_line, use_container_width=True)

                else:
                    # ==========================================
                    # 🔮 NEW: FUTURE PROJECTION LOGIC
                    # ==========================================

                    # 1. Extra Input: Targeted Return
                    target_return = c_bench_3.number_input(
                        "Target Portfolio Return (%)",
                        min_value=0.0, max_value=30.0, value=8.0, step=0.5,
                        help="Assumed average annual return for your investment portfolio."
                    )

                    # 2. Generate Projection Data
                    years = 5
                    future_dates = pd.date_range(start=today, periods=years * 12, freq='ME')  # Monthly endpoints

                    proj_data = []

                    # Starting Values (From Today)
                    val_savings = hypothetical_cash_value
                    val_portfolio = current_portfolio_value

                    # Monthly Rates
                    r_save_mo = (savings_rate_input / 100) / 12
                    r_port_mo = (target_return / 100) / 12

                    for date in future_dates:
                        # Compound Growth
                        val_savings = val_savings * (1 + r_save_mo)
                        val_portfolio = val_portfolio * (1 + r_port_mo)

                        proj_data.append({
                            "Date": date,
                            "Scenario": f"Savings ({savings_rate_input}%)",
                            "Value": val_savings
                        })
                        proj_data.append({
                            "Date": date,
                            "Scenario": f"Portfolio ({target_return}%)",
                            "Value": val_portfolio
                        })

                    df_proj = pd.DataFrame(proj_data)

                    # 3. Calculate 5-Year Outcome
                    final_savings = df_proj[df_proj["Scenario"].str.contains("Savings")].iloc[-1]["Value"]
                    final_port = df_proj[df_proj["Scenario"].str.contains("Portfolio")].iloc[-1]["Value"]
                    diff = final_port - final_savings

                    # 4. Visualization
                    st.caption(f"Projecting values 5 years into the future based on assumed rates.")

                    fig_proj = px.line(
                        df_proj,
                        x="Date",
                        y="Value",
                        color="Scenario",
                        title="5-Year Wealth Projection",
                        color_discrete_map={
                            f"Savings ({savings_rate_input}%)": "#636EFA",
                            f"Portfolio ({target_return}%)": "#00CC96" if diff > 0 else "#EF553B"
                        }
                    )

                    # Add Area Fill to highlight the "Gap"
                    fig_proj.update_traces(fill='tonexty')
                    fig_proj.update_layout(hovermode="x unified", legend_title="", height=350)

                    st.plotly_chart(fig_proj, use_container_width=True)

                    # 5. Summary Text
                    if diff > 0:
                        st.success(
                            f"🎉 **Potential Gain:** By sticking to your strategy, you could generate an extra **{diff:,.0f} NOK** over 5 years compared to the bank.")
                    else:
                        st.error(
                            f"⚠️ **Warning:** At {target_return}%, your portfolio is projected to underperform the savings account by **{abs(diff):,.0f} NOK**.")

                st.divider()

            # ==========================================
            # 1. CONSOLIDATED HOLDINGS TABLE
            # ==========================================
            st.markdown("#### 1. Current Holdings (Grouped)")

            # Group by Fund Name and ISIN, Summing financial values
            df_grouped = df_raw.groupby(['Allocation', 'ISIN']).agg({
                'Invested': 'sum',
                'Current Value': 'sum',
                'Profit': 'sum',
                'Volatility': 'first',  # Risk metrics are same for the fund
                'Max Drawdown': 'first'
            }).reset_index()

            # Add "Return %" column for the group
            df_grouped['Return %'] = (df_grouped['Profit'] / df_grouped['Invested'])


            # Styling Logic (Red if Drawdown < -20%)
            def highlight_risk(row):
                if row['Max Drawdown'] < -0.20:
                    return ['background-color: #ffcccc; color: #990000'] * len(row)
                return [''] * len(row)


            # Display Styled Table
            st.dataframe(
                df_grouped.style.apply(highlight_risk, axis=1).format({
                    "Invested": "{:,.0f}",
                    "Current Value": "{:,.0f}",
                    "Profit": "{:+,.0f}",
                    "Return %": "{:.1%}",
                    "Volatility": "{:.1%}",
                    "Max Drawdown": "{:.1%}"
                }),
                use_container_width=True,
                column_order=["Allocation", "Invested", "Current Value", "Profit", "Return %", "Volatility",
                              "Max Drawdown"]
            )

            # ==========================================
            # 2. TRANSACTION LOG (DETAILED)
            # ==========================================
            with st.expander("📜 View Transaction Log & Edit History"):
                st.markdown("#### Detailed Trade History")

                # We use the raw dataframe here
                edited_df = st.data_editor(
                    df_raw,
                    column_config={
                        "Delete?": st.column_config.CheckboxColumn("Delete", default=False),
                        "Date": st.column_config.DateColumn("Date", format="DD.MM.YYYY"),
                        "ID": None,  # Hide ID
                        "Volatility": None,  # Hide metrics in log view to reduce clutter
                        "Max Drawdown": None
                    },
                    disabled=["Allocation", "ISIN", "Date", "Invested", "Current Value", "Profit"],
                    hide_index=True,
                    use_container_width=True
                )

                to_delete_ids = edited_df[edited_df["Delete?"] == True]["ID"].tolist()
                if to_delete_ids:
                    if st.button(f"🗑️ Delete Selected Rows ({len(to_delete_ids)})"):
                        delete_transactions(to_delete_ids)
                        st.success("Deleted!")
                        st.rerun()

            # --- CHARTING ---
            fund_isins = sorted([x for x in df_raw['ISIN'].unique() if x not in ['BANK', 'DEBT']])
            if fund_isins:
                st.markdown("### Fund Performance Chart")
                sel = st.selectbox("Select Fund", fund_isins, format_func=get_safe_fund_name,
                                   key="portfolio_chart_select")
                fig = plot_history(session, st.session_state.user_id, sel, "kr", 1.0)
                if fig: st.plotly_chart(fig)
        else:
            st.info("No market transactions yet.")