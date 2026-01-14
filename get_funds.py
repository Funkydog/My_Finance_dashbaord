import json
import re
import requests
from datetime import datetime, timedelta

def fetch_fund_history(isin, fund_name):
    """
    Fetches 3+ years of historical data by parsing the JSON
    embedded in the Financial Times chart page.
    """
    # FT Chart URL
    url = f"https://markets.ft.com/data/funds/tearsheet/charts?s={isin}:NOK"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # 1. Regex to find the JSON data hidden in the script tags
        # FT stores chart data in a variable often named 'elements' or inside the config
        # We look for the standardized JSON structure used by their charting lib
        data_pattern = re.search(r'\"elements\":\[(.*?)\]', response.text)

        if not data_pattern:
            print(f"Could not extract data for {fund_name}")
            return []

        # 2. Parse the JSON
        # The regex captures the list inside "elements": [...]
        json_str = f"[{data_pattern.group(1)}]"
        data = json.loads(json_str)

        # 3. Extract Time Series
        # The structure is usually: Component -> Series -> Data
        # We iterate to find the 'Line' chart which contains the historical Nav
        history = []

        if data and 'Component' in data[0]:
            series_data = data[0]['Component']['Series']
            for series in series_data:
                # We only want the main fund data (Type usually 'Line')
                if series.get('Type') == 'Line':
                    for point in series.get('Data', []):
                        # FT dates are often ISO strings
                        d_date = datetime.strptime(point['Date'], "%Y-%m-%dT%H:%M:%S")

                        # Filter for last 3 years
                        if d_date > datetime.now() - timedelta(days=3 * 365):
                            history.append({
                                "isin": isin,
                                "fund_name": fund_name,
                                "date": d_date,
                                "close_price": point['Value']
                            })
                    break  # Stop after finding the main line

        return history

    except Exception as e:
        print(f"Error fetching {fund_name}: {e}")
        return []


def update_database_with_history(session):
    """
    Orchestrator: Runs the fetcher for all funds and updates DB.
    """
    funds = {
        "DNB Norden Indeks": "NO0010815871",
        "KLP AksjeGlobal P": "NO0010776040",
        "KLP AksjeEuropa Indeks P": "NO0010745862",
        "Norne Kombi 50": "SE0015194634",
        "Norne Aksje Norge": "NO0010922859"
    }

    count = 0
    for name, isin in funds.items():
        # Check if we already have recent data to avoid spamming
        last_entry = session.query(FundPriceHistory).filter_by(isin=isin).order_by(FundPriceHistory.date.desc()).first()

        # If no data or data is old (> 7 days), fetch update
        if not last_entry or last_entry.date < datetime.now() - timedelta(days=7):
            print(f"Updating history for {name}...")
            new_data = fetch_fund_history(isin, name)

            for entry in new_data:
                # Optional: Check if date exists to avoid duplicates
                exists = session.query(FundPriceHistory).filter_by(isin=isin, date=entry['date']).first()
                if not exists:
                    record = FundPriceHistory(**entry)
                    session.add(record)
                    count += 1
            session.commit()

    return count