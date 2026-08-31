import os
import io
import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import functions_framework

# Configuration Variables
MAX_PER_SECTOR = 2
PORTFOLIO_SIZE = 10
MAX_SMA_EXTENSION_PCT = 35.0
WEIGHTING_METHOD = "equal"  # Options: "equal", "momentum", "rank"
MOMENTUM_POWER = 1.0


def get_sp500_universe():
    """Fetches current S&P 500 tickers and sector mapping."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        tables = pd.read_html(io.StringIO(res.text))
        df = tables[0][['Symbol', 'Security', 'GICS Sector']].copy()
        df.columns = ['Ticker', 'Company', 'Sector']
        df['Ticker'] = df['Ticker'].str.replace('.', '-', regex=False)
        return df
    except Exception as e:
        print(f"Fallback to static CSV universe due to: {e}")
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        df = pd.read_csv(url)
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        return df.rename(columns={'Symbol': 'Ticker', 'Name': 'Company', 'Sector': 'Sector'})


def download_price_history_in_chunks(tickers, start_date, end_date, chunk_size=50):
    """Downloads yfinance data in chunks to avoid rate-limits and URL length limits."""
    all_data = []
    total_tickers = len(tickers)
    
    for i in range(0, total_tickers, chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            df = yf.download(chunk, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"   [!] Error downloading chunk: {e}")
            
    if not all_data:
        raise RuntimeError("Failed to download any price data from Yahoo Finance.")
        
    return pd.concat(all_data, axis=1)


def select_portfolio_candidates(close_prices, sp_df, as_of_date):
    """Core selection logic using 6M momentum, 200-SMA gate, and extension checks."""
    valid_rows = []
    
    hist_subset = close_prices.loc[:as_of_date]
    if len(hist_subset) < 200:
        return pd.DataFrame()

    for _, sp_row in sp_df.iterrows():
        sym = sp_row['Ticker']
        if sym not in hist_subset.columns:
            continue

        series = hist_subset[sym].dropna()
        if len(series) < 126:  
            continue

        latest_p = series.iloc[-1]
        sma_200 = series.iloc[-200:].mean()

        if latest_p < sma_200:
            continue

        extension_pct = ((latest_p - sma_200) / sma_200) * 100.0
        if extension_pct > MAX_SMA_EXTENSION_PCT:
            continue

        mom_6m = ((latest_p - series.iloc[-126]) / series.iloc[-126]) * 100.0

        valid_rows.append({
            'Ticker': sym,
            'Company': sp_row['Company'],
            'Sector': sp_row['Sector'],
            'Latest_Price': round(latest_p, 2),
            '6M_Momentum_%': round(mom_6m, 2),
            'Extension_%': round(extension_pct, 2)
        })

    df_cand = pd.DataFrame(valid_rows)
    if df_cand.empty:
        return pd.DataFrame()

    df_cand.sort_values(by='6M_Momentum_%', ascending=False, inplace=True)

    selected, sector_counts = [], {}
    for _, row in df_cand.iterrows():
        sec = row['Sector']
        if sector_counts.get(sec, 0) < MAX_PER_SECTOR:
            selected.append(row)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(selected) == PORTFOLIO_SIZE:
            break

    res_df = pd.DataFrame(selected).reset_index(drop=True)
    res_df['Rank'] = res_df.index + 1

    if WEIGHTING_METHOD == "equal":
        raw_w = np.ones(len(res_df))
    elif WEIGHTING_METHOD == "rank":
        raw_w = 1.0 / res_df['Rank']
    else:
        safe_mom = res_df['6M_Momentum_%'].clip(lower=0.1)
        raw_w = safe_mom ** MOMENTUM_POWER

    res_df['Target_Weight_%'] = ((raw_w / raw_w.sum()) * 100.0).round(2)
    return res_df


def send_slack_notification(message):
    """Sends notification to Slack using proper JSON payload structure to prevent 400 errors."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/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")

    payload = {"text": message}
    
    response = requests.post(webhook_url, json=payload, timeout=10)
    response.raise_for_status()


@functions_framework.http
def run_rebalance_function(request):
    """Cloud Function entry point matching the target configuration."""
    try:
        print("Fetching S&P 500 universe...")
        sp_df = get_sp500_universe()
        tickers = sp_df['Ticker'].tolist()

        end_date = datetime.date.today().strftime('%Y-%m-%d')
        start_date = (datetime.date.today() - datetime.timedelta(days=400)).strftime('%Y-%m-%d')

        print("Downloading recent price history in batches...")
        raw_hist = download_price_history_in_chunks(tickers, start_date, end_date, chunk_size=50)
        
        close_prices = raw_hist['Close'] if 'Close' in raw_hist.columns else raw_hist['Adj Close']
        close_prices = close_prices.ffill().dropna(how='all', axis=1)

        as_of_date = close_prices.index[-1].strftime('%Y-%m-%d')
        portfolio = select_portfolio_candidates(close_prices, sp_df, as_of_date)

        if portfolio.empty:
            msg = "⚠️ Momentum Rebalance Signal: No stocks met the selection criteria this week."
        else:
            header = f"📊 *Weekly Momentum Rebalance Signal* ({as_of_date})\n*Weighting Method:* `{WEIGHTING_METHOD.upper()}`\n\n```"
            table_str = portfolio[['Rank', 'Ticker', 'Sector', '6M_Momentum_%', 'Target_Weight_%']].to_string(index=False)
            footer = "```"
            msg = f"{header}\n{table_str}\n{footer}"

        send_slack_notification(msg)
        return "Rebalance signal successfully sent to Slack.", 200

    except Exception as e:
        error_msg = f"Error executing weekly rebalance signal: {str(e)}"
        print(error_msg)
        return error_msg, 500
