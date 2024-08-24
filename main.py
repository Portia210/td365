import pandas as pd
from datetime import datetime, timedelta
import configparser
from io import StringIO
import os
import glob
import subprocess
import sys
import numpy as np


def get_start_date():
    while True:
        start_date_input = input(
            'Enter the start date (dd/mm/yyyy), "today", "yest", or press Enter for all dates: ').lower()

        if start_date_input == "":
            return None
        elif start_date_input == "today":
            return datetime.now().date()
        elif start_date_input == "yest":
            return datetime.now().date() - timedelta(days=1)
        else:
            try:
                return datetime.strptime(start_date_input, "%d/%m/%Y").date()
            except ValueError:
                print("Invalid date format. Please use dd/mm/yyyy or 'today' or 'yesterday'.")

def clean_value(x):
    if isinstance(x, str):
        return x.replace('="', '').replace('"', '').strip()
    return x


def adjust_time(time_str, hours_to_add):
    time = datetime.strptime(time_str, '%H:%M:%S')
    adjusted_time = time + timedelta(hours=hours_to_add)
    return adjusted_time.strftime('%H:%M:%S')


def adjust_duration(duration_str, hours_to_add):
    hours, minutes, seconds = map(int, duration_str.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    adjusted_seconds = total_seconds + hours_to_add * 3600
    adjusted_hours = adjusted_seconds // 3600
    adjusted_minutes = (adjusted_seconds % 3600) // 60
    adjusted_seconds = adjusted_seconds % 60
    return f"{adjusted_hours:02d}:{adjusted_minutes:02d}:{adjusted_seconds:02d}"


def get_latest_transaction_history_file(directory):
    pattern = os.path.join(directory, '*TransactionHistory*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No TransactionHistory CSV file found in {directory}")
    return max(files, key=os.path.getmtime)


def process_csv(file_path):
    # Load and clean the CSV file
    with open(file_path, 'r', encoding="UTF-16") as file:
        data = file.read()
    cleaned_data = data.replace('="', '').replace('"', '')
    df = pd.read_csv(StringIO(cleaned_data), delimiter='\t')
    df = df.apply(lambda col: col.map(clean_value) if col.dtype == 'object' else col)

    # Process date and time columns
    df.rename(columns={"Description": "Symbol"}, inplace=True)
    df[['Date', 'Open Time']] = df['Open Period'].str.split(n=1, expand=True)
    df['Close Time'] = pd.to_datetime(df['Transaction Date'], format='%d/%m/%Y %H:%M:%S').dt.strftime('%H:%M:%S')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Open Time'] = pd.to_datetime(df['Open Time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')

    # Calculate Trade Duration
    open_time = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Open Time'])
    close_time = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Close Time'])
    df['Trade Duration'] = (close_time - open_time).dt.components['hours'].astype(str).str.zfill(2) + ':' + \
                           (close_time - open_time).dt.components['minutes'].astype(str).str.zfill(2) + ':' + \
                           (close_time - open_time).dt.components['seconds'].astype(str).str.zfill(2)

    # Reorder and select columns
    df = df[['Date', 'Open Time', 'Close Time', 'Trade Duration', 'Symbol', 'Amount', 'Opening', 'Closing', 'P/L',
             'Balance']]

    # Read the configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')
    time_difference = float(config['Timezone']['offset'])

    # Adjust times
    df['Open Time'] = df['Open Time'].apply(lambda x: adjust_time(x, time_difference))
    df['Close Time'] = df['Close Time'].apply(lambda x: adjust_time(x, time_difference))
    df['Trade Duration'] = df['Trade Duration'].apply(lambda x: adjust_duration(x, 0))

    # Adjust Date if necessary
    df['Date'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Open Time'])
    df['Date'] = df['Date'].dt.date

    # Sort by 'Close Time'
    df = df.sort_values(by='Close Time')

    # Calculate Points and Daily P/L
    df['Points'] = (df['Closing'].astype(float) - df['Opening'].astype(float)).round(1)
    df['P/L'] = df['P/L'].astype(float)

    return df


def export_processed_trades(df, date):
    output_date = date.strftime("%d-%m-%Y")
    df.to_csv(f"{output_date} trades.csv", index=False)
    print(f"Exported trades data for {output_date}")


def analyze_profitability(df):
    # Filter out irrelevant entries
    df = df[df["Symbol"] != "Online Transfer Cash In"]

    # Calculate Gain Trades
    gain_trades_table = df[df['P/L'] > 0]
    gain_trades_table.loc[:, "Amount"] = np.abs(gain_trades_table["Amount"].values)
    gain_trades = gain_trades_table.groupby('Symbol')
    gain_trades_df = pd.DataFrame({
        'Symbol': gain_trades['P/L'].count().index,
        'Gain Trades': gain_trades['Amount'].sum().values,
        'Gains': gain_trades['P/L'].sum().values,
    })
    gain_trades_df['Avg Gain'] = gain_trades_df.apply(
        lambda row: round(row['Gains'] / row['Gain Trades'], 1) if row['Gain Trades'] > 0 else 0, axis=1
    )

    # Calculate Loss Trades
    loss_trades_table = df[df['P/L'] < 0]
    loss_trades_table.loc[:, "Amount"] = np.abs(loss_trades_table["Amount"].values)
    loss_trades_table.loc[:, "P/L"] = np.abs(loss_trades_table["P/L"].values)
    loss_trades = loss_trades_table.groupby('Symbol')
    loss_trades_df = pd.DataFrame({
        'Symbol': loss_trades['P/L'].count().index,
        'Loss Trades': loss_trades['Amount'].sum().values,
        'Losses': -loss_trades['P/L'].sum().values,
    })
    loss_trades_df['Avg Loss'] = loss_trades_df.apply(
        lambda row: round(row['Losses'] / row['Loss Trades'], 1) if row['Loss Trades'] > 0 else 0, axis=1
    )

    # Calculate Breakeven Trades
    breakeven_trades_table = df[(df['P/L'] >= -2) & (df['P/L'] <= 2)]  # Assuming -2 < P/L < 2 is the breakeven range
    breakeven_trades = breakeven_trades_table.groupby('Symbol')
    breakeven_trades_df = pd.DataFrame({
        'Symbol': breakeven_trades['P/L'].count().index,
        'Breakeven Trades': breakeven_trades['Amount'].count().values,
        'Breakeven': breakeven_trades['P/L'].sum().values
    })
    breakeven_trades_df['Avg Breakeven'] = breakeven_trades_df.apply(
        lambda row: round(row['Breakeven'] / row['Breakeven Trades'], 1) if row['Breakeven Trades'] > 0 else 0, axis=1
    )

    # Merge all DataFrames
    result_df = pd.merge(gain_trades_df, loss_trades_df, on='Symbol', how='outer')
    result_df = pd.merge(result_df, breakeven_trades_df, on='Symbol', how='outer')
    result_df = result_df.fillna(0)

    # Calculate percentages
    total_trades = result_df['Gain Trades'] + result_df['Loss Trades'] + result_df['Breakeven Trades']
    result_df['Profit %'] = (result_df['Gain Trades'] / total_trades * 100).round(1)
    result_df['Loss %'] = (result_df['Loss Trades'] / total_trades * 100).round(1)
    result_df['Breakeven %'] = (result_df['Breakeven Trades'] / total_trades * 100).round(1)

    # Calculate the overall sum
    result_df['Sum'] = (result_df['Gains'] + result_df['Losses'] + result_df['Breakeven']).round(1)

    # Calculate Total row
    total = pd.DataFrame({
        'Symbol': ['Total'],
        'Gain Trades': [result_df['Gain Trades'].sum()],
        'Loss Trades': [result_df['Loss Trades'].sum()],
        'Breakeven Trades': [result_df['Breakeven Trades'].sum()],
        'Gains': [result_df['Gains'].sum().round(1)],
        'Losses': [result_df['Losses'].sum().round(1)],
        'Breakeven': [result_df['Breakeven'].sum().round(1)],
        'Sum': [result_df['Sum'].sum().round(1)]
    })

    total_contracts = total['Gain Trades'].values[0] + total['Loss Trades'].values[0] + \
                      total['Breakeven Trades'].values[0]
    total['Profit %'] = round(total['Gain Trades'] / total_contracts * 100, 1) if total_contracts > 0 else 0
    total['Loss %'] = round(total['Loss Trades'] / total_contracts * 100, 1) if total_contracts > 0 else 0
    total['Breakeven %'] = round(total['Breakeven Trades'] / total_contracts * 100, 1) if total_contracts > 0 else 0
    total['Avg Gain'] = round(total['Gains'] / total['Gain Trades'], 1) if total['Gain Trades'].values[0] > 0 else 0
    total['Avg Loss'] = round(total['Losses'] / total['Loss Trades'], 1) if total['Loss Trades'].values[0] > 0 else 0
    total['Avg Breakeven'] = round(total['Breakeven'] / total['Breakeven Trades'], 1) if \
    total['Breakeven Trades'].values[0] > 0 else 0

    # Append Total row to result_df
    result_df = pd.concat([result_df, total], ignore_index=True)

    # Reorder the columns
    result_df = result_df[
        ['Symbol', 'Gain Trades', 'Loss Trades', 'Breakeven Trades', 'Profit %', 'Loss %', 'Breakeven %', 'Avg Gain',
         'Avg Loss', 'Avg Breakeven', 'Gains', 'Losses', 'Breakeven', 'Sum']]

    print(result_df.to_string())
    return result_df


def export_performance(result_df, date):
    output_date = date.strftime("%d-%m-%Y")
    result_df.to_csv(f"{output_date} performance.csv", index=False)
    print(f"Exported performance data for {output_date}")


def delete_csv_files():
    for file in glob.glob("*.csv"):
        os.remove(file)
    print("Deleted existing CSV files.")


def main():
    # Delete existing CSV files
    delete_csv_files()

    # Get the latest TransactionHistory file
    downloads_dir = r"C:\Users\hp\Downloads"
    csv_path = get_latest_transaction_history_file(downloads_dir)
    print(f"Using file: {csv_path}")

    # Get start date input
    start_date = get_start_date()

    if start_date:
        print(f"Analysis will start from: {start_date}")
    else:
        print("Analysis will include all dates")

    # Process the CSV file
    df = process_csv(csv_path)

    # Filter data based on start date if provided
    if start_date:
        df = df[df['Date'] >= start_date]

    # Group by date and analyze profitability for each date
    for date, group in df.groupby('Date'):
        # Export processed trades
        export_processed_trades(group, date)

        # Analyze profitability
        performance_df = analyze_profitability(group)

        # Export performance analysis
        export_performance(performance_df, date)

    print("Processing completed. Check the generated trades and performance CSV files.")


if __name__ == "__main__":
    main()