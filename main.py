import pandas as pd
from datetime import datetime, timedelta
import configparser
from io import StringIO
import os
import glob
import subprocess
import sys


def get_start_date():
    while True:
        start_date_input = input(
            'Enter the start date (dd/mm/yyyy), "today", "yesterday", or press Enter for all dates: ').lower()

        if start_date_input == "":
            return None
        elif start_date_input == "today":
            return datetime.now().date()
        elif start_date_input == "yesterday":
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
    df = df[['Date', 'Open Time', 'Close Time', 'Trade Duration', 'Description', 'Amount', 'Opening', 'Closing', 'P/L',
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


def analyze_profitability(df, date):
    # Calculate the winning and losing trade data
    gain_trades = df[df['P/L'] > 0].groupby('Description')
    gain_trades_df = pd.DataFrame({
        'Symbol': gain_trades['P/L'].count().index,
        'Gain Trades': gain_trades['P/L'].count().values,
        'Gains': gain_trades['P/L'].sum().values,
        'Avg Gain': gain_trades['P/L'].mean().values
    })

    loss_trades = df[df['P/L'] < 0].groupby('Description')
    loss_trades_df = pd.DataFrame({
        'Symbol': loss_trades['P/L'].count().index,
        'Loss Trades': loss_trades['P/L'].count().values,
        'Losses': loss_trades['P/L'].sum().values,
        'Avg Loss': loss_trades['P/L'].mean().values
    })

    # Merge the winning and losing trade DataFrames
    result_df = pd.merge(gain_trades_df, loss_trades_df, on='Symbol', how='outer')
    result_df = result_df.fillna(0)

    # Calculate the Profit Ratio, Avg Gain per Contract, and Avg Loss per Contract
    result_df['Profit Ratio'] = (
                result_df['Gain Trades'] / (result_df['Gain Trades'] + result_df['Loss Trades'])).round(2)
    result_df['Avg Gain P.C'] = result_df['Avg Gain'].round(1)
    result_df['Avg Loss P.C'] = result_df['Avg Loss'].round(1)
    result_df['Gains'] = result_df['Gains'].round(1)
    result_df['Losses'] = result_df['Losses'].round(1)
    result_df['Sum'] = (result_df['Gains'] + result_df['Losses']).round(1)

    # Reorder the columns
    result_df = result_df[
        ['Symbol', 'Gain Trades', 'Loss Trades', 'Profit Ratio', 'Avg Gain P.C', 'Avg Loss P.C', 'Gains', 'Losses',
         'Sum']]

    # Format the date for the output file name
    output_date = date.strftime("%d-%m-%Y")

    # Export to CSV
    result_df.to_csv(f"{output_date} trades.csv", index=False)

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
        analyze_profitability(group, date)

    print("Processing completed. Check the generated performance CSV files.")
    # Run the analysis script
    print("Starting analysis...")
    try:
        subprocess.run([sys.executable, "analyze.py"], check=True)
        print("Analysis completed successfully.")
    except subprocess.CalledProcessError:
        print("Error occurred during analysis.")


if __name__ == "__main__":
    main()