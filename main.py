import pandas as pd
from datetime import datetime, timedelta
import configparser
from io import StringIO
import os
import glob


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


def process_csv(file_path, selected_date):
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

    # Filter data based on selected date
    df = df[df['Date'] == selected_date]

    # Sort by 'Close Time'
    df = df.sort_values(by='Close Time')

    # Calculate Points and Daily P/L
    df['Points'] = (df['Closing'].astype(float) - df['Opening'].astype(float)).round(1)
    df['Daily P/L'] = df['P/L'].cumsum().round(1)

    # Reorder columns
    columns_order = ['Date', 'Open Time', 'Close Time', 'Trade Duration', 'Description', 'Amount', 'Opening', 'Closing',
                     'Points',
                     'P/L', 'Daily P/L', 'Balance']
    df = df[columns_order]

    return df


def main():
    # Get the latest TransactionHistory file
    downloads_dir = r"C:\Users\hp\Downloads"
    csv_path = get_latest_transaction_history_file(downloads_dir)
    print(f"Using file: {csv_path}")

    # Get date input
    date_input = input('Enter the date (dd/mm) or press Enter for today\'s date: ')
    if date_input:
        selected_date = datetime.strptime(f"{date_input}/{datetime.now().year}", "%d/%m/%Y").date()
    else:
        selected_date = datetime.now().date()

    # Process the CSV file
    result_df = process_csv(csv_path, selected_date)

    # Format the DataFrame for Excel
    for col in result_df.columns:
        if result_df[col].dtype == 'float64':
            result_df[col] = result_df[col].map('{:.2f}'.format)

    # Format the Date column
    result_df['Date'] = result_df['Date'].astype(str)

    # Save the result in Excel-friendly format
    output_path = os.path.join(f'{selected_date}.csv')
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"Processing completed. Check '{output_path}' for the result.")


if __name__ == "__main__":
    main()
