import pandas as pd
import os
import glob


def analyze_profitability(file_name):
    df = pd.read_csv(file_name)

    # Check for the correct column name for P/L
    pl_column = 'P/L'
    if pl_column not in df.columns:
        possible_pl_columns = [col for col in df.columns if 'p' in col.lower() and 'l' in col.lower()]
        if possible_pl_columns:
            pl_column = possible_pl_columns[0]
        else:
            print(f"Error: Could not find P/L column in {file_name}")
            print("Available columns:", df.columns.tolist())
            return

    # Check for the correct column name for Description
    desc_column = 'Description'
    if desc_column not in df.columns:
        possible_desc_columns = [col for col in df.columns if 'desc' in col.lower() or 'symb' in col.lower()]
        if possible_desc_columns:
            desc_column = possible_desc_columns[0]
        else:
            print(f"Error: Could not find Description column in {file_name}")
            print("Available columns:", df.columns.tolist())
            return

    # Calculate the winning and losing trade data
    gain_trades = df[df[pl_column] > 0].groupby(desc_column)
    gain_trades_df = pd.DataFrame({
        'Symbol': gain_trades[pl_column].count().index,
        'Gain Trades': gain_trades[pl_column].count().values,
        'Gains': gain_trades[pl_column].sum().values,
        'Avg Gain': gain_trades[pl_column].mean().values
    })

    loss_trades = df[df[pl_column] < 0].groupby(desc_column)
    loss_trades_df = pd.DataFrame({
        'Symbol': loss_trades[pl_column].count().index,
        'Loss Trades': loss_trades[pl_column].count().values,
        'Losses': loss_trades[pl_column].sum().values,
        'Avg Loss': loss_trades[pl_column].mean().values
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

    print(f"Analysis for {file_name}:")
    print(result_df.to_string(index=False))

    # Generate output file name
    output_file = file_name.replace('.csv', '_performance.csv')
    result_df.to_csv(output_file, index=False)
    print(f"Performance data saved to {output_file}\n")


def main():
    # Get all CSV files with 'trades' in the filename
    trade_files = glob.glob('*trades*.csv')

    if not trade_files:
        print("No CSV files with 'trades' in the name found in the current directory.")
        return

    for file in trade_files:
        analyze_profitability(file)


if __name__ == "__main__":
    main()