import os
import re
import pandas as pd

def _extract_dividend_amount(description: str, face_value: float):
    # Function to extract dividend amount

    # Check for percentage pattern, e.g., "Dividend 20%"
    percentage_match = re.search(r'(\d+)%', description)
    if percentage_match:
        percent_value = float(percentage_match.group(1))
        return (percent_value / 100) * face_value

    # Check for absolute amount pattern, e.g., "Dividend Rs 5" or "Re 1"
    absolute_match = re.search(r'Rs.\s?(\d*\.?\d+)|Re\s?(\d*\.?\d+)', description)
    if absolute_match:
        # Extract either Rs or Re value
        amount_value = float(absolute_match.group(1) or absolute_match.group(2))
        return amount_value

        # Return NaN if no dividend information found
    return None


if __name__ == '__main__':
    path = r'C:\Users\paras\PycharmProjects\EQQuant\additional_data\nifty50_corporate_actions\dividend_nifty50_stocks.csv'
    div_df = pd.read_csv(path)
    div_df['dividend_amount'] = div_df.apply(
                lambda x: _extract_dividend_amount(x['PURPOSE'], x['FACE VALUE']), axis=1)
    div_df.to_csv(r'C:\Users\paras\PycharmProjects\EQQuant\additional_data\nifty50_corporate_actions\dividend_nifty50_stocks_processed.csv')