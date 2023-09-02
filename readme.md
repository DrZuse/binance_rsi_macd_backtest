# RSI MACD Strategy Analyzer

This Python script is designed to analyze historical price data of a cryptocurrency (in this case, BTCUSDT) to identify potential trading opportunities based on the Relative Strength Index (RSI) and Moving Average Convergence Divergence (MACD) indicators. It also calculates the profitability of the trading strategy.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

This script reads historical price data from CSV files and performs the following tasks:

1. Reads historical price data from CSV files containing OHLCV (Open, High, Low, Close, Volume) data.
2. Calculates the RSI (Relative Strength Index) with a user-defined time window (default is 14).
3. Calculates the MACD (Moving Average Convergence Divergence) indicator, including MACD line, signal line, and MACD histogram.
4. Identifies potential trading signals based on RSI and MACD.
5. Calculates the profitability of the trading strategy based on predefined conditions.

## Prerequisites

Before using this script, make sure you have the following prerequisites in place:

- Python 3.x installed on your system.
- Required Python packages: `numpy`, `pandas`, `multiprocessing`.

You can install these packages using pip:

```bash
pip install numpy pandas
```

## Usage

1. Clone or download this repository to your local machine.

2. Place your CSV files containing historical price data in the `../big_dataframes/binance/spot/daily/klines/BTCUSDT/1s/` directory. Ensure that the CSV files have the following columns: 'Open time', 'Open', 'High', 'Low', 'Close', 'Volume'. You can modify the `names` variable in the script to match your column names if they are different.

3. Customize the script as needed. You can adjust the RSI time window (`rsi_value`) and other parameters to match your trading strategy.

4. Run the script:

```bash
python multi_backtest.py
```

## Results

The script will output the following information:

- Number of profitable trades.
- Number of losing trades.
- Profit ratio (profitable trades / losing trades).
- User-defined profit and loss percentages.
- User-defined RSI value.
- Total profit (profitable trades * profit percentage - losing trades * loss percentage).

The results will help you evaluate the performance of your trading strategy based on the provided parameters and historical price data.

## License

This script is provided under the MIT License. Feel free to use, modify, or distribute it as needed.
