﻿# Riskthinking_ai
 
In this project I created a pipeline using Airflow and Docker to carryout the following tasks:

1.  Download the ETF and stock datasets from the primary dataset available at https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset.

2.  Setup a data structure to retain all data from ETFs and stocks with the following columns.

Symbol: string
Security Name: string
Date: string (YYYY-MM-DD)
Open: float
High: float
Low: float
Close: float
Adj Close: float
Volume: A suitable Number type (int or float)
Convert the resulting dataset into a structured format (e.g. Parquet).

3.  Build some feature engineering on top of the dataset to calculate the following:

i.  Calculate the moving average of the trading volume (Volume) of 30 days per each stock and ETF, and retain it in a newly added column vol_moving_avg.
ii. Similarly, calculate the rolling median and retain it in a newly added column adj_close_rolling_med.
iii.  Retain the resulting dataset into the same format as Problem 1, but in its own stage/directory distinct from the first.

4.   Integrated an ML predictive model training step into the data pipeline.
5.   Created an API to serve the model and Jupyter notebook of the working API.
