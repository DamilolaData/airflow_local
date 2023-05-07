from pprint import pprint
import yfinance as yf
import pandas as pd
import datetime
import pickle
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

@task
def get_and_process_tickers(file_path):
    df = pd.read_csv(file_path)
    stock_list = df[df['ETF'] == 'N']['Symbol'].to_list()
    etf_list = df[df['ETF'] == 'Y']['Symbol'].to_list()
    stock_list = stock_list[0:2]
    etf_list = etf_list[0:2]
    value_dict = {'stock': stock_list,
                  'etf': etf_list}
    return value_dict

@task
def download_data(meta_info, start_date='2023-01-01', end_date='2023-05-02'):
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    _all_data = []

    for category, ticker_list in meta_info.items():
        for ticker in ticker_list:
            print(f"Downloading data for {ticker}")
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                data['Symbol'] = ticker
                data['Security Name'] = category
                data.reset_index(inplace=True)
                _all_data.append(data)
            except Exception as e:
                print(f"Error downloading data for {ticker}: {e}")

    df_final = pd.concat(_all_data)
    df_final = df_final[['Symbol', 'Security Name', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    #final_df.to_parquet("/opt/airflow/dags/files/yfinance_data.parquet")
    #df_final.to_csv("/opt/airflow/dags/files/data/stg/yfinance_data.csv")
    df_final.to_parquet("/opt/airflow/dags/files/data/stg/yfinance_data.parquet")

    print(f'processed {df_final.shape[0]} rows')
    return 'yfinance_data.parquet'


def feature_engineering():
    df_final = pd.read_parquet('/opt/airflow/dags/files/data/stg/yfinance_data.parquet')

    # Calculate 30-day moving average of trading volume for each stock/ETF
    print(df_final.head())
    df_final['vol_moving_avg'] = df_final.groupby('Symbol')['Volume'].rolling(window=30).mean().reset_index(drop=True)

    # Calculate rolling median of "Adj Close" column and store in new column
    df_final['adj_close_rolling_med'] = df_final['Adj Close'].rolling(window=30).median()

    df_final = df_final[['Symbol', 'Security Name', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume','vol_moving_avg','adj_close_rolling_med']]

    #df_final.to_csv("/opt/airflow/dags/files/data/prd/yfinance_data.csv")
    df_final.to_parquet("/opt/airflow/dags/files/data/prd/yfinance_data.parquet")
    #df_final.to_parquet("/opt/airflow/dags/files/rt_ai.parquet")

    print(f'processed {df_final.shape[0]} rows')
    print(df_final.head())
    return 'yfinance_data.csv'

def ml_training():
    # Assume `data` is loaded as a Pandas DataFrame
    df_final = pd.read_parquet('/opt/airflow/dags/files/data/prd/yfinance_data.parquet')
    df_final['Date'] = pd.to_datetime(df_final['Date'])
    df_final.set_index('Date', inplace=True)

    # Remove rows with NaN values
    df_final.dropna(inplace=True)
    print(f'processed {df_final.shape[0]} rows')
    print(df_final.head())
    # Select features and target
    features = ['vol_moving_avg', 'adj_close_rolling_med']
    target = 'Volume'

    X = df_final[features]
    y = df_final[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model to disk
    with open('/opt/airflow/dags/files/data/mdl/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Load the model from disk
    with open('/opt/airflow/dags/files/data/mdl/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Print the model object
    print(model)

    # Make predictions on test data
    y_pred = model.predict(X_test)
    print(y_pred)

    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(mae)
    print(mse)


with DAG(
    dag_id='riskthinking_ai_de',
    default_args={
        'owner': 'airflow',
    },
    schedule_interval='@daily',
    start_date=days_ago(2),
    catchup=False,
    tags=['example'],
) as dag:

    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')

    get_tickers_task = get_and_process_tickers("/opt/airflow/dags/files/symbols_valid_meta.csv")
    download_task = download_data(get_tickers_task)
    feature_engineering_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering,
        provide_context=True,
        dag=dag
    )
    ml_training_task = PythonOperator(
        task_id='ml_training',
        python_callable=ml_training,
        provide_context=True,
        dag=dag
    )

    start >> get_tickers_task >> download_task >> feature_engineering_task >> ml_training_task >> end
