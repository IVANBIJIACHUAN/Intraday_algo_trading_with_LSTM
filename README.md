# Intraday_algo_trading_with_LSTM

Project for algo trading course.

Predict move and downs with intraday trade data using LSTM. Achieve better predictivity than tick factor (ewa momentum).

2020/04/04 Update: 

1_WRDS_data_fetcher.ipynb: Use WRDS to fetch quote data and preprocess, saving to file "data"

2_LSTM_prediction_upanddown_v2.ipynb: save model and scaler to file "model"

3_LSTM_enhanced_VWAP: Add tools gen_train_test_data.py, LSTM.py, vwap2.py. Merge preprocessed quotes and trades (TAQ) data. Create test cases for LSTM-enhanced vwap. Slightly better than tick factor with some specific coefficients.
