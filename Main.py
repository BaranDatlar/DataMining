from StockMarketData import StockMarketData
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


def ParseJSONToDataFrame(data):
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)  # Verileri float türüne çevirme
    return df

def CreateNormalizedData(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Normalleştirilmiş veriyi DataFrame'e geri dönüştürme
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return scaled_df

stockMarketData = StockMarketData("savedJSONFile.json")

data = stockMarketData.ExtractDataJSON()
df = ParseJSONToDataFrame(data)
scaled_df = CreateNormalizedData(df)


# Veri ön işlemesi için adımlar
# Zaman serisi pencereleme işlemi örneği

# Öncelikle veriyi eğitim ve test seti olarak ayıracağız.
train_size = int(len(scaled_df) * 0.8)
test_size = len(scaled_df) - train_size
train, test = scaled_df.iloc[0:train_size], scaled_df.iloc[train_size:len(scaled_df)]

print(test.head())

# Pencere boyutunu belirleme
window_size = 3

# Pencereleme fonksiyonu

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Eğitim ve test veri setlerini pencereleme
X_train, y_train = create_dataset(train[['close']], train.close, window_size)
X_test, y_test = create_dataset(test[['close']], test.close, window_size)

print(X_train.shape, y_train.shape)

print("INDEX" , scaled_df.index[0])

# Model kurulumu
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Modeli derleme
model.compile(optimizer='adam', loss='mean_squared_error')

# Model eğitimi
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

# Loss grafikleri

# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.title('Model Loss Progress')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()

predicted = model.predict(X_test)

# Tahminlerin ve gerçek değerlerin ters ölçeklendirilmesi
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(df[['close']])
predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(real_prices, predicted_prices)
rmse = sqrt(mse)
accuracy = max(0, 100 - (rmse / np.mean(real_prices[:, 0]) * 100))

print(f'Test MSE: {mse}')
print(f'Test RMSE: {rmse}')
print("Accuracy" + str(accuracy))


# plt.figure(figsize=(10,6))
# plt.plot(real_prices, label='Actual Prices')
# plt.plot(predicted_prices, label='Predicted Prices')
# plt.title('Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

future_steps = 30

# Son pencereyi kullanarak gelecek fiyatları tahmin etme
last_window = scaled_df[['close']].head(window_size).values.reshape(1, window_size, 1)
future_prices = []

for _ in range(future_steps):
    predicted_price = model.predict(last_window)
    future_prices.append(predicted_price[0, 0])
    last_window = np.append(last_window[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

# Tahmin edilen fiyatları ters ölçeklendirme
predicted_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

# Gelecek tarihleri oluşturma
last_date = scaled_df.index[0]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')

# Tahmin edilen fiyatları grafik üzerinde gösterme
plt.figure(figsize=(12, 6))
plt.plot(future_dates, predicted_prices, label='Predicted Prices')
plt.title('Future Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



