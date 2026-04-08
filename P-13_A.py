# LSTM for Univariate Time Series Prediction (Python)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic sine wave + noise
np.random.seed(1)
t = np.arange(0, 100, 0.1)
data = np.sin(t) + 0.1 * np.random.randn(len(t))

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1))

# Prepare sequences
def create_sequences(data, step=10):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i+step, 0])
        y.append(data[i+step, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(data, step=10)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(10, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=20, batch_size=32, verbose=0)

# Predict next value
predicted = model.predict(X[-1].reshape(1,10,1))
predicted_value = scaler.inverse_transform(predicted)

print("Predicted next value:", predicted_value[0][0])
