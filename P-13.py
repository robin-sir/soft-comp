# LSTM for Next-Step Prediction (Univariate Time Series)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Generate sine wave + noise
# -----------------------------
np.random.seed(1)
t = np.arange(0, 200, 0.1)
series = np.sin(t) + 0.1 * np.random.randn(len(t))

# -----------------------------
# Prepare data
# -----------------------------
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1,1))

X, y = [], []
time_steps = 10
for i in range(len(series_scaled) - time_steps):
    X.append(series_scaled[i:i+time_steps])
    y.append(series_scaled[i+time_steps])

X = np.array(X)
y = np.array(y)

# -----------------------------
# Build LSTM model
# -----------------------------
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(time_steps, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# -----------------------------
# Train model
# -----------------------------
history = model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# -----------------------------
# Predict next value
# -----------------------------
last_sequence = series_scaled[-time_steps:].reshape(1, time_steps, 1)
next_value_scaled = model.predict(last_sequence)
next_value = scaler.inverse_transform(next_value_scaled)

print("Predicted next value:", next_value[0][0])

# -----------------------------
# Plot result
# -----------------------------
plt.plot(series, label='Original Series')
plt.scatter(len(series), next_value[0][0], color='red', label='Predicted Next')
plt.legend()
plt.show()
# Build and train an LSTM to predict the next value in a univariate time series (e.g., daily sine wave + noise or stock-like synthetic series).

