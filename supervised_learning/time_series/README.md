**Time Series - BTC Forecasting**

Ce dossier contient des scripts pour prétraiter des CSV historiques de BTC,
entraîner un RNN/LSTM et faire des prédictions avec visualisation.

Fichiers principaux
- `preprocess_data.py`: resample minute -> hourly, ajoute une feature `Close_MA24`, normalise et sauvegarde le scaler.
- `forecast_btc.py`: pipeline d'entraînement, prédiction et plot. Supporte `--predict-only` pour charger un modèle existant.

Usage rapide

1) Prétraiter les CSV bruts (ex: coinbase.csv, bitstamp.csv) :

```bash
python3 preprocess_data.py coinbase.csv bitstamp.csv --out-csv preprocessed_btc.csv --out-scaler btc_scaler.save
```

2) Entraîner le modèle (run contrôlé) :

```bash
python3 forecast_btc.py --csv preprocessed_btc.csv --epochs 50 --max-steps-per-epoch 150 --patience 5 --plot-after
```

3) Utiliser un modèle sauvegardé pour prédire et tracer :

```bash
python3 forecast_btc.py --csv preprocessed_btc.csv --predict-only --model-out btc_rnn_model.h5 --plot-after
```

Notes
- Le scaler est sauvegardé avec `joblib` par `preprocess_data.py`. `forecast_btc.py`
  cherche `btc_scaler.save` par défaut; vous pouvez fournir `--scaler` si nécessaire.
- Les prédictions sont inverse-transformées proprement via `scaler.inverse_transform`.
# Forecasting Bitcoin (BTC) with Deep Learning: My Time Series Journey

![BTC Forecasting](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*eQwQ6l04e6QF6UVxZ4WR8A.png)

## Introduction: Time Series Forecasting and Bitcoin

Time series forecasting is a powerful tool for predicting future values based on previously observed data points. In the world of finance, and especially with volatile assets like Bitcoin (BTC), being able to forecast prices can be both intellectually rewarding and potentially lucrative. In this project, I set out to use deep learning—specifically Recurrent Neural Networks (RNNs)—to predict the price of BTC one hour ahead, using the previous 24 hours of minute-level data.

---

## Data Preprocessing: Cleaning and Structuring for Deep Learning

### Why Preprocessing Matters

Raw financial data is noisy, contains missing values, and often includes features that are not directly useful for prediction. Proper preprocessing ensures that the model receives clean, normalized, and relevant data, which is crucial for effective learning.

### My Approach

- **Feature Selection:** I kept only the most relevant columns: Open, High, Low, Close, Volume (BTC), Volume (Currency), and Weighted Price.
- **Cleaning:** I removed any rows containing NaN or infinite values to avoid training instabilities.
- **Normalization:** I used MinMaxScaler to scale all features between 0 and 1, which helps neural networks converge faster.
- **Sequencing:** For each prediction, I created a sequence of 24 hours (1440 minutes) of data as input, with the target being the close price one hour (60 minutes) after the end of the sequence.

```python
# Example: Creating sequences for supervised learning
for i in range(len(data) - seq_len - pred_offset + 1):
    X = data[i:i+seq_len]
    y = data[i+seq_len+pred_offset-1][3]  # 'Close' price
```

---

## Feeding Data to the Model: tf.data.Dataset

TensorFlow's `tf.data.Dataset` API is ideal for efficiently feeding large datasets to a model, especially when working with sequences.

- **Batching:** I used small batch sizes to avoid memory issues.
- **Shuffling:** Training data is shuffled for better generalization.
- **Prefetching:** Data is prefetched to keep the GPU busy.

```python
ds = tf.data.Dataset.from_tensor_slices((X, y))
ds = ds.shuffle(buffer_size=min(10000, len(X)))
ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

---

## Model Architecture: Deep LSTM for Time Series

I chose a simple but effective architecture:

- **Stacked LSTM Layers:** Two LSTM layers (64 and 32 units) to capture temporal dependencies.
- **Dense Output:** A single neuron to predict the normalized close price.
- **Loss Function:** Mean Squared Error (MSE), standard for regression tasks.

```python
model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

---

## Results: Performance and Visualization

After training for 10 epochs on a subset of the data (due to hardware constraints), the model achieved:

- **Test MSE:** ~0.0000
- **Test MAE:** ~0.0037

Below is a sample plot comparing predicted vs. actual close prices on the test set:

![BTC Prediction Example](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*eQwQ6l04e6QF6UVxZ4WR8A.png)

*(Replace with your own matplotlib plot if possible)*

---

## Conclusion: Lessons Learned and Next Steps

Forecasting BTC is challenging due to its volatility and the influence of external factors. However, deep learning models like LSTMs can capture temporal patterns and provide reasonable short-term forecasts. For production use, more data, feature engineering, and model tuning would be necessary.

**Experience:**  
- Data cleaning is critical—NaNs or outliers can break your model.
- GPU acceleration and efficient data pipelines (tf.data) are essential for deep learning on time series.
- Even with a simple architecture, you can get meaningful results if your preprocessing is solid.

**See the full code and details on my GitHub:**  
[GitHub Repo](https://github.com/yourusername/holbertonschool-machine_learning/tree/main/supervised_learning/time_series)

---

## Blog and Social Links

- **Blog post:** [Medium - Forecasting Bitcoin with Deep Learning](https://medium.com/@yourusername/forecasting-bitcoin-with-deep-learning-XXXXXXXXXXXX)
- **Shared on LinkedIn:** [LinkedIn Post](https://www.linkedin.com/in/yourusername/detail/recent-activity/shares/)

---

*Replace the image URLs and links with your own plots and published blog URLs after posting!*
