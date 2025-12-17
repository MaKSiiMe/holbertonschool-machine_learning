#!/usr/bin/env python3
"""Bayesian Optimization for Neural Network Hyperparameter Tuning"""

import numpy as np
from tensorflow import keras as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

BayesianOptimization = __import__('5-bayes_opt').BayesianOptimization

_data_cache = {}
_hp_space = {
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'units': [16, 32, 64, 128],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'epochs': [10, 15, 20, 25],
    'batch_size': [16, 32, 64]
}


def load_data():
    """Load and cache breast cancer dataset."""
    if 'loaded' not in _data_cache:
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        _data_cache.update({
            'X_train': X_train, 'X_val': X_val,
            'y_train': y_train, 'y_val': y_val, 'loaded': True
        })
    return tuple(_data_cache[k] for k in ['X_train', 'X_val', 'y_train', 'y_val'])


def decode_hyperparameters(x):
    """Decode normalized value to hyperparameters."""
    idx = int(x * 9999)
    params = {}
    for key, vals in _hp_space.items():
        params[key] = vals[idx % len(vals)]
        idx //= len(vals)
    return params


def create_model(learning_rate, units, dropout):
    """Build the neural network model."""
    model = K.Sequential([
        K.layers.Flatten(input_shape=(30,)),
        K.layers.Dense(units, activation='relu'),
        K.layers.Dropout(dropout),
        K.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=K.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model


def objective_function(x):
    """Objective function for Bayesian optimization."""
    X_train, X_val, y_train, y_val = load_data()
    params = decode_hyperparameters(x)
    
    model = create_model(params['learning_rate'], params['units'], params['dropout'])
    
    checkpoint_name = "_".join([f"{k}{v}" for k, v in params.items()])
    checkpoint_path = f"checkpoints/model_{checkpoint_name}.h5"
    
    K.backend.clear_session()
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=0
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=callbacks,
        verbose=0
    )
    
    val_loss = min(history.history['val_loss'])
    val_accuracy = max(history.history['val_accuracy'])
    
    if not hasattr(objective_function, 'best_loss') or val_loss < objective_function.best_loss:
        objective_function.best_loss = val_loss
        objective_function.best_params = params
        objective_function.best_accuracy = val_accuracy
    
    with open("bayes_opt.txt", "a") as f:
        f.write(f"lr={params['learning_rate']:.5f}, units={params['units']}, "
                f"dropout={params['dropout']:.2f}, epochs={params['epochs']}, "
                f"batch={params['batch_size']}, val_loss={val_loss:.4f}, "
                f"val_acc={val_accuracy:.4f}\n")
    
    return val_loss


def run_bayesian_optimization():
    """Run Bayesian optimization."""
    np.random.seed(42)
    
    with open("bayes_opt.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BAYESIAN OPTIMIZATION FOR NEURAL NETWORK\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: Breast Cancer (binary classification)\n")
        f.write(f"Model: Sequential Neural Network\n")
        f.write(f"Optimization Metric: Validation Loss (minimized)\n\n")
        f.write("-" * 80 + "\n")
        f.write("HYPERPARAMETER SEARCH SPACE\n")
        f.write("-" * 80 + "\n")
        for k, v in _hp_space.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n" + "-" * 80 + "\n")
        f.write("ITERATION RESULTS\n")
        f.write("-" * 80 + "\n")
    
    X_init = np.random.uniform(0, 1, (3, 1))
    Y_init = np.array([objective_function(x) for x in X_init]).reshape(-1, 1)
    
    bo = BayesianOptimization(
        f=objective_function,
        X_init=X_init,
        Y_init=Y_init,
        bounds=(0, 1),
        ac_samples=50,
        l=0.1,
        sigma_f=1.0,
        xsi=0.01,
        minimize=True
    )
    
    iteration_history = []
    
    for iteration in range(30):
        X_next, _ = bo.acquisition()
        if np.any(np.isclose(bo.gp.X, X_next)):
            break
        
        Y_next = bo.f(X_next)
        bo.gp.update(X_next, Y_next)
        
        iteration_history.append({
            'iteration': iteration + 1,
            'loss': Y_next if np.isscalar(Y_next) else Y_next[0],
            'best_loss': np.min(bo.gp.Y)
        })
    
    best_idx = np.argmin(bo.gp.Y)
    best_params = decode_hyperparameters(bo.gp.X[best_idx][0])
    best_loss = bo.gp.Y[best_idx][0]
    
    plot_convergence(iteration_history)
    save_final_report(bo, best_params, best_loss)
    
    return bo, best_params, best_loss


def plot_convergence(history):
    """Plot optimization convergence."""
    data = {k: [h[k] for h in history] for k in ['iteration', 'loss', 'best_loss']}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(data['iteration'], data['loss'], 'b-o', label='Current', markersize=6)
    ax1.plot(data['iteration'], data['best_loss'], 'r-s', label='Best so far', markersize=6)
    ax1.set(xlabel='Iteration', ylabel='Validation Loss', 
            title='Bayesian Optimization Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(data['iteration'], data['best_loss'], 'g-', linewidth=2)
    ax2.fill_between(data['iteration'], data['best_loss'], alpha=0.3)
    ax2.set(xlabel='Iteration', ylabel='Best Loss Found', 
            title='Cumulative Best Performance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_plot.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_final_report(bo, best_params, best_loss):
    """Save final optimization report."""
    with open("bayes_opt.txt", "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("OPTIMIZATION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Evaluations: {len(bo.gp.X)}\n")
        f.write(f"Best Validation Loss: {best_loss:.6f}\n")
        if hasattr(objective_function, 'best_accuracy'):
            f.write(f"Best Validation Accuracy: {objective_function.best_accuracy:.4f}\n")
        f.write("\n" + "-" * 80 + "\n")
        f.write("BEST HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSIONS\n")
        f.write("=" * 80 + "\n")
        f.write("Bayesian optimization successfully explored the hyperparameter space\n")
        f.write("using Gaussian Processes and Expected Improvement acquisition.\n")
        f.write("The model used early stopping and saved checkpoints of best iterations.\n")
        f.write("Convergence plot shows the balance between exploration and exploitation.\n")
        f.write("=" * 80 + "\n")


if __name__ == '__main__':
    run_bayesian_optimization()
