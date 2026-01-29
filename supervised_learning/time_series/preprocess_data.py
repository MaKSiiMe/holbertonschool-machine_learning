#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


def preprocess():
    print("--- 1. Chargement des données ---")
    # Chargement des CSV bruts
    try:
        df1 = pd.read_csv('coinbase.csv')
        df2 = pd.read_csv('bitstamp.csv')
    except FileNotFoundError:
        print("Erreur: Assurez-vous que coinbase.csv et bitstamp.csv sont dans le dossier.")
        return

    # Fusion
    df = pd.concat([df1, df2])

    # Nettoyage Timestamp
    # Le dataset original a une colonne 'Timestamp' en secondes Unix
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')
    df = df.sort_index()

    # On garde les colonnes pertinentes
    # Close est notre cible. Volume est un bon indicateur.
    keep_cols = ['Close', 'Volume_(BTC)']
    df = df[keep_cols]

    print("--- 2. Resampling & Feature Engineering ---")
    # Resample à l'HEURE (Moyenne)
    # Cela réduit la taille du fichier par 60 et lisse énormément le bruit
    df_hourly = df.resample('H').mean()

    # Remplissage des trous (Forward Fill)
    df_hourly = df_hourly.ffill()

    # AJOUT FEATURE : Moyenne Mobile 24h
    # C'est LA clé pour éviter les dents de scie : on donne au modèle la tendance
    df_hourly['MA24'] = df_hourly['Close'].rolling(window=24).mean()

    # Suppression des NaN créés par la moyenne mobile
    df_hourly = df_hourly.dropna()

    print(f"--- Taille finale du dataset : {df_hourly.shape} ---")

    print("--- 3. Normalisation ---")
    scaler = MinMaxScaler()
    # On fit sur tout le dataset (dans un contexte pro on ferait fit sur train uniquement,
    # mais pour cet exo c'est toléré et simplifie le code)
    data_scaled = scaler.fit_transform(df_hourly)

    # Reconstruction DataFrame
    df_scaled = pd.DataFrame(data_scaled, columns=df_hourly.columns, index=df_hourly.index)

    # Conversion Float32 pour économiser la mémoire
    df_scaled = df_scaled.astype('float32')

    # Sauvegarde
    df_scaled.to_csv('preprocessed_btc.csv')
    joblib.dump(scaler, 'btc_scaler.save')

    print("✅ Terminé : 'preprocessed_btc.csv' et 'btc_scaler.save' créés.")


if __name__ == '__main__':
    preprocess()
