# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 2025

@author: OMISTAJA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle

# Asetetaan satunnaislukugeneraattorin siemen toistettavuuden takaamiseksi
np.random.seed(42)

def train_and_save_model():
    """
    Lataa koripallodatan, kouluttaa 5 muuttujan Random Forest -mallin,
    evaluoi sen ja tallentaa sen tiedostoon
    """
    """
    Lataa koripallodatan, kouluttaa 5 muuttujan Random Forest -mallin,
    evaluoi sen ja tallentaa sen tiedostoon
    """
    print("Ladataan koripallodataa 5 muuttujalla...")
    
    # Ladataan data - huomioi, että erottimena on puolipiste
    # ja desimaalierotin on pilkku (eurooppalainen formaatti)
    data = pd.read_csv('siivottuja_5muuttujaa_ryhmat_kaikki.csv', sep=';', decimal=',')
    
    # Tarkistetaan datan muoto
    print(f"Datan dimensiot: {data.shape}")
    print("\nDatan ensimmäiset rivit:")
    print(data.head())

    # Tarkistetaan onko puuttuvia arvoja
    print("\nPuuttuvat arvot:")
    print(data.isnull().sum())

    # Tutkitaan ennustettavaa muuttujaa (XP)
    print("\nXP-sarakkeen tilastot:")
    print(data['XP'].describe())
    
    # Visualisoidaan ennustettavan muuttujan jakauma
    plt.figure(figsize=(10, 6))
    sns.histplot(data['XP'], bins=20, kde=True)
    plt.title('Pistepotentiaalin (XP) jakauma')
    plt.xlabel('Pistepotentiaali (XP)')
    plt.ylabel('Frekvenssi')
    plt.savefig('xp_jakauma_5muuttujaa.png')
    
    # Käsitellään kategoriset muuttujat - kaikki paitsi XP
    X = data.drop('XP', axis=1)
    y = data['XP']
    
    # Muokataan kaikki muuttujat kategorisiksi
    X_categorical = X.astype('category')
    
    # One-hot koodaus kategorisille muuttujille
    X_encoded = pd.get_dummies(X_categorical)
    
    # Jaetaan data opetus- ja testiosiin (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    print(f"\nOpetusaineiston koko: {X_train.shape}")
    print(f"Testiaineiston koko: {X_test.shape}")
    
    # Määritetään Random Forest -mallin parametrit
    params = {
        'n_estimators': 100,            # Puiden määrä metsässä
        'max_depth': 5,                 # Puun maksimisyvyys
        'min_samples_leaf': 1,          # Solmun minimikokovaatimus
        'random_state': 42              # Toistettavuus
    }
    
    # Koulutetaan Random Forest -malli
    print("\nKoulutetaan Random Forest -mallia viidellä muuttujalla...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Cross-validointi mallin robustisuuden arvioimiseksi
    print("\nSuoritetaan 5-fold ristiinvalidointi...")
    cv_scores = cross_val_score(model, X_encoded, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    
    # Arvioidaan malli testidatalla
    print("\nArvioidaan mallin suorituskykyä testijoukolla...")
    y_pred = model.predict(X_test)
    
    # Lasketaan arviointi-metriikat
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Visualisoidaan todelliset vs. ennustetut arvot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Todelliset arvot')
    plt.ylabel('Ennustetut arvot')
    plt.title('Todelliset vs. Ennustetut XP-arvot (5 muuttujaa)')
    plt.savefig('ennusteet_vs_todelliset_5muuttujaa.png')
    
    # Analysoidaan ominaisuuksien tärkeyttä
    print("\nOminaisuuksien tärkeysjärjestys:")
    feature_importances = model.feature_importances_
    
    # Luodaan DataFrame ominaisuuksien tärkeyksille
    importance_df = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': feature_importances
    })
    
    # Järjestetään tärkeimmät ominaisuudet
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Näytetään top 20 ominaisuutta
    print("\nTop 20 ominaisuutta tärkeysjärjestyksessä:")
    print(importance_df.head(20))
    
    # Muunnetaan one-hot koodatut ominaisuudet takaisin alkuperäisiksi ja aggregoidaan arvot
    original_importances = {}
    for feature, importance_value in zip(X_encoded.columns, feature_importances):
        # Erotellaan alkuperäinen sarake one-hot koodatusta nimestä (esim. 'Shot_3' -> 'Shot')
        original_column = feature.split('_')[0]
        if original_column in original_importances:
            original_importances[original_column] += importance_value
        else:
            original_importances[original_column] = importance_value
    
    # Järjestetään tärkeimmät alkuperäiset ominaisuudet
    sorted_original_importances = sorted(original_importances.items(), key=lambda x: x[1], reverse=True)
    
    print("\nAggregoidut alkuperäiset ominaisuudet tärkeysjärjestyksessä:")
    for feature, importance_value in sorted_original_importances:
        print(f"{feature}: {importance_value:.4f}")
    
    # Tallennetaan malli
    print("\nTallennetaan malli...")
    pickle.dump(model, open("koripallo_supistettu_rf_malli.pkl", "wb"))
    print("Malli tallennettu nimellä: koripallo_supistettu_rf_malli.pkl")
    
    # Tallennetaan feature names tulevaa käyttöä varten
    feature_names = list(X_encoded.columns)
    pickle.dump(feature_names, open("feature_names_5muuttujaa.pkl", "wb"))
    print("Feature names tallennettu nimellä: feature_names_5muuttujaa.pkl")
    
    return model, X_encoded.columns, y

if __name__ == "__main__":
    train_and_save_model()