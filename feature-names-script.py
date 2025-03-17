# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:29:23 2025

@author: OMISTAJA
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 2025

@author: OMISTAJA
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Tämä skripti tallentaa mallin koulutusominaisuudet (feature names)
# erillistä käyttöliittymää varten, jotta vältytään feature mismatch -ongelmalta

def save_model_with_features():
    """
    Lataa alkuperäinen data, kouluttaa Random Forest -mallin ja tallentaa sekä
    mallin että sen käyttämät sarakkeiden nimet (feature names).
    """
    print("Ladataan koripallodataa (5 muuttujaa)...")
    
    # Ladataan data - huomioi, että erottimena on puolipiste
    # ja desimaalierotin on pilkku (eurooppalainen formaatti)
    data = pd.read_csv('siivottuja_5muuttujaa_ryhmat_kaikki.csv', sep=';', decimal=',')
    
    # Tarkistetaan data
    print(f"Datan dimensiot: {data.shape}")
    print("\nDatan ensimmäiset rivit:")
    print(data.head())
    
    # Käsitellään kategoriset muuttujat - kaikki paitsi XP
    X = data.drop('XP', axis=1)
    y = data['XP']
    
    # Muunnetaan kaikki muuttujat kategorisiksi ja tehdään one-hot koodaus
    X_categorical = X.astype('category')
    X_dummies = pd.get_dummies(X_categorical)
    
    # Tallennetaan sarakkeiden nimet (feature names)
    feature_names = list(X_dummies.columns)
    print(f"Sarakkeiden määrä one-hot koodauksen jälkeen: {len(feature_names)}")
    print("Sarakkeet (20 ensimmäistä):")
    for i, col in enumerate(feature_names[:20]):
        print(f"  {i+1}. {col}")
    
    # Koulutetaan malli
    print("\nKoulutetaan Random Forest -mallia...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_dummies, y)
    
    # Tallennetaan malli
    pickle.dump(model, open("koripallo_supistettu_rf_malli.pkl", "wb"))
    print("Malli tallennettu tiedostoon 'koripallo_supistettu_rf_malli.pkl'")
    
    # Tallennetaan sarakkeiden nimet
    pickle.dump(feature_names, open("feature_names_5muuttujaa.pkl", "wb"))
    print("Sarakkeiden nimet tallennettu tiedostoon 'feature_names_5muuttujaa.pkl'")
    
    print("\nValmis! Nyt voit käyttää 'koripallo_supistettu_rf_malli.pkl' ja 'feature_names_5muuttujaa.pkl' -tiedostoja käyttöliittymässä.")
    
    return model, feature_names
    
if __name__ == "__main__":
    save_model_with_features()