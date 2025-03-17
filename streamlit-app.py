# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:31:09 2025

@author: OMISTAJA
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os

# Aseta sivun otsikko
st.set_page_config(page_title="Koripallo Random Forest Analysaattori", layout="wide")
st.title("Koripallo Random Forest Analysaattori")

# Alusta istuntomuuttujat, jos niitä ei ole vielä
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.feature_names = None
    st.session_state.original_data = None
    st.session_state.training_columns = ['Set', 'Line up', 'Shot', 'Shot clock', 'Player']  # Viisi muuttujaa
    st.session_state.unique_values = {}
    st.session_state.columns = ['Set', 'Line up', 'Shot', 'Shot clock', 'Player']  # Viisi muuttujaa

# Lataa malli ja data, jos niitä ei ole vielä ladattu
@st.cache_resource
def load_model_and_data():
    model = None
    feature_names = None
    original_data = None
    training_columns = ['Set', 'Line up', 'Shot', 'Shot clock', 'Player']  # Viisi muuttujaa
    unique_values = {}
    columns = ['Set', 'Line up', 'Shot', 'Shot clock', 'Player']  # Viisi muuttujaa
    
    # Yritä ladata malli
    try:
        model = pickle.load(open("koripallo_supistettu_rf_malli.pkl", "rb"))
        st.success("Malli ladattu onnistuneesti!")
    except:
        st.error("Mallia ei löytynyt. Lataa malli ennen ennusteiden tekemistä.")
        # Jos haluat automaattisesti kouluttaa mallin, voit lisätä koodin tähän
    
    # Lataa alkuperäinen data
    try:
        original_data = pd.read_csv('siivottuja_5muuttujaa_ryhmat_kaikki.csv', sep=';', decimal=',')
        
        # Kerää jokaisen sarakkeen uniikit arvot
        for col in columns:
            unique_values[col] = sorted(original_data[col].unique().astype(str))
        
        # Muunnetaan data kategorisiksi ja tehdään one-hot koodaus
        X_categorical = original_data.drop('XP', axis=1).astype('category')
        X_dummies = pd.get_dummies(X_categorical)
        feature_names = list(X_dummies.columns)
        
        st.success("Data ladattu onnistuneesti!")
        
    except Exception as e:
        st.error(f"Virhe datan latauksessa: {str(e)}")
        # Määritellään esimerkkiarvot, jos dataa ei ole saatavilla
        unique_values = {
            'Set': ["1", "2", "3", "4", "5"],
            'Line up': ["1", "2", "3", "4", "5"],
            'Shot': ["1", "2", "3"],
            'Shot clock': ["1", "2", "3", "4"],
            'Player': ["1", "2", "3", "4", "5"]
        }
        st.warning("Alkuperäistä dataa ei löytynyt. Käytetään esimerkkiarvoja.")
    
    return model, feature_names, original_data, training_columns, unique_values, columns

# Lataa malli ja data
model, feature_names, original_data, training_columns, unique_values, columns = load_model_and_data()

# Päivitä istuntomuuttujat
st.session_state.model = model
st.session_state.feature_names = feature_names
st.session_state.original_data = original_data
st.session_state.training_columns = training_columns
st.session_state.unique_values = unique_values
st.session_state.columns = columns

# Funktio syötteen valmisteluun ennustusta varten
def prepare_input_for_prediction(selected_data, feature_names, model):
    # Luo DataFrame valituista tiedoista
    input_df = pd.DataFrame([selected_data])
    
    # Muunna kategorisiksi
    input_categorical = input_df.astype('category')
    
    # Tee one-hot koodaus
    input_dummies = pd.get_dummies(input_categorical)
    
    # Tarkista, että kaikki koulutusdatan sarakkeet ovat käytössä
    if hasattr(model, 'feature_names_in_'):
        # RandomForest mallin tapauksessa käytä feature_names_in_
        model_features = model.feature_names_in_
        
        # Luo uusi DataFrame, joka sisältää kaikki koulutusdatan sarakkeet
        final_df = pd.DataFrame(0, index=input_dummies.index, columns=model_features)
        
        # Kopioi olemassa olevat sarakkeet
        for col in input_dummies.columns:
            if col in model_features:
                final_df[col] = input_dummies[col]
        
        # Täytä puuttuvat sarakkeet nollilla
        missing_cols = set(model_features) - set(input_dummies.columns)
        for col in missing_cols:
            final_df[col] = 0
            
        return final_df
    
    # Jos feature_names_in_ ei ole saatavilla, käytä feature_names parametria
    elif feature_names is not None:
        # Luo uusi DataFrame, joka sisältää kaikki koulutusdatan sarakkeet
        final_df = pd.DataFrame(0, index=input_dummies.index, columns=feature_names)
        
        # Kopioi olemassa olevat sarakkeet
        for col in input_dummies.columns:
            if col in feature_names:
                final_df[col] = input_dummies[col]
            else:
                st.warning(f"Varoitus: Saraketta {col} ei löydy koulutusdatasta.")
        
        return final_df
    else:
        # Jos ei ole tietoa koulutusdatan sarakkeista, käytä suoraan syötettä
        return input_dummies

# Välilehdet
tab1, tab2, tab3 = st.tabs(["XP Ennuste", "Paras Set", "Muuttujien Tärkeys"])

# Välilehti 1: XP Ennuste
with tab1:
    st.header("Valitse muuttujat ja niiden arvot XP-ennustetta varten")
    
    # Käytä kolumneja asetteluun
    col1, col2 = st.columns(2)
    
    # Luo valintalaatikot ja pudotusvalikot
    selected_data = {}
    
    # Jaa sarakkeet kahdelle puolelle
    mid_point = len(st.session_state.columns) // 2
    
    with col1:
        for i, col in enumerate(st.session_state.columns[:mid_point]):
            st.subheader(col)
            if st.session_state.unique_values.get(col):
                value = st.selectbox(f"Valitse {col}", st.session_state.unique_values[col], key=f"select_{col}")
                selected_data[col] = int(value) if value.isdigit() else value
    
    with col2:
        for i, col in enumerate(st.session_state.columns[mid_point:]):
            st.subheader(col)
            if st.session_state.unique_values.get(col):
                value = st.selectbox(f"Valitse {col}", st.session_state.unique_values[col], key=f"select_{col}")
                selected_data[col] = int(value) if value.isdigit() else value
    
    # Ennustuspainike
    if st.button("Ennusta XP", key="predict_button"):
        if st.session_state.model is None:
            st.error("Mallia ei ole ladattu. Lataa malli ensin.")
        else:
            # Varmista, että kaikki vaaditut sarakkeet on valittu
            missing_columns = [col for col in st.session_state.training_columns if col not in selected_data]
            if missing_columns:
                st.error(f"Valitse arvot seuraaville muuttujille: {', '.join(missing_columns)}")
            else:
                # Valmistele syöte ennustusta varten
                input_data = prepare_input_for_prediction(selected_data, st.session_state.feature_names, st.session_state.model)
                
                # Tee ennuste
                try:
                    prediction = st.session_state.model.predict(input_data)[0]
                    
                    # Näytä tulos
                    st.success(f"Ennustettu XP: {prediction:.3f}")
                    
                    # Visualisoi tulos
                    if st.session_state.original_data is not None:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        
                        # Käyttäjän valinta ja keskiarvo
                        comparison_data = {
                            'Kategoria': ['Käyttäjän valinta', 'Aineiston keskiarvo'],
                            'XP': [prediction, st.session_state.original_data['XP'].mean()]
                        }
                        
                        # Luodaan barplot
                        bars = ax.bar(comparison_data['Kategoria'], comparison_data['XP'], color=['lightgreen', 'lightgray'])
                        
                        # Lisää tekstit palkkien päälle
                        for bar, xp in zip(bars, comparison_data['XP']):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{xp:.3f}',
                                   ha='center', va='bottom')
                        
                        ax.set_ylabel('Ennustettu XP')
                        ax.set_title('Ennuste verrattuna aineiston keskiarvoon')
                        
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Virhe ennustetta tehdessä: {str(e)}")
                    st.error(f"Tarkempi virhetieto: {str(e)}")

# Välilehti 2: Paras Set
with tab2:
    st.header("Etsi paras Set muiden muuttujien perusteella")
    
    # Käytä kolumneja asetteluun
    col1, col2 = st.columns(2)
    
    # Luo valintalaatikot ja pudotusvalikot, paitsi Set
    selected_data_set = {}
    
    # Jaa sarakkeet kahdelle puolelle
    non_set_columns = [col for col in st.session_state.columns if col != "Set"]
    mid_point = len(non_set_columns) // 2
    
    with col1:
        for i, col in enumerate(non_set_columns[:mid_point]):
            st.subheader(col)
            if st.session_state.unique_values.get(col):
                value = st.selectbox(f"Valitse {col}", st.session_state.unique_values[col], key=f"select_set_{col}")
                selected_data_set[col] = int(value) if value.isdigit() else value
    
    with col2:
        for i, col in enumerate(non_set_columns[mid_point:]):
            st.subheader(col)
            if st.session_state.unique_values.get(col):
                value = st.selectbox(f"Valitse {col}", st.session_state.unique_values[col], key=f"select_set_{col}")
                selected_data_set[col] = int(value) if value.isdigit() else value
    
    # Etsintäpainike
    if st.button("Etsi paras Set", key="find_best_set"):
        if st.session_state.model is None:
            st.error("Mallia ei ole ladattu. Lataa malli ensin.")
        else:
            # Varmista, että kaikki vaaditut sarakkeet on valittu
            missing_columns = [col for col in non_set_columns if col not in selected_data_set]
            if missing_columns:
                st.error(f"Valitse arvot seuraaville muuttujille: {', '.join(missing_columns)}")
            else:
                # Löydä kaikki mahdolliset Set-arvot
                set_values = st.session_state.unique_values.get("Set", [])
                
                if not set_values:
                    st.error("Set-arvoja ei löytynyt.")
                else:
                    # Tee ennusteet jokaiselle Set-arvolle
                    results = []
                    
                    with st.spinner("Lasketaan parasta Settiä..."):
                        for set_val in set_values:
                            # Luo kopio valituista tiedoista ja lisää nykyinen Set-arvo
                            current_data = selected_data_set.copy()
                            current_data["Set"] = int(set_val) if set_val.isdigit() else set_val
                            
                            # Valmistele syöte ennustusta varten
                            input_data = prepare_input_for_prediction(current_data, st.session_state.feature_names, st.session_state.model)
                            
                            # Tee ennuste
                            try:
                                prediction = st.session_state.model.predict(input_data)[0]
                                results.append((set_val, prediction))
                            except Exception as e:
                                st.error(f"Virhe ennustetta tehdessä Set-arvolle {set_val}: {str(e)}")
                                break
                    
                    # Jos tuloksia löytyi, visualisoi ne
                    if results:
                        # Järjestä tulokset XP-ennusteen mukaan
                        results.sort(key=lambda x: x[1], reverse=True)
                        
                        # Erottele data
                        set_values_plot = [str(r[0]) for r in results]
                        xp_values = [r[1] for r in results]
                        
                        # Näytä paras tulos
                        best_set, best_xp = results[0]
                        st.success(f"Paras Set: {best_set} (XP: {best_xp:.3f})")
                        
                        # Luo pylväsdiagrammi
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(set_values_plot, xp_values, color='skyblue')
                        
                        # Korosta paras tulos
                        best_idx = xp_values.index(max(xp_values))
                        bars[best_idx].set_color('green')
                        
                        # Lisää arvot palkkien päälle
                        for bar, xp in zip(bars, xp_values):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                   f"{xp:.3f}", ha='center', va='bottom')
                        
                        # Muotoile kuvaaja
                        ax.set_xlabel('Set')
                        ax.set_ylabel('Ennustettu XP')
                        ax.set_title('Eri Set-arvojen ennustetut XP-arvot')
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                        
                        # Näytä kuvaaja
                        st.pyplot(fig)

# Välilehti 3: Muuttujien tärkeys
with tab3:
    st.header("Muuttujien tärkeys mallin ennusteessa")
    
    if st.session_state.model is None:
        st.error("Mallia ei ole ladattu. Lataa malli ensin.")
    else:
        # Hae muuttujien tärkeys mallista, jos se on RandomForestRegressor
        if isinstance(st.session_state.model, RandomForestRegressor):
            # Hae feature importances
            importances = st.session_state.model.feature_importances_
            
            if hasattr(st.session_state.model, 'feature_names_in_'):
                feature_names = st.session_state.model.feature_names_in_
            else:
                feature_names = [f"Feature {i}" for i in range(len(importances))]
            
            # Luo DataFrame muuttujien tärkeydelle
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # Yhdistä one-hot -koodatut ominaisuudet takaisin alkuperäisiin sarakkeisiin
            original_importances = {}
            for feature, importance in zip(feature_names, importances):
                # Esim. "Set_1" -> "Set"
                parts = feature.split('_')
                if len(parts) > 1:
                    original_col = parts[0]
                else:
                    original_col = feature
                
                if original_col in original_importances:
                    original_importances[original_col] += importance
                else:
                    original_importances[original_col] = importance
            
            # Luo DataFrame ja järjestä tärkeyden mukaan
            original_importance_df = pd.DataFrame({
                'ominaisuus': list(original_importances.keys()),
                'tärkeys': list(original_importances.values())
            }).sort_values('tärkeys', ascending=False)
            
            # Näytä DataFrame
            st.subheader("Kokonaistärkeys alkuperäisille ominaisuuksille")
            st.dataframe(original_importance_df)
            
            # Visualisoi alkuperäisten ominaisuuksien tärkeys
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(original_importance_df['ominaisuus'], 
                         original_importance_df['tärkeys'], 
                         color='forestgreen')
            
            # Lisää arvot palkkien viereen
            for i, (importance, feature) in enumerate(zip(original_importance_df['tärkeys'], 
                                                      original_importance_df['ominaisuus'])):
                ax.text(importance + 0.01, i, f"{importance:.4f}", va='center')
            
            ax.set_xlabel('Tärkeys (Gini-indeksi)')
            ax.set_title('Ominaisuuksien tärkeys mallissa')
            ax.invert_yaxis()  # Käännä y-akseli, jotta tärkein on ylimpänä
            
            # Näytä kuvaaja
            st.pyplot(fig)
        else:
            st.warning("Malli ei ole Random Forest -tyyppiä, joten muuttujien tärkeyttä ei voida näyttää.")

# Tietoja sovelluksesta
with st.expander("Tietoja sovelluksesta"):
    st.write("""
    ## Koripallo Random Forest Analysaattori
    
    Tämä sovellus käyttää Random Forest -koneoppimismallia koripallon pelianalyysiin. 
    Malli on koulutettu käyttämään viittä tärkeintä ominaisuutta: Set, Line up, Shot, Shot clock ja Player.
    
    Sovellus mahdollistaa:
    - XP-arvojen ennustamisen valituilla muuttujien arvoilla
    - Parhaan Set-vaihtoehdon löytämisen muiden muuttujien perusteella
    - Muuttujien tärkeyden visualisoinnin mallin ennusteessa
    
    Suppeampi malli viidellä muuttujalla on yksinkertaisempi ja mahdollisesti robustimpi kuin laajempi malli,
    joka käyttää kaikkia mahdollisia muuttujia.
    """)

# Mallin uudelleenkoulutus -toiminnallisuus
with st.expander("Mallin uudelleenkoulutus"):
    st.write("""
    ## Mallin uudelleenkoulutus
    
    Jos haluat kouluttaa mallin uudelleen, voit ladata uuden koulutusdatan CSV-tiedostona.
    Tiedoston pitää sisältää samat sarakkeet: Set, Line up, Shot, Shot clock, Player ja XP.
    """)
    
    uploaded_file = st.file_uploader("Lataa uusi koulutusdata (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            # Lue uusi koulutusdata
            train_data = pd.read_csv(uploaded_file, sep=';', decimal=',')
            
            # Näytä datan esikatselu
            st.subheader("Datan esikatselu")
            st.dataframe(train_data.head())
            
            # Koulutusvaihtoehdot
            st.subheader("Vaihtoehdot")
            n_estimators = st.slider("Puiden määrä metsässä", 10, 500, 100, 10)
            max_depth = st.slider("Puun maksimisyvyys", 2, 20, 5, 1)
            
            # Kouluta malli -painike
            if st.button("Kouluta malli"):
                with st.spinner("Koulutetaan mallia..."):
                    # Erota X ja y
                    if 'XP' in train_data.columns:
                        X = train_data.drop('XP', axis=1)
                        y = train_data['XP']
                        
                        # Muokataan kategorisiksi
                        X_categorical = X.astype('category')
                        
                        # One-hot koodaus
                        X_encoded = pd.get_dummies(X_categorical)
                        
                        # Kouluta malli
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                        model.fit(X_encoded, y)
                        
                        # Tallennus
                        pickle.dump(model, open("koripallo_supistettu_rf_malli.pkl", "wb"))
                        
                        # Päivitä istuntomuuttuja
                        st.session_state.model = model
                        
                        # Päivitä myös feature_names
                        st.session_state.feature_names = list(X_encoded.columns)
                        
                        st.success("Malli koulutettu ja tallennettu onnistuneesti!")
                    else:
                        st.error("Datasta puuttuu XP-sarake. Koulutus epäonnistui.")
                        
        except Exception as e:
            st.error(f"Virhe datan käsittelyssä: {str(e)}")