# Koripallo Analysaattori

Random Forest -malliin perustuva koripallo-analysaattori, joka ennustaa XP-arvoja viiden tärkeimmän muuttujan perusteella.

## Asennus

1. Kloonaa tämä repository:
git clone https://github.com/[käyttäjätunnuksesi]/koripallo-analysaattori.git
Copy
2. Asenna tarvittavat paketit:
pip install -r requirements.txt
Copy
3. Kouluta malli:
python model-training-script.py
Copy
4. Käynnistä Streamlit-sovellus:
streamlit run streamlit-app.py
Copy
## Tietoa sovelluksesta

Tämä sovellus käyttää Random Forest -mallia koripallotilastodatan analysointiin. Se keskittyy viiteen tärkeimpään muuttujaan: Set, Line up, Shot, Shot clock ja Player.

Sovellus tarjoaa kolme päätoimintoa:
- XP-arvojen ennustaminen valituilla parametreilla
- Parhaan Set-vaihtoehdon löytäminen
- Muuttujien tärkeyden visualisointi

## Käyttöohje

1. **XP Ennuste** -välilehdellä:
- Valitse arvot kaikille viidelle muuttujalle
- Paina "Ennusta XP" -painiketta
- Näet ennustetun XP-arvon ja vertailun aineiston keskiarvoon

2. **Paras Set** -välilehdellä:
- Valitse arvot muille muuttujille (Line up, Shot, Shot clock, Player)
- Paina "Etsi paras Set" -painiketta
- Näet, mikä Set-arvo tuottaa parhaan XP-ennusteen

3. **Muuttujien Tärkeys** -välilehdellä:
- Näet visuaalisen esityksen mallin käyttämien muuttujien tärkeydestä