import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Načtení trénovaného modelu
def load_model(location):
    if location == "ETU":
        return joblib.load("model_ETU_novy_model_new_3.pkl")
    elif location == "EPRU":
        return joblib.load("model_EPRU_novy_model_3.pkl")

def transform_features(df, location):
    df["Teplota_venkovni"] = df["Teplota venkovní"].astype(str).str.replace(',', '.').astype(float)
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%y %H:%M")
    df["hodina"] = df["Datum"].dt.hour
    df["den_v_tydnu"] = df["Datum"].dt.dayofweek
    df["mesic"] = df["Datum"].dt.month
    df["je_leto"] = df["mesic"].isin([6, 7, 8])

    df["month_sin"] = np.sin(2 * np.pi * df["mesic"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["mesic"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hodina"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hodina"] / 24)
    df["sezonost_mesic"] = df["mesic"] * df["Teplota_venkovni"]

    if location == "ETU":
        features = ['Teplota venkovní', 'je_leto', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'sezonost_mesic']
    elif location == "EPRU":
        features = ['Teplota venkovní', 'je_leto', 'month_cos', 'hour_sin', 'hour_cos', 'sezonost_mesic']

    return df, df[features]


# ====================
# Streamlit UI
# ====================
st.set_page_config(layout="wide")
st.title("🔥 Predikce potřeby tepla — EPRU a ETU")

location = st.selectbox("Vyberte lokalitu", ["ETU", "EPRU"])

uploaded_file = st.file_uploader("📤 Nahrajte Excel soubor (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df["Teplota_venkovni"] = df["Teplota venkovní"].astype(str).str.replace(',', '.').astype(float)
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%y %H:%M")
    # Kontrola, zda soubor obsahuje správná data
    if "Teplota venkovní" not in df.columns or "Datum" not in df.columns:
        st.error("Soubor musí obsahovat sloupce 'Datum' a 'Teplota venkovní'.")
    elif len(df) != 24:
        st.warning("Soubor by měl obsahovat 24 řádků (hodinové predikce).")
    else:
        model = load_model(location)
        df_transformed, X = transform_features(df, location)
        predictions = model.predict(X)
        df_transformed["Predikce_tepla"] = predictions
        
        df_display = df_transformed[["Datum", "Teplota_venkovni", "Predikce_tepla"]]
        df_display = df_display[["Teplota_venkovni", "Predikce_tepla"]].copy()
        df_display = df_display.applymap(lambda x: str(round(x, 2)).replace('.', ','))
        df_display["Datum"] = df_transformed["Datum"]
        df_display = df_display[["Datum", "Teplota_venkovni", "Predikce_tepla"]]
        # Show output
        st.subheader(f"📊 Výsledky predikce - {location}:")
        st.dataframe(df_transformed[["Datum", "Teplota_venkovni", "Predikce_tepla"]], use_container_width=True)

        csv = df_display.to_csv(index=False, sep = ";").encode('utf-8')
        st.download_button(
            label="📥 Stáhnout výsledky jako CSV",
            data=csv,
            file_name=f'predikce_{location}.csv',
            mime='text/csv',
        )
        
        predikce = df_transformed["Predikce_tepla"].sum()
        st.write(f"### Predikované množství tepla: {predikce:.2f} GJ/den")
        # Plot
        fig, ax = plt.subplots(figsize=(15,3))
        ax.plot(df_transformed["hodina"], df_transformed["Predikce_tepla"], marker="o", color = "red", label="Predikovaná dodávka tepla")
        ax.set_title(f"Množství predikované tepla — {location}")
        ax.set_xlabel("čas [h]")
        ax.set_ylabel("Množství tepla (GJ/h)")
        ax.legend()
        st.pyplot(fig)
    # Zobrazení výsledku
    
