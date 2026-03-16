import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. MÜHENDİSLİK MOTORU ---

def get_bearing_capacity_factors(phi_deg):
    phi_rad = np.radians(phi_deg)
    if phi_deg == 0: return 5.14, 1.0, 0.0
    Nq = np.tan(np.radians(45 + phi_deg/2))**2 * np.exp(np.pi * np.tan(phi_rad))
    Nc = (Nq - 1) / np.tan(phi_rad)
    Ny = 2 * (Nq + 1) * np.tan(phi_rad)
    return round(Nc, 2), round(Nq, 2), round(Ny, 2)

def process_geotech_data(df, B, L, Df, dw, Mw, a_max, max_depth_limit):
    # Veri Temizleme ve Filtreleme
    df.columns = df.columns.str.strip().str.upper()
    for col in ['DEPTH', 'SPT_N', 'UNIT_WEIGHT']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Kullanıcının seçtiği derinliğe kadar olan kısmı al
    df = df[df['DEPTH'] <= max_depth_limit].dropna(subset=['DEPTH', 'SPT_N']).reset_index(drop=True)

    # SPT & Gerilme Düzeltmeleri
    gamma_w = 9.81
    df['Cr'] = df['DEPTH'].apply(lambda d: 0.75 if d<3 else (0.85 if d<6 else 1.0))
    df['N60'] = df['SPT_N'] * df['Cr']
    df['Sigma_V'] = df['DEPTH'] * df['UNIT_WEIGHT']
    df['u'] = df.apply(lambda r: (r['DEPTH'] - dw) * gamma_w if r['DEPTH'] > dw else 0, axis=1)
    df['Sigma_V_Eff'] = df['Sigma_V'] - df['u']
    df['C_N'] = np.sqrt(100 / df['Sigma_V_Eff'].replace(0, 10)).clip(upper=2.0)
    df['N1_60'] = df['N60'] * df['C_N']

    # Analizler
    df['phi'] = (27.1 + 0.3 * df['N1_60'] - 0.00054 * (df['N1_60']**2)).round(1)
    msf = 6.9 * np.exp(-Mw/4) - 0.058
    rd = 1.0 - 0.00765 * df['DEPTH']
    df['CSR'] = 0.65 * (a_max) * (df['Sigma_V']/df['Sigma_V_Eff']) * rd
    df['CRR'] = np.exp((df['N1_60']/14.1) + ((df['N1_60']/126)**2) - 2.8) * msf
    df['FS_Liq'] = (df['CRR'] / df['CSR']).round(2)

    # Taşıma Gücü
    q_all_list = []
    for i, row in df.iterrows():
        Nc, Nq, Ny = get_bearing_capacity_factors(row['phi'])
        q_ult = (row['Sigma_V_Eff'] * Nq * (1 + 0.1*(B/L))) + (0.5 * row['UNIT_WEIGHT'] * B * Ny * (1 + 0.1*(B/L)))
        q_all_list.append(round(q_ult / 3.0, 1))
    
    df['q_Emniyetli_kPa'] = q_all_list
    return df

# --- 2. GÖRSELLEŞTİRME ---

def draw_plots(df, selected_depth):
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, 
                        subplot_titles=("N1(60) Profili", "Sıvılaşma FS", "Emniyetli Taşıma (kPa)"))

    fig.add_trace(go.Scatter(x=df['N1_60'], y=df['DEPTH'], name='N1(60)', mode='lines+markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['FS_Liq'], y=df['DEPTH'], name='FS', line=dict(color='red'), mode='lines+markers'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['q_Emniyetli_kPa'], y=df['DEPTH'], name='q_all', line=dict(color='green'), mode='lines+markers'), row=1, col=3)
    
    fig.add_vline(x=1.0, line_dash="dash", line_color="black", row=1, col=2)
    
    # Y eksenini kullanıcının seçtiği derinliğe göre ayarla
    fig.update_yaxes(range=[selected_depth, 0], title_text="Derinlik (m)", autorange=False)
    fig.update_layout(height=800, template="plotly_dark", title_text=f"Analiz Derinliği: {selected_depth} m")
    return fig

# --- 3. ARAYÜZ ---

st.set_page_config(page_title="GeoTech-AI Pro", layout="wide")
st.title("🏗️ GeoTech-AI: Profesyonel Geoteknik Analiz")

with st.sidebar:
    st.header("🛠️ Analiz Parametreleri")
    max_depth = st.slider("Analiz Yapılacak Maks. Derinlik [m]", 5, 200, 50)
    st.divider()
    B = st.number_input("Temel Genişliği (B) [m]", 1.0)
    L = st.number_input("Temel Boyu (L) [m]", 1.0)
    dw = st.number_input("Su Seviyesi (dw) [m]", 3.0)
    Mw = st.sidebar.number_input("Magnitüd (Mw)", 7.5)
    amax = st.sidebar.number_input("İvme (amax) [g]", 0.4)
    st.divider()
    uploaded_file = st.file_uploader("Saha Verisi (Excel/CSV)", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        df_in = pd.read_csv(uploaded_file, decimal=",") if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # Analizi yaparken seçilen derinlik sınırını kullanıyoruz
        df_res = process_geotech_data(df_in, B, L, 1.5, dw, Mw, amax, max_depth)
        
        t1, t2 = st.tabs(["📝 Sonuç Tablosu", "📊 Analiz Grafikleri"])
        with t1:
            st.success(f"Seçilen {max_depth} metre derinlik için analiz tamamlandı.")
            st.dataframe(df_res, use_container_width=True)
        with t2:
            st.plotly_chart(draw_plots(df_res, max_depth), use_container_width=True)
            
    except Exception as e:
        st.error(f"Hata: {e}")
else:
    st.info("👈 Analiz derinliğini belirleyin ve verilerinizi yükleyin.")
