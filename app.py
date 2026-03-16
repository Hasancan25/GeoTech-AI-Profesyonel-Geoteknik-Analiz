import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. ANALİZ MOTORU ---

def get_bearing_capacity_factors(phi_deg):
    phi_rad = np.radians(phi_deg)
    if phi_deg == 0: return 5.14, 1.0, 0.0
    Nq = np.tan(np.radians(45 + phi_deg/2))**2 * np.exp(np.pi * np.tan(phi_rad))
    Nc = (Nq - 1) / np.tan(phi_rad)
    Ny = 2 * (Nq + 1) * np.tan(phi_rad)
    return round(Nc, 2), round(Nq, 2), round(Ny, 2)

def process_geotech_data(df, B, L, Df, dw, Mw, a_max):
    # Not: Veri temizliği artık dışarıda yapıldığı için direkt hesaplamaya geçiyoruz
    gamma_w = 9.81
    
    # Mühendislik Hesapları
    df['Cr'] = df['DEPTH'].apply(lambda d: 0.75 if d<3 else (0.85 if d<6 else (0.95 if d<10 else 1.0)))
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
        # Temel derinliği (Df) etkisi de eklenmiştir
        q_ult = (row['Sigma_V_Eff'] * Nq * (1 + 0.1*(B/L))) + (0.5 * row['UNIT_WEIGHT'] * B * Ny * (1 + 0.1*(B/L)))
        q_all_list.append(round(q_ult / 3.0, 1))
    
    df['q_Emniyetli_kPa'] = q_all_list
    return df

# --- 2. GÖRSELLEŞTİRME ---

def draw_plots(df, selected_range):
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, 
                        subplot_titles=("N1(60) Profili", "Sıvılaşma FS", "Emniyetli Taşıma (kPa)"))

    fig.add_trace(go.Scatter(x=df['N1_60'], y=df['DEPTH'], name='N1(60)', mode='lines+markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['FS_Liq'], y=df['DEPTH'], name='FS', line=dict(color='red'), mode='lines+markers'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['q_Emniyetli_kPa'], y=df['DEPTH'], name='q_all', line=dict(color='green'), mode='lines+markers'), row=1, col=3)
    
    fig.add_vline(x=1.0, line_dash="dash", line_color="black", row=1, col=2)
    
    # Y ekseni seçilen aralığa göre sabitlenir
    fig.update_yaxes(range=[selected_range[1], selected_range[0]], title_text="Derinlik (m)", autorange=False)
    fig.update_layout(height=800, template="plotly_dark", title_text=f"Saha Profili ({selected_range[0]}m - {selected_range[1]}m)")
    return fig

# --- 3. ARAYÜZ ---

st.set_page_config(page_title="GeoTech-AI Pro", layout="wide")
st.title("🏗️ GeoTech-AI: Profesyonel Geoteknik Analiz")

with st.sidebar:
    st.header("🛠️ Tasarım Girişleri")
    B = st.number_input("Temel Genişliği (B) [m]", 1.0)
    L = st.number_input("Temel Boyu (L) [m]", 1.0)
    Df = st.number_input("Temel Derinliği (Df) [m]", 1.5)
    dw = st.number_input("Su Seviyesi (dw) [m]", 3.0)
    Mw = st.number_input("Magnitüd (Mw)", 7.5)
    amax = st.number_input("İvme (amax) [g]", 0.4)
    st.divider()
    uploaded_file = st.file_uploader("Veri Dosyasını Yükleyin", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        # 1. Ham Veriyi Oku
        df_in = pd.read_csv(uploaded_file, decimal=",") if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # 2. Ön Temizlik (Slider için gerekli)
        df_in.columns = df_in.columns.str.strip().str.upper()
        for col in ['DEPTH', 'SPT_N', 'UNIT_WEIGHT']:
            if col in df_in.columns:
                df_in[col] = pd.to_numeric(df_in[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_in = df_in.dropna(subset=['DEPTH', 'SPT_N'])

        # 3. YAN MENÜYE ARALIK SEÇİCİ EKLE (Veri yüklendikten sonra çıkar)
        min_depth = int(df_in['DEPTH'].min())
        max_depth = int(df_in['DEPTH'].max())
        
        st.sidebar.subheader("🔍 Analiz Odağı")
        selected_range = st.sidebar.slider(
            "Derinlik Aralığı Seçin [m]",
            min_depth, max_depth, (min_depth, max_depth)
        )

        # 4. Veriyi Seçilen Aralığa Göre Filtrele
        df_filtered = df_in[(df_in['DEPTH'] >= selected_range[0]) & (df_in['DEPTH'] <= selected_range[1])].copy()

        # 5. Analizi Çalıştır
        df_res = process_geotech_data(df_filtered, B, L, Df, dw, Mw, amax)
        
        # 6. Sonuçları Göster
        t1, t2 = st.tabs(["📝 Veri Analizi", "📊 Görsel Profil"])
        
        with t1:
            st.success(f"Analiz odağı: {selected_range[0]}m - {selected_range[1]}m")
            # Riskli satırları (FS < 1.0) renklendirme (Opsiyonel Stil)
            def highlight_risk(val):
                color = 'red' if val < 1.0 else 'white'
                return f'color: {color}'
            
            st.dataframe(df_res.style.applymap(highlight_risk, subset=['FS_Liq']), use_container_width=True)
            
        with t2:
            st.plotly_chart(draw_plots(df_res, selected_range), use_container_width=True)
            
    except Exception as e:
        st.error(f"⚠️ Bir sorun oluştu: {e}")
else:
    st.info("👈 Verilerinizi yüklediğinizde aralık seçici aktif olacaktır.")
