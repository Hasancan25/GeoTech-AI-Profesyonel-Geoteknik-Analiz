import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. MÜHENDİSLİK FONKSİYONLARI (MOTOR) ---

def process_geotech_data(df, B, L, Df, dw, Mw, a_max, gamma_sat, net_pressure):
    # Veri Temizleme
    df['USCS'] = df['USCS'].astype(str).str.strip().str.upper()
    df['SPT_N'] = pd.to_numeric(df['SPT_N'], errors='coerce').fillna(50)
    
# --- 1. Tij Boyu Düzeltmesi (Cr) Fonksiyonu ---
def get_cr(depth):
    if depth < 3: return 0,75
    elif 3 <= depth < 4: return 0,80
    elif 4 <= depth < 6: return 0,85
    elif 6 <= depth < 10: return 0,95
    else: return 1,0  # 10 metreden 200 metreye kadar tij boyu çarpanı 1 kabul edilir.

# --- 2. Dosya Yükleme Paneli ---
uploaded_file = st.sidebar.file_uploader("Saha Verisi Yükle (Max 200m)", type=['xlsx', 'csv'])

if uploaded_file:
    # DOSYA OKUMA: df burada oluşuyor
    if uploaded_file.name.endswith('.csv'):
        # Virgüllü sayıları (1,50 gibi) doğru okumak için decimal="," ekledik
        df_raw = pd.read_csv(uploaded_file, decimal=",")
    else:
        df_raw = pd.read_excel(uploaded_file)

    # KRİTİK TEMİZLİK: df oluştuktan hemen sonra sütunları temizle
    df_raw.columns = df_raw.columns.str.strip().str.upper()

    # Sayısal olması gereken kolonları virgül-nokta karmaşasından kurtar
    for col in ['DEPTH', 'SPT_N', 'UNIT_WEIGHT']:
        if col in df_raw.columns:
            # Önce stringe çevir, virgülü noktaya çevir, sonra sayı yap
            df_raw[col] = pd.to_numeric(df_raw[col].astype(str).str.replace(',', '.'), errors='coerce')

    try:
        # Analiz motorunu çalıştır
        df_final = process_geotech_data(df_raw, B, L, Df, dw, Mw, amax)
        
        st.success(f"Analiz Tamamlandı! 200 metre derinliğe kadar {len(df_final)} adet veri noktası işlendi.")
        
        # Grafik Çizdirme (Ölçeği 200m'ye göre ayarlar)
        fig = show_interactive_plots(df_final)
        # Grafik derinliğini verideki maksimum derinliğe göre otomatik ölçekle
        fig.update_yaxes(range=[df_final['DEPTH'].max() + 5, 0]) 
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Bir sorun çıktı: {e}")
    
    # Efektif Gerilme ve CN
    gamma_w = 9.81
    df['Sigma_V'] = df['DEPTH'] * df['UNIT_WEIGHT']
    df['u'] = df.apply(lambda r: (r['DEPTH'] - dw) * gamma_w if r['DEPTH'] > dw else 0, axis=1)
    df['Sigma_V_Eff'] = df['Sigma_V'] - df['u']
    df['C_N'] = np.sqrt(100 / df['Sigma_V_Eff'].replace(0, 10)).clip(upper=2.0)
    df['N1_60'] = df['N60'] * df['C_N']

    # Parametreler (Phi ve Cu)
    results = []
    for i, row in df.iterrows():
        if 'C' in row['USCS']: # Kil
            cu = row['N60'] * 6.25
            results.append({'phi': np.nan, 'cu': cu})
        else: # Kum
            phi = 27.1 + 0.3 * row['N1_60'] - 0.00054 * (row['N1_60']**2)
            results.append({'phi': round(phi, 1), 'cu': np.nan})
    
    param_df = pd.DataFrame(results)
    df = pd.concat([df, param_df], axis=1)

    # Sıvılaşma Analizi
    msf = 6.9 * np.exp(-Mw/4) - 0.058
    rd = 1.0 - 0.00765 * df['DEPTH']
    df['CSR'] = 0.65 * (a_max/9.81) * (df['Sigma_V']/df['Sigma_V_Eff']) * rd
    df['CRR'] = np.exp((df['N1_60']/14.1) + ((df['N1_60']/126)**2) - 2.8) * msf
    df['FS_Liq'] = (df['CRR'] / df['CSR']).round(2)
    
    return df

# --- 2. STREAMLIT ARAYÜZÜ ---

st.set_page_config(page_title="GeoTech-AI Pro", layout="wide")
st.title("🏗️ GeoTech-AI: Profesyonel Geoteknik Analiz")

st.sidebar.header("🛠️ Tasarım Parametreleri")
B = st.sidebar.number_input("Temel Genişliği (B) [m]", 1.0)
L = st.sidebar.number_input("Temel Boyu (L) [m]", 1.0)
Df = st.sidebar.number_input("Temel Derinliği (Df) [m]", 1.5)
dw = st.sidebar.number_input("Yeraltı Su Seviyesi [m]", 3.0)
Mw = st.sidebar.number_input("Deprem Magnitüdü (Mw)", 7.5)
amax = st.sidebar.number_input("Yer İvmesi (amax) [g]", 0.4)

uploaded_file = st.sidebar.file_uploader("Saha Verisi Yükle (Excel/CSV)", type=['csv', 'xlsx'])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    # Analizi Çalıştır
    df_final = process_geotech_data(df_raw, B, L, Df, dw, Mw, amax*9.81, 19.0, 150.0)
    
    tab1, tab2 = st.tabs(["📊 Analiz Sonuçları", "📈 Görselleştirme"])
    
    with tab1:
        st.dataframe(df_final, use_container_width=True)
    
    with tab2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        ax1.invert_yaxis()
        ax1.plot(df_final['N1_60'], df_final['DEPTH'], 's-b')
        ax1.set_title("N1(60) Profili")
        ax2.plot(df_final['FS_Liq'], df_final['DEPTH'], 'o-r')
        ax2.axvline(1.0, color='black', linestyle='--')
        ax2.set_title("Sıvılaşma Güvenlik Sayısı")
        st.pyplot(fig)
else:
    st.info("Lütfen sol taraftan bir ölçüm dosyası yükleyin.")


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ... (Analiz fonksiyonların bittikten sonra, grafik kısmına şu kodu yaz)

def create_interactive_plots(df):
    # İki sütunlu subplot oluşturuyoruz
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, 
                        subplot_titles=("N1(60) Profili", "Sıvılaşma Güvenlik Sayısı (FS)"))

    # 1. Grafik: N1(60)
    fig.add_trace(
        go.Scatter(x=df['N1_60'], y=df['DEPTH'], name="N1(60)",
                   mode='lines+markers', line=dict(color='blue')),
        row=1, col=1
    )

    # 2. Grafik: FS_Liq
    fig.add_trace(
        go.Scatter(x=df['FS_Liq'], y=df['DEPTH'], name="FS Liquefaction",
                   mode='lines+markers', line=dict(color='red')),
        row=1, col=2
    )

    # Kritik sınır (FS=1.0) için dikey çizgi
    fig.add_vline(x=1.0, line_dash="dash", line_color="black", row=1, col=2)

    # Eksen ayarları
    fig.update_yaxes(autorange="reversed", title_text="Derinlik (m)")
    fig.update_layout(height=600, title_text="Saha Analiz Grafikleri", showlegend=False)
    
    return fig

# Arayüzde göstermek için:
if uploaded_file:
    # ... analiz kodların ...
    fig = create_interactive_plots(df_final)
    st.plotly_chart(fig, use_container_width=True)




import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ... (Analiz fonksiyonlarının bittiği yerde, grafik gösterme kısmına gel)

def draw_plots(df):
    # Yan yana iki grafik oluşturuyoruz
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, 
                        subplot_titles=("N1(60) Profili", "Sıvılaşma Risk Analizi (FS)"))

    # 1. Grafik: N1(60) vs Derinlik
    fig.add_trace(go.Scatter(x=df['N1_60'], y=df['DEPTH'], mode='lines+markers', name='N1(60)'), row=1, col=1)

    # 2. Grafik: Sıvılaşma Güvenlik Sayısı (FS) vs Derinlik
    fig.add_trace(go.Scatter(x=df['FS_Liq'], y=df['DEPTH'], mode='lines+markers', name='FS Liquefaction', line=dict(color='red')), row=1, col=2)
    
    # FS=1.0 kritik sınır çizgisi
    fig.add_vline(x=1.0, line_dash="dash", line_color="black", row=1, col=2)

    # Derinlik eksenini ters çevir (0 üstte olsun)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=600, showlegend=False, template="plotly_white")
    
    return fig





# 1. Önce yükleme butonunu koyuyoruz
uploaded_file = st.sidebar.file_uploader("Saha Verisi Yükle (Excel veya CSV)", type=['xlsx', 'csv'])

if uploaded_file:
    # 2. Dosyayı okuyup 'df' değişkenini oluşturuyoruz
    if uploaded_file.name.endswith('.csv'):
        # Virgüllü sayıları (1,50 gibi) doğru okumak için decimal="," ekledik
        df_raw = pd.read_csv(uploaded_file, decimal=",")
    else:
        df_raw = pd.read_excel(uploaded_file)

    # 3. ŞİMDİ 'df_raw' oluştuğu için sütunları temizleyebiliriz
    df_raw.columns = df_raw.columns.str.strip().str.upper()

    # 4. Analiz fonksiyonunu çağırıyoruz (Sütun isimleri artık temiz ve büyük harf!)
    try:
        # Önceki kodundaki fonksiyon ismini ve parametreleri buraya yaz
        df_final = process_geotech_data(df_raw, B, L, Df, dw, Mw, amax)
        
        st.success("Analiz başarıyla tamamlandı!")
        st.dataframe(df_final) # Sonuç tablosu
        
        # Grafik fonksiyonunu buraya ekle
        # fig = create_interactive_plots(df_final)
        # st.plotly_chart(fig)

    except KeyError as e:
        st.error(f"Hata: Dosyanızda '{e}' isimli bir sütun bulunamadı. Lütfen Excel başlıklarını kontrol edin.")


