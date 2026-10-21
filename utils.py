import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from google.cloud import storage

# --- Dicionário Mestre de Cores ---
DIC_CORES_MESTRE = {
    'Estiagem e Seca': {'hex': '#dc2626', 'rgb': [220, 38, 38]},
    'Enxurradas': {'hex': '#06b6d4', 'rgb': [6, 182, 212]},
    'Inundações': {'hex': '#1e3a8a', 'rgb': [30, 58, 138]},
    'Vendavais e Ciclones': {'hex': '#84cc16', 'rgb': [132, 204, 22]},
    'Granizo': {'hex': '#db2777', 'rgb': [219, 39, 119]},
    'Movimento de Massa': {'hex': '#78350f', 'rgb': [120, 53, 15]},
    'Incêndio Florestal': {'hex': '#eab308', 'rgb': [234, 179, 8]},
    'Tornado': {'hex': '#0f172a', 'rgb': [15, 23, 42]},
    'Alagamentos': {'hex': '#2563eb', 'rgb': [37, 99, 235]},
    'Chuvas Intensas': {'hex': '#7e22ce', 'rgb': [126, 34, 206]},
    'Doenças infecciosas': {'hex': '#16a34a', 'rgb': [22, 163, 74]},
    'Erosão': {'hex': '#ea580c', 'rgb': [234, 88, 12]},
    'Onda de Frio': {'hex': '#38bdf8', 'rgb': [56, 189, 248]},
    'Rompimento/Colapso de barragens': {'hex': '#881337', 'rgb': [136, 19, 55]},
    'Outros': {'hex': '#94a3b8', 'rgb': [148, 163, 184]}
}

PALETA_TOP4 = ['#dc2626', '#2563eb', '#16a34a', '#eab308']
PALETA_TOP4_RGB = [[220, 38, 38], [37, 99, 235], [22, 163, 74], [234, 179, 8]]

# --- Nomes Exatos dos Buckets do GCP ---
BUCKET_ECONOMICO = "conversas-impacto-economico"
BUCKET_SOCIAL = "conversas-impacto-social"
BUCKET_RELATORIOS = "relatorios-diagnosticos"

def formata_moeda(valor):
    if valor >= 1e9:
        return f"R$ {valor / 1e9:,.2f} B".replace(',', 'v').replace('.', ',').replace('v', '.')
    elif valor >= 1e6:
        return f"R$ {valor / 1e6:,.2f} M".replace(',', 'v').replace('.', ',').replace('v', '.')
    else:
        return f"R$ {valor:,.2f}".replace(',', 'v').replace('.', ',').replace('v', '.')

def formata_numero(valor):
    valor = float(valor) if pd.notna(valor) else 0.0
    if valor >= 1e6:
        return f"{valor / 1e6:,.2f} Mi pessoas".replace(',', 'v').replace('.', ',').replace('v', '.')
    elif valor >= 1e3:
        return f"{valor:,.0f} pessoas".replace(',', '.')
    else:
        return f"{int(valor)} pessoas"

@st.cache_data
def carregar_coordenadas():
    url = "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"
    df_coords = pd.read_csv(url)
    df_coords['codigo_ibge_6'] = df_coords['codigo_ibge'].astype(str).str[:6]
    return df_coords[['codigo_ibge_6', 'latitude', 'longitude', 'nome']]

@st.cache_data
def carregar_dados_atlas(caminho_csv="BD_Atlas_Final.csv"):
    df_coords = carregar_coordenadas()
    if not os.path.exists(caminho_csv):
        caminho_csv = "bd_atlas.csv"
        
    if os.path.exists(caminho_csv):
        try:
            df = pd.read_csv(caminho_csv, sep=';', low_memory=False)
            if 'Cod_IBGE_Mun' not in df.columns:
                df = pd.read_csv(caminho_csv, sep=',', low_memory=False)
        except Exception:
            df = pd.read_csv(caminho_csv, sep=',', low_memory=False)
    else:
        st.warning("Arquivo de dados não encontrado. Gerando base de testes.")
        np.random.seed(42)
        n = 1000
        ibges = df_coords['codigo_ibge_6'].sample(n, replace=True).values
        datas = pd.date_range(start='2010-01-01', end='2023-12-31', periods=n)
        df = pd.DataFrame({
            'Cod_IBGE_Mun': ibges,
            'Data_Registro': datas.strftime('%Y-%m-%d'),
            'regiao': np.random.choice(['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul'], n),
            'Sigla_UF': np.random.choice(['BA', 'RJ', 'SP', 'MG', 'RS', 'SC', 'PR', 'PE'], n),
            'descricao_tipologia': np.random.choice([
                'Estiagem e Seca', 'Enxurradas', 'Inundações', 'Vendavais e Ciclones', 
                'Granizo', 'Movimento de Massa', 'Incêndio Florestal', 'Tornado',
                'Alagamentos', 'Chuvas Intensas', 'Doenças infecciosas', 'Erosão'
            ], n),
            'DM_total_danos_materiais': np.random.exponential(500000, n),
            'PEPL_total_publico': np.random.exponential(800000, n),
            'PEPR_total_privado': np.random.exponential(1200000, n),
            'DH_total_danos_humanos_diretos': np.random.randint(0, 500, n),
            'DH_MORTOS': np.random.randint(0, 10, n),
            'DH_DESAPARECIDOS': np.random.randint(0, 5, n),
            'DH_FERIDOS': np.random.randint(0, 50, n),
            'DH_ENFERMOS': np.random.randint(0, 30, n),
            'DH_DESABRIGADOS': np.random.randint(0, 200, n),
            'DH_DESALOJADOS': np.random.randint(0, 300, n),
            'DH_AFETADOS_SECA_ESTIAGEM': np.random.randint(0, 1000, n)
        })

    if 'Data_Registro' in df.columns:
        df['dt_registro'] = pd.to_datetime(df['Data_Registro'], errors='coerce')
    elif 'Data_Evento' in df.columns:
        df['dt_registro'] = pd.to_datetime(df['Data_Evento'], errors='coerce')
    else:
        df['dt_registro'] = pd.Timestamp('2020-01-01')

    if 'Cod_IBGE_Mun' in df.columns:
        df['codigo_ibge_6'] = df['Cod_IBGE_Mun'].astype(str).str.extract(r'(\d{6})')[0]
    
    cols_num = [c for c in df.columns if 'total' in c or '(R$)' in c or c.startswith('DM_') or c.startswith('PE') or c.startswith('DH_')]
    for c in cols_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    if 'PE_PLePR' in df.columns:
        df['prejuizo_total'] = df['PE_PLePR'] + df.get('DM_total_danos_materiais', 0)
    else:
        df['prejuizo_total'] = df.get('PEPR_total_privado', 0) + df.get('PEPL_total_publico', 0) + df.get('DM_total_danos_materiais', 0)

    df_merged = df.merge(df_coords, on='codigo_ibge_6', how='inner')
    return df_merged

def obter_maapeamento_top4(df_dados, col_cat, col_val):
    df_sum = df_dados.groupby(col_cat)[col_val].sum().reset_index()
    df_sum = df_sum[df_sum[col_val] > 0].sort_values(by=col_val, ascending=False)
    top4_cats = df_sum.head(4)[col_cat].tolist()
    
    mapping_dict = {}
    for idx, cat in enumerate(top4_cats):
        if cat in DIC_CORES_MESTRE:
            mapping_dict[cat] = DIC_CORES_MESTRE[cat]
        else:
            mapping_dict[cat] = {'hex': PALETA_TOP4[idx % 4], 'rgb': PALETA_TOP4_RGB[idx % 4]}
            
    mapping_dict['Outros'] = DIC_CORES_MESTRE['Outros']
    return top4_cats, mapping_dict

def renderizar_legenda_dinamica(mapping_dict):
    items = []
    for cat, info in mapping_dict.items():
        items.append(
            f'<div class="legend-badge"><span class="legend-dot" style="background-color: {info["hex"]};"></span><span>{cat}</span></div>'
        )
    html_str = f'<div class="legend-box"><div class="legend-title">📌 Legenda do Mapa (Top 4 Categorias + Outros)</div><div class="legend-items-container">{"".join(items)}</div></div>'
    st.markdown(html_str, unsafe_allow_html=True)
    

def gerar_mapa_pydeck_3d(df_dados, col_cat, col_metrica, nome_metrica_exata="Prejuízo", eh_social=False):
    if df_dados.empty or col_metrica not in df_dados.columns:
        view_state = pdk.ViewState(latitude=-14.2350, longitude=-51.9253, zoom=3.8, pitch=45, bearing=15)
        return pdk.Deck(layers=[], initial_view_state=view_state)

    top4_cats, map_cores = obter_maapeamento_top4(df_dados, col_cat, col_metrica)
    renderizar_legenda_dinamica(map_cores)

    cols_group = ['codigo_ibge_6', 'nome', 'latitude', 'longitude', col_cat]
    for c in ['Sigla_UF', 'regiao']:
        if c in df_dados.columns and c not in cols_group:
            cols_group.append(c)

    df_mun = df_dados.groupby(cols_group).agg(
        val_sum=(col_metrica, 'sum'),
        ocorrencias=('codigo_ibge_6', 'count')
    ).reset_index()

    if 'Sigla_UF' not in df_mun.columns:
        df_mun['Sigla_UF'] = '-'
    if 'regiao' not in df_mun.columns:
        df_mun['regiao'] = '-'

    df_mun = df_mun[df_mun['val_sum'] > 0].copy()

    if df_mun.empty:
        view_state = pdk.ViewState(latitude=-14.2350, longitude=-51.9253, zoom=3.8, pitch=45, bearing=15)
        return pdk.Deck(layers=[], initial_view_state=view_state)

    df_totais_mun = df_mun.groupby('codigo_ibge_6')['val_sum'].sum().reset_index().rename(columns={'val_sum': 'total_municipio'})
    df_mun = df_mun.merge(df_totais_mun, on='codigo_ibge_6', how='left')

    func_fmt = formata_numero if eh_social else formata_moeda

    def extrair_resumo_categorias(cod_ibge):
        sub = df_mun[df_mun['codigo_ibge_6'] == cod_ibge]
        itens = [f"• {row[col_cat]}: {func_fmt(row['val_sum'])}" for _, row in sub.iterrows()]
        return "\n".join(itens)

    resumos_dict = {cod: extrair_resumo_categorias(cod) for cod in df_mun['codigo_ibge_6'].unique()}
    df_mun['outras_categorias_str'] = df_mun['codigo_ibge_6'].map(resumos_dict)

    df_mun['cat_agrupada'] = df_mun[col_cat].apply(lambda x: x if x in top4_cats else 'Outros')
    df_mun = df_mun.sort_values(by='val_sum', ascending=False).reset_index(drop=True)

    df_mun['max_freq_cat'] = df_mun.groupby('cat_agrupada')['ocorrencias'].transform('max')
    max_val = df_mun['val_sum'].max()

    df_mun['elevation'] = (df_mun['val_sum'] / max_val) * 280000
    df_mun['prejuizo_fmt'] = df_mun['val_sum'].apply(func_fmt)
    df_mun['total_municipio_fmt'] = df_mun['total_municipio'].apply(func_fmt)

    def calc_color(row):
        cat = row['cat_agrupada']
        freq = row['ocorrencias']
        max_f = row['max_freq_cat'] if row['max_freq_cat'] > 0 else 1
        rgb = map_cores.get(cat, DIC_CORES_MESTRE['Outros'])['rgb']
        alpha = int(110 + (freq / max_f) * 145)
        return [rgb[0], rgb[1], rgb[2], min(alpha, 255)]

    df_mun['color'] = df_mun.apply(calc_color, axis=1)

    layer_3d = pdk.Layer(
        "ColumnLayer",
        data=df_mun,
        get_position=["longitude", "latitude"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=11000,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    centro_lat = df_mun['latitude'].median()
    centro_lon = df_mun['longitude'].median()
    
    view_state = pdk.ViewState(latitude=centro_lat, longitude=centro_lon, zoom=4.2, pitch=45, bearing=15)

    tooltip = {
        "html": f"<b>Município:</b> {{nome}} - {{Sigla_UF}} ({{regiao}})<br/>"
                f"<b>Categoria Focada:</b> {{{col_cat}}}<br/>"
                f"<b>Frequência na Categoria:</b> {{ocorrencias}} registro(s)<br/>"
                f"<b>{nome_metrica_exata} (Categoria):</b> {{prejuizo_fmt}}<br/>"
                f"<hr style='margin: 5px 0; border-color: #475569;'/>"
                f"<b>📊 {nome_metrica_exata.upper()} TOTAL NO MUNICÍPIO:</b> {{total_municipio_fmt}}<br/>"
                f"<b>Detalhamento das Categorias em {{nome}}:</b><br/>"
                f"{{outras_categorias_str}}",
        "style": {"backgroundColor": "#0f172a", "color": "white", "fontSize": "12px", "padding": "10px", "borderRadius": "6px", "whiteSpace": "pre-line"}
    }

    return pdk.Deck(layers=[layer_3d], initial_view_state=view_state, tooltip=tooltip, map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json")

# --- FUNÇÃO DE SALVAMENTO NO BUCKET COM SOBRESCRIÇÃO ---
def salvar_historico_no_bucket(bucket_name, file_prefix, historico, modulo_nome, creds_path="clima-ecoa-d53491e6166f.json"):
    """
    Salva/sobrescreve o histórico no arquivo fixo do dia com padrão:
    - conversa_economico_YYYYMMDD_HHMMSS.json
    - conversa_social_YYYYMMDD_HHMMSS.json
    """
    try:
        if not os.path.exists(creds_path):
            if "GCP_SERVICE_ACCOUNT" in st.secrets:
                client = storage.Client.from_service_account_info(st.secrets["GCP_SERVICE_ACCOUNT"])
            else:
                return False, "Arquivo de credenciais GCP não encontrado."
        else:
            client = storage.Client.from_service_account_json(creds_path)

        bucket = client.bucket(bucket_name)

        # Se já existe um timestamp gravado na sessão, reaproveita para sobrescrever o mesmo arquivo do dia
        sess_key_file = f"filename_{file_prefix}"
        if sess_key_file not in st.session_state:
            timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            st.session_state[sess_key_file] = f"{file_prefix}_{timestamp_str}.json"

        nome_arquivo = st.session_state[sess_key_file]
        blob = bucket.blob(nome_arquivo)

        payload = {
            "modulo": modulo_nome,
            "ultima_atualizacao": pd.Timestamp.now().isoformat(),
            "historico": historico
        }

        json_data = json.dumps(payload, ensure_ascii=False, indent=2)
        blob.upload_from_string(json_data, content_type="application/json")
        return True, "Sucesso"
    except Exception as e:
        return False, str(e)