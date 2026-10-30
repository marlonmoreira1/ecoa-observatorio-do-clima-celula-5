import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ia_assistant import renderizar_chat_ia
from utils import carregar_dados_atlas, gerar_mapa_pydeck_3d, formata_numero, gerar_grafico_pareto
from utils import BUCKET_SOCIAL

st.set_page_config(
    page_title="Observatório do Clima - Impacto Social",
    page_icon="👥",
    layout="wide"
)

# Estilização CSS dos Cards Sociais (5 Cards Separados + CSS da Legenda do Mapa)
st.markdown("""
<style>
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        flex: 1;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
    }
    .card-mortos { border-left: 5px solid #dc2626; }
    .card-desaparecidos { border-left: 5px solid #9333ea; }
    .card-feridos { border-left: 5px solid #d97706; }
    .card-deslocados { border-left: 5px solid #2563eb; }
    .card-afetados { border-left: 5px solid #059669; }

    /* Estilo da Legenda do Mapa */
    .legend-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .legend-title {
        font-weight: 700;
        font-size: 12px;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .legend-items-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px 16px;
    }
    .legend-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 600;
        color: #334155;
    }
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 3px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

df_raw = carregar_dados_atlas()

st.sidebar.title("🔍 Filtros de Análise (Social)")
df_filtrado = df_raw.copy()

datas_validas = df_raw['dt_registro']
if not datas_validas.empty:
    # Cria uma lista ordenada apenas com Mês/Ano únicos da base
    periodos = sorted(datas_validas.dt.to_period('M').unique())
    periodos_str = [p.strftime('%m/%Y') for p in periodos]
    
    if len(periodos_str) >= 2:
        sel_periodo = st.sidebar.select_slider(
            "Período de Registro (Mês/Ano)",
            options=periodos_str,
            value=(periodos_str[0], periodos_str[-1])
        )
        
        # Converte de volta para data (do 1º dia do mês inicial ao último dia do mês final)
        data_inicio = pd.to_datetime(sel_periodo[0], format='%m/%Y')
        data_fim = pd.to_datetime(sel_periodo[1], format='%m/%Y') + pd.offsets.MonthEnd(1)
        
        df_filtrado = df_filtrado[
            (df_filtrado['dt_registro'] >= data_inicio) & 
            (df_filtrado['dt_registro'] <= data_fim)
        ]

regioes = sorted([r for r in df_filtrado['regiao'].dropna().unique()])
sel_regiao = st.sidebar.multiselect("Região", options=regioes, default=[])
if sel_regiao:
    df_filtrado = df_filtrado[df_filtrado['regiao'].isin(sel_regiao)]

ufs = sorted([u for u in df_filtrado['Sigla_UF'].dropna().unique()])
sel_uf = st.sidebar.multiselect("Estado (UF)", options=ufs, default=[])
if sel_uf:
    df_filtrado = df_filtrado[df_filtrado['Sigla_UF'].isin(sel_uf)]

tipologias_unicas = sorted([t for t in df_filtrado['descricao_tipologia'].dropna().unique()])
sel_tipologia = st.sidebar.multiselect("Tipo de Desastre", options=tipologias_unicas, default=[])
if sel_tipologia:
    df_filtrado = df_filtrado[df_filtrado['descricao_tipologia'].isin(sel_tipologia)]

df = df_filtrado


if 'DH_total_afetados_geral' not in df.columns:
    df['DH_total_afetados_geral'] = (
        df.get('DH_MORTOS', 0) +
        df.get('DH_DESAPARECIDOS', 0) +
        df.get('DH_DESABRIGADOS', 0) + 
        df.get('DH_FERIDOS', 0) +
        df.get('DH_ENFERMOS', 0) +
        df.get('DH_OUTROS AFETADOS', 0)
    )
if 'DH_deslocados_total' not in df.columns:
    df['DH_deslocados_total'] = df.get('DH_DESABRIGADOS', 0) + df.get('DH_DESALOJADOS', 0)

tot_mortos = df.get('DH_MORTOS', pd.Series([0])).sum()
tot_desaparecidos = df.get('DH_DESAPARECIDOS', pd.Series([0])).sum()
tot_feridos = df.get('DH_FERIDOS', pd.Series([0])).sum() + df.get('DH_ENFERMOS', pd.Series([0])).sum()
tot_deslocados = df['DH_deslocados_total'].sum()
tot_afetados = df['DH_total_afetados_geral'].sum()

st.title("👥 Observatório do Clima — Impacto Social e Humano")

# CARDS SUPERIORES COM VÍTIMAS FATAIS E DESAPARECIDOS SEPARADOS
st.markdown(f"""
<div class="metric-container">
    <div class="metric-card card-mortos">
        <div class="metric-title">Vítimas Fatais</div>
        <div class="metric-value">{formata_numero(tot_mortos)}</div>
    </div>
    <div class="metric-card card-desaparecidos">
        <div class="metric-title">Desaparecidos</div>
        <div class="metric-value">{formata_numero(tot_desaparecidos)}</div>
    </div>
    <div class="metric-card card-feridos">
        <div class="metric-title">Feridos e Enfermos</div>
        <div class="metric-value">{formata_numero(tot_feridos)}</div>
    </div>
    <div class="metric-card card-deslocados">
        <div class="metric-title">Deslocados (Sem Teto)</div>
        <div class="metric-value">{formata_numero(tot_deslocados)}</div>
    </div>
    <div class="metric-card card-afetados">
        <div class="metric-title">Total Geral de Afetados</div>
        <div class="metric-value">{formata_numero(tot_afetados)}</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Total de Afetados", 
    "🚑 Danos à Vida e Saúde", 
    "🏠 Deslocados e Sem Teto", 
    "🌾 Afetados por Seca/Estiagem"
])

with tab1:
    st.subheader("Análise Geral da População Afetada")
    deck1 = gerar_mapa_pydeck_3d(df, 'descricao_tipologia', 'DH_total_afetados_geral', nome_metrica_exata="Total de Afetados", eh_social=True)
    st.pydeck_chart(deck1)
    st.markdown("---")
    fig1 = gerar_grafico_pareto(df, 'descricao_tipologia', 'DH_total_afetados_geral', "Pareto: População Afetada por Tipo de Desastre", eh_social=True)
    st.plotly_chart(fig1, use_container_width=True)
    # st.markdown("---")
    # renderizar_chat_ia("soc_tab1", "Total de Afetados", df, bucket_destinatario=BUCKET_SOCIAL, prefixo_arquivo="conversa_social")

with tab2:
    st.subheader("Análise de Danos à Vida (Mortos, Feridos e Enfermos)")
    sub_tab1, sub_tab2 = st.tabs(["☠️ Mortalidade e Desaparecidos", "🚑 Feridos e Enfermos"])
    with sub_tab1:
        df['DH_mortes_desap'] = df.get('DH_MORTOS', 0) + df.get('DH_DESAPARECIDOS', 0)
        deck2_a = gerar_mapa_pydeck_3d(df, 'descricao_tipologia', 'DH_mortes_desap', nome_metrica_exata="Vítimas Fatais e Desaparecidos", eh_social=True)
        st.pydeck_chart(deck2_a)
    with sub_tab2:
        df['DH_feridos_enf'] = df.get('DH_FERIDOS', 0) + df.get('DH_ENFERMOS', 0)
        deck2_b = gerar_mapa_pydeck_3d(df, 'descricao_tipologia', 'DH_feridos_enf', nome_metrica_exata="Feridos e Enfermos", eh_social=True)
        st.pydeck_chart(deck2_b)

    st.markdown("---")
    
    # ---------------- ALTERADO AQUI ----------------
    fig2_a = gerar_grafico_pareto(df, 'descricao_tipologia', 'DH_MORTOS', "Pareto: Mortes por Tipo de Desastre", eh_social=True)
    st.plotly_chart(fig2_a, use_container_width=True)

    fig2_b = gerar_grafico_pareto(df, 'descricao_tipologia', 'DH_FERIDOS', "Pareto: Feridos por Tipo de Desastre", eh_social=True)
    st.plotly_chart(fig2_b, use_container_width=True)
    # -----------------------------------------------

    # st.markdown("---")
    # renderizar_chat_ia("soc_tab2", "Danos à Vida e Saúde", df, bucket_destinatario=BUCKET_SOCIAL, prefixo_arquivo="conversa_social")

with tab3:
    st.subheader("Análise de População Deslocada (Desabrigados e Desalojados)")
    sub_des1, sub_des2 = st.tabs(["🏠 Desabrigados (Abrigos Públicos)", "🚪 Desalojados (Casa de Terceiros)"])
    with sub_des1:
        deck3_a = gerar_mapa_pydeck_3d(df, 'descricao_tipologia', 'DH_DESABRIGADOS', nome_metrica_exata="Desabrigados", eh_social=True)
        st.pydeck_chart(deck3_a)
    with sub_des2:
        deck3_b = gerar_mapa_pydeck_3d(df, 'descricao_tipologia', 'DH_DESALOJADOS', nome_metrica_exata="Desalojados", eh_social=True)
        st.pydeck_chart(deck3_b)

    st.markdown("---")
    
    # ---------------- ALTERADO AQUI ----------------
    fig3_a = gerar_grafico_pareto(df, 'descricao_tipologia', 'DH_DESABRIGADOS', "Pareto: Desabrigados por Tipo de Desastre", eh_social=True)
    st.plotly_chart(fig3_a, use_container_width=True)

    fig3_b = gerar_grafico_pareto(df, 'descricao_tipologia', 'DH_DESALOJADOS', "Pareto: Desalojados por Tipo de Desastre", eh_social=True)
    st.plotly_chart(fig3_b, use_container_width=True)
    # -----------------------------------------------

    # st.markdown("---")
    # renderizar_chat_ia("soc_tab3", "Deslocados e Sem Teto", df, bucket_destinatario=BUCKET_SOCIAL, prefixo_arquivo="conversa_social")

with tab4:
    st.subheader("Análise do Impacto Social por Secas e Estiagens")
    deck4 = gerar_mapa_pydeck_3d(df, 'descricao_tipologia', 'DH_AFETADOS_SECA_ESTIAGEM', nome_metrica_exata="Afetados por Seca/Estiagem", eh_social=True)
    st.pydeck_chart(deck4)
    st.markdown("---")
    fig4 = gerar_grafico_pareto(df, 'Sigla_UF', 'DH_AFETADOS_SECA_ESTIAGEM', "Pareto: Estados mais Atingidos por Seca e Estiagem", eh_social=True)
    st.plotly_chart(fig4, use_container_width=True)
    # st.markdown("---")
    # renderizar_chat_ia("soc_tab4", "Afetados por Seca e Estiagem", df, bucket_destinatario=BUCKET_SOCIAL, prefixo_arquivo="conversa_social")