import os
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
from ia_assistant import renderizar_chat_ia
from utils import carregar_dados_atlas, gerar_mapa_pydeck_3d, formata_moeda, gerar_grafico_pareto
from utils import BUCKET_ECONOMICO

st.set_page_config(
    page_title="Observatório do Clima - Impacto Econômico",
    page_icon="🌡️",
    layout="wide"
)

st.markdown("""
<style>
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        flex: 1;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0f172a;
    }
    .total-card { border-left: 5px solid #dc2626; }
    .privado-card { border-left: 5px solid #d97706; }
    .publico-card { border-left: 5px solid #2563eb; }
    .material-card { border-left: 5px solid #059669; }

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

st.sidebar.title("🔍 Filtros de Análise")
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

v_total = df['prejuizo_total'].sum()
v_privado = df.get('PEPR_total_privado', pd.Series([0])).sum()
v_publico = df.get('PEPL_total_publico', pd.Series([0])).sum()
v_material = df.get('DM_total_danos_materiais', pd.Series([0])).sum()

st.title("📊 Observatório do Clima — Impacto Econômico")

st.markdown(f"""
<div class="metric-container">
    <div class="metric-card total-card">
        <div class="metric-title">Prejuízo Total Econômico</div>
        <div class="metric-value">{formata_moeda(v_total)}</div>
    </div>
    <div class="metric-card privado-card">
        <div class="metric-title">Prejuízo Privado</div>
        <div class="metric-value">{formata_moeda(v_privado)}</div>
    </div>
    <div class="metric-card publico-card">
        <div class="metric-title">Prejuízo Público</div>
        <div class="metric-value">{formata_moeda(v_publico)}</div>
    </div>
    <div class="metric-card material-card">
        <div class="metric-title">Dano Material</div>
        <div class="metric-value">{formata_moeda(v_material)}</div>
    </div>
</div>
""", unsafe_allow_html=True)



tab1, tab2, tab3, tab4 = st.tabs(["💰 Prejuízo Total", "🏢 Prejuízo Privado", "🏛️ Prejuízo Público", "🏚️ Dano Material"])

with tab1:
    st.subheader("Análise de Prejuízo Total")
    deck1 = gerar_mapa_pydeck_3d(df, 'descricao_tipologia', 'prejuizo_total', nome_metrica_exata="Prejuízo Total Econômico")
    st.pydeck_chart(deck1)
    st.markdown("---")
    fig1 = gerar_grafico_pareto(df, 'descricao_tipologia', 'prejuizo_total', "Pareto: Prejuízo Total por Tipo de Desastre")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("---")
    renderizar_chat_ia("tab1", "Prejuízo Total", df, bucket_destinatario=BUCKET_ECONOMICO, prefixo_arquivo="conversa_economico")

with tab2:
    st.subheader("Análise de Prejuízo Privado")
    sub_map1, sub_map2 = st.tabs(["🗺️ Mapa por Tipo de Desastre", "📊 Mapa por Setor Privado"])
    with sub_map1:
        deck2_a = gerar_mapa_pydeck_3d(df, 'descricao_tipologia', 'PEPR_total_privado', nome_metrica_exata="Prejuízo Privado")
        st.pydeck_chart(deck2_a)
    with sub_map2:
        cols_priv = [c for c in df.columns if c.startswith('PEPR_') and c != 'PEPR_total_privado']
        if cols_priv:
            df_priv_melt = df.melt(
                id_vars=['codigo_ibge_6', 'nome', 'Sigla_UF', 'regiao', 'latitude', 'longitude', 'dt_registro'],
                value_vars=cols_priv,
                var_name='setor_raw',
                value_name='valor_setor'
            )
            df_priv_melt['Setor Privado'] = df_priv_melt['setor_raw'].str.replace('PEPR_', '', regex=False).str.replace('(R$)', '', regex=False).str.strip()
            deck2_b = gerar_mapa_pydeck_3d(df_priv_melt, 'Setor Privado', 'valor_setor', nome_metrica_exata="Prejuízo Privado por Setor")
            st.pydeck_chart(deck2_b)
    
    st.markdown("---")
    
    # ---------------- ALTERADO AQUI ----------------
    fig2 = gerar_grafico_pareto(df, 'descricao_tipologia', 'PEPR_total_privado', "Pareto: Prejuízo Privado por Tipo de Desastre")
    st.plotly_chart(fig2, use_container_width=True)
    
    cols_priv = [c for c in df.columns if c.startswith('PEPR_') and c != 'PEPR_total_privado']
    if cols_priv:
        df_setores_priv = df[cols_priv].sum().reset_index()
        df_setores_priv.columns = ['Setor Privado', 'Valor']
        df_setores_priv['Setor Privado'] = df_setores_priv['Setor Privado'].str.replace('PEPR_', '', regex=False).str.replace('(R$)', '', regex=False).str.strip()
        fig_set_priv = gerar_grafico_pareto(df_setores_priv, 'Setor Privado', 'Valor', "Pareto: Setores Privados Mais Atingidos")
        st.plotly_chart(fig_set_priv, use_container_width=True)
    # -----------------------------------------------

    st.markdown("---")
    renderizar_chat_ia("tab2", "Prejuízo Privado", df, bucket_destinatario=BUCKET_ECONOMICO, prefixo_arquivo="conversa_economico")

with tab3:
    st.subheader("Análise de Prejuízo Público")
    sub_pub1, sub_pub2 = st.tabs(["🗺️ Mapa por Tipo de Desastre", "📊 Mapa por Serviço Público"])
    with sub_pub1:
        deck3_a = gerar_mapa_pydeck_3d(df, 'descricao_tipologia', 'PEPL_total_publico', nome_metrica_exata="Prejuízo Público")
        st.pydeck_chart(deck3_a)
    with sub_pub2:
        cols_pub = [c for c in df.columns if c.startswith('PEPL_') and c != 'PEPL_total_publico']
        if cols_pub:
            df_pub_melt = df.melt(
                id_vars=['codigo_ibge_6', 'nome', 'Sigla_UF', 'regiao', 'latitude', 'longitude', 'dt_registro'],
                value_vars=cols_pub,
                var_name='servico_raw',
                value_name='valor_servico'
            )
            df_pub_melt['Serviço Público'] = df_pub_melt['servico_raw'].str.replace('PEPL_', '', regex=False).str.replace('(R$)', '', regex=False).str.strip()
            deck3_b = gerar_mapa_pydeck_3d(df_pub_melt, 'Serviço Público', 'valor_servico', nome_metrica_exata="Prejuízo Público por Serviço")
            st.pydeck_chart(deck3_b)
    
    st.markdown("---")
    
    # ---------------- ALTERADO AQUI ----------------
    fig3 = gerar_grafico_pareto(df, 'descricao_tipologia', 'PEPL_total_publico', "Pareto: Prejuízo Público por Tipo de Desastre")
    st.plotly_chart(fig3, use_container_width=True)
    
    cols_pub = [c for c in df.columns if c.startswith('PEPL_') and c != 'PEPL_total_publico']
    if cols_pub:
        df_setores_pub = df[cols_pub].sum().reset_index()
        df_setores_pub.columns = ['Serviço Público', 'Valor']
        df_setores_pub['Serviço Público'] = df_setores_pub['Serviço Público'].str.replace('PEPL_', '', regex=False).str.replace('(R$)', '', regex=False).str.strip()
        fig_set_pub = gerar_grafico_pareto(df_setores_pub, 'Serviço Público', 'Valor', "Pareto: Serviços Públicos Mais Atingidos")
        st.plotly_chart(fig_set_pub, use_container_width=True)
    # -----------------------------------------------

    st.markdown("---")
    renderizar_chat_ia("tab3", "Prejuízo Público", df, bucket_destinatario=BUCKET_ECONOMICO, prefixo_arquivo="conversa_economico")

with tab4:
    st.subheader("Análise de Danos Materiais")
    deck4 = gerar_mapa_pydeck_3d(df, 'descricao_tipologia', 'DM_total_danos_materiais', nome_metrica_exata="Dano Material")
    st.pydeck_chart(deck4)
    st.markdown("---")
    fig4 = gerar_grafico_pareto(df, 'descricao_tipologia', 'DM_total_danos_materiais', "Pareto: Danos Materiais por Tipo de Desastre")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")
    renderizar_chat_ia("tab4", "Danos Materiais", df, bucket_destinatario=BUCKET_ECONOMICO, prefixo_arquivo="conversa_economico")