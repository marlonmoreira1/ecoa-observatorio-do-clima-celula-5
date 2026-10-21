import os
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
from ia_assistant import renderizar_chat_ia
from utils import carregar_dados_atlas, gerar_mapa_pydeck_3d, formata_moeda
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

datas_validas = df_raw['dt_registro'].dropna()
if not datas_validas.empty:
    min_date = datas_validas.min().date()
    max_date = datas_validas.max().date()
    periodo_registro = st.sidebar.date_input("Data de Registro", value=[], min_value=min_date, max_value=max_date)
    if isinstance(periodo_registro, (list, tuple)) and len(periodo_registro) == 2:
        df_filtrado = df_filtrado[
            (df_filtrado['dt_registro'].dt.date >= periodo_registro[0]) & 
            (df_filtrado['dt_registro'].dt.date <= periodo_registro[1])
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

def gerar_grafico_pareto(df_dados, col_categoria, col_valor, titulo, max_categorias=15):
    df_grp = df_dados.groupby(col_categoria)[col_valor].sum().reset_index()
    df_grp = df_grp[df_grp[col_valor] > 0].sort_values(by=col_valor, ascending=False).reset_index(drop=True)
    
    if df_grp.empty:
        fig = go.Figure()
        fig.update_layout(title="Sem dados suficientes", template="plotly_white")
        return fig
    
    total_geral = df_grp[col_valor].sum()
    if len(df_grp) > max_categorias:
        top_df = df_grp.iloc[:max_categorias].copy()
        outros_val = df_grp.iloc[max_categorias:][col_valor].sum()
        outros_row = pd.DataFrame({col_categoria: ['Outros'], col_valor: [outros_val]})
        df_grp = pd.concat([top_df, outros_row], ignore_index=True)
    
    df_grp['pct'] = (df_grp[col_valor] / total_geral) * 100
    df_grp['acumulado'] = df_grp['pct'].cumsum()
    corte_idx = (df_grp['acumulado'] >= 50).idxmax() if not df_grp['acumulado'].empty else 0
    cores = ['#ef4444' if i <= corte_idx else '#cbd5e1' for i in range(len(df_grp))]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_grp[col_categoria],
        y=df_grp[col_valor],
        name="Valor (R$)",
        marker_color=cores,
        text=[formata_moeda(v) for v in df_grp[col_valor]],
        textposition="outside",
        textfont=dict(size=11, color="#0f172a"),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df_grp[col_categoria],
        y=df_grp['acumulado'],
        name="% Acumulado",
        yaxis="y2",
        mode="lines+markers+text",
        text=[f"{p:.1f}%" for p in df_grp['acumulado']],
        textposition="top center",
        textfont=dict(size=10, color="#b45309", weight="bold"),
        line=dict(color="#d97706", width=3),
        marker=dict(size=7),
        showlegend=False
    ))
    fig.add_shape(type="line", x0=-0.5, x1=len(df_grp)-0.5, y0=50, y1=50, yref="y2", line=dict(color="#dc2626", width=1.5, dash="dash"))
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15, color="#0f172a")),
        xaxis=dict(title="", tickangle=-30, tickfont=dict(size=11, color="#334155")),
        yaxis=dict(title="", showticklabels=False, visible=False),
        yaxis2=dict(title="", overlaying="y", side="right", range=[0, 118], showticklabels=False, visible=False),
        showlegend=False,
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=90)
    )
    return fig

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
    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = gerar_grafico_pareto(df, 'descricao_tipologia', 'PEPR_total_privado', "Pareto: Prejuízo Privado por Tipo de Desastre")
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        cols_priv = [c for c in df.columns if c.startswith('PEPR_') and c != 'PEPR_total_privado']
        if cols_priv:
            df_setores_priv = df[cols_priv].sum().reset_index()
            df_setores_priv.columns = ['Setor Privado', 'Valor']
            df_setores_priv['Setor Privado'] = df_setores_priv['Setor Privado'].str.replace('PEPR_', '', regex=False).str.replace('(R$)', '', regex=False).str.strip()
            fig_set_priv = gerar_grafico_pareto(df_setores_priv, 'Setor Privado', 'Valor', "Pareto: Setores Privados Mais Atingidos")
            st.plotly_chart(fig_set_priv, use_container_width=True)

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
    col_a, col_b = st.columns(2)
    with col_a:
        fig3 = gerar_grafico_pareto(df, 'descricao_tipologia', 'PEPL_total_publico', "Pareto: Prejuízo Público por Tipo de Desastre")
        st.plotly_chart(fig3, use_container_width=True)
    with col_b:
        cols_pub = [c for c in df.columns if c.startswith('PEPL_') and c != 'PEPL_total_publico']
        if cols_pub:
            df_setores_pub = df[cols_pub].sum().reset_index()
            df_setores_pub.columns = ['Serviço Público', 'Valor']
            df_setores_pub['Serviço Público'] = df_setores_pub['Serviço Público'].str.replace('PEPL_', '', regex=False).str.replace('(R$)', '', regex=False).str.strip()
            fig_set_pub = gerar_grafico_pareto(df_setores_pub, 'Serviço Público', 'Valor', "Pareto: Serviços Públicos Mais Atingidos")
            st.plotly_chart(fig_set_pub, use_container_width=True)

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