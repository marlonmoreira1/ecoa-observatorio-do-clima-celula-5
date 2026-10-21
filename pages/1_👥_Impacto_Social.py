import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ia_assistant import renderizar_chat_ia
from utils import carregar_dados_atlas, gerar_mapa_pydeck_3d, formata_numero
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

def gerar_grafico_pareto_social(df_dados, col_categoria, col_valor, titulo, max_categorias=15):
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
        name="Pessoas",
        marker_color=cores,
        text=[formata_numero(v) for v in df_grp[col_valor]],
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

if 'DH_total_afetados_geral' not in df.columns:
    df['DH_total_afetados_geral'] = (
        df.get('DH_total_danos_humanos_diretos', 0) + 
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
    fig1 = gerar_grafico_pareto_social(df, 'descricao_tipologia', 'DH_total_afetados_geral', "Pareto: População Afetada por Tipo de Desastre")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("---")
    renderizar_chat_ia("soc_tab1", "Total de Afetados", df, bucket_destinatario=BUCKET_SOCIAL, prefixo_arquivo="conversa_social")

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
    col_a, col_b = st.columns(2)
    with col_a:
        fig2_a = gerar_grafico_pareto_social(df, 'descricao_tipologia', 'DH_MORTOS', "Pareto: Mortes por Tipo de Desastre")
        st.plotly_chart(fig2_a, use_container_width=True)
    with col_b:
        fig2_b = gerar_grafico_pareto_social(df, 'descricao_tipologia', 'DH_FERIDOS', "Pareto: Feridos por Tipo de Desastre")
        st.plotly_chart(fig2_b, use_container_width=True)

    st.markdown("---")
    renderizar_chat_ia("soc_tab2", "Danos à Vida e Saúde", df, bucket_destinatario=BUCKET_SOCIAL, prefixo_arquivo="conversa_social")

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
    col_a, col_b = st.columns(2)
    with col_a:
        fig3_a = gerar_grafico_pareto_social(df, 'descricao_tipologia', 'DH_DESABRIGADOS', "Pareto: Desabrigados por Tipo de Desastre")
        st.plotly_chart(fig3_a, use_container_width=True)
    with col_b:
        fig3_b = gerar_grafico_pareto_social(df, 'descricao_tipologia', 'DH_DESALOJADOS', "Pareto: Desalojados por Tipo de Desastre")
        st.plotly_chart(fig3_b, use_container_width=True)

    st.markdown("---")
    renderizar_chat_ia("soc_tab3", "Deslocados e Sem Teto", df, bucket_destinatario=BUCKET_SOCIAL, prefixo_arquivo="conversa_social")

with tab4:
    st.subheader("Análise do Impacto Social por Secas e Estiagens")
    deck4 = gerar_mapa_pydeck_3d(df, 'descricao_tipologia', 'DH_AFETADOS_SECA_ESTIAGEM', nome_metrica_exata="Afetados por Seca/Estiagem", eh_social=True)
    st.pydeck_chart(deck4)
    st.markdown("---")
    fig4 = gerar_grafico_pareto_social(df, 'Sigla_UF', 'DH_AFETADOS_SECA_ESTIAGEM', "Pareto: Estados mais Atingidos por Seca e Estiagem")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")
    renderizar_chat_ia("soc_tab4", "Afetados por Seca e Estiagem", df, bucket_destinatario=BUCKET_SOCIAL, prefixo_arquivo="conversa_social")