import os
import streamlit as st
from google.cloud import storage

st.set_page_config(
    page_title="Observatório do Clima - Relatório Executivo",
    page_icon="📄",
    layout="wide"
)

BUCKET_RELATORIOS = "relatorio-diagnostico"
CREDS_PATH = os.getenv("GCP_SERVICE_ACCOUNT")

def obter_cliente_gcp():
    if CREDS_PATH and os.path.exists(CREDS_PATH):
        return storage.Client.from_service_account_json(CREDS_PATH)
    elif "GCP_SERVICE_ACCOUNT" in st.secrets:
        return storage.Client.from_service_account_info(st.secrets["GCP_SERVICE_ACCOUNT"])
    return None

def obter_ultimo_relatorio():
    """Busca o relatório Markdown (.md) mais recente no bucket de relatórios."""
    try:
        client = obter_cliente_gcp()
        if not client:
            st.error("Credenciais do Google Cloud não encontradas.")
            return None, None

        bucket = client.bucket(BUCKET_RELATORIOS)
        blobs = list(bucket.list_blobs())

        relatorios_md = [b for b in blobs if b.name.endswith(".md")]

        if not relatorios_md:
            return None, None

        # Ordena para pegar o arquivo mais recente
        relatorios_md.sort(key=lambda x: x.updated, reverse=True)
        ultimo_blob = relatorios_md[0]

        conteudo = ultimo_blob.download_as_text()
        return ultimo_blob.name, conteudo
    except Exception as e:
        st.error(f"Erro ao buscar o relatório no Google Cloud Storage: {e}")
        return None, None

st.title("📄 Relatório Executivo Consolidado")
st.markdown("Visualização e download do **Diagnóstico Executivo de Impacto Climático** mais recente gerado.")

st.markdown("---")

nome_arquivo, conteudo_md = obter_ultimo_relatorio()

if conteudo_md:
    col_info, col_btn = st.columns([3, 1])

    with col_info:
        st.success(f"📌 **Arquivo carregado:** `{nome_arquivo}`")

    with col_btn:
        st.download_button(
            label="📥 Baixar Relatório (.md)",
            data=conteudo_md,
            file_name=nome_arquivo,
            mime="text/markdown",
            use_container_width=True
        )

    st.markdown("---")

    # Renderiza o relatório Markdown diretamente na interface
    st.markdown(conteudo_md)

else:
    st.info("ℹ️ Nenhum relatório executivo disponível no momento. Execute o script `gerar_relatorio.py` para gerar uma nova síntese.")