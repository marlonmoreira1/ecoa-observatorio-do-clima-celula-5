import json
import os
import re
import time

import pandas as pd
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from utils import salvar_historico_no_bucket

load_dotenv()

MODELO = "gemini-3.5-flash-lite"
MAX_LINHAS = 100            # linhas por resultado enviadas ao modelo
MAX_CHARS_TOTAL = 16_000   # ~5 mil tokens de dados por resposta, no total
MAX_CONSULTAS = 10          # consultas por pergunta (diagnóstico)


def obter_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key
    try:
        if "GEMINI_API_KEY_VIDTEK" in st.secrets:
            return st.secrets["GEMINI_API_KEY_VIDTEK"]
        elif "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------- chamada ao modelo
def _gerar(model, prompt: str, tentativas: int = 3) -> str:
    """generate_content com retry/backoff para 429."""
    for i in range(tentativas):
        try:
            return model.generate_content(prompt).text
        except Exception as e:
            if "429" in str(e) and i < tentativas - 1:
                time.sleep(2 ** (i + 1))
                continue
            raise


# ---------------------------------------------------------------- esquema enxuto
def montar_esquema(df: pd.DataFrame) -> str:
    """Colunas, tipos e valores de exemplo. Tamanho fixo, não cresce com a base."""
    linhas = []
    for c in df.columns:
        if df[c].dtype == "object":
            ex = df[c].dropna().astype(str).value_counts().head(5).index.tolist()
            linhas.append(f"- {c} (texto) ex.: {ex}")
        else:
            linhas.append(f"- {c} ({df[c].dtype}) min={df[c].min()} max={df[c].max()}")
    return "\n".join(linhas)


# ---------------------------------------------------------------- planejamento
def planejar_consultas(model, pergunta: str, esquema: str) -> list[dict]:
    prompt = f"""
Você gera código Pandas. O DataFrame se chama 'df' e tem estas colunas:
{esquema}

Pergunta do usuário: "{pergunta}"

Responda SOMENTE com JSON válido: uma lista de até {MAX_CONSULTAS} objetos
{{"titulo": "...", "codigo": "..."}}.

REGRAS:
1. Cada "codigo" é UMA expressão Pandas de uma linha sobre 'df' (sem atribuições, sem import).
2. SEMPRE agregue (groupby + sum/count/mean). Nunca devolva linhas brutas de 'df'.
3. SEMPRE limite o tamanho: use .nlargest(10) ou .head(10) em agrupamentos com muitas categorias
   (município, estado). Agrupamentos pequenos (região, tipologia) podem ir completos.
4. Pergunta simples = 1 consulta. Pergunta de diagnóstico/visão geral = várias, uma por
   dimensão pedida (região, UF, município, tempo, tipologia).
5. Use só colunas que existem na lista acima.
"""
    bruto = _gerar(model, prompt)
    bruto = re.sub(r"^```(?:json)?|```$", "", bruto.strip(), flags=re.M).strip()
    consultas = json.loads(bruto)
    return [c for c in consultas if "codigo" in c][:MAX_CONSULTAS]


# ---------------------------------------------------------------- execução local
def executar_query_local(codigo: str, df: pd.DataFrame):
    try:
        resultado = eval(codigo, {"__builtins__": {}}, {"df": df, "pd": pd})
        return resultado, None
    except Exception as e:
        return None, str(e)


def resumir_para_prompt(resultado, limite_linhas: int = MAX_LINHAS) -> str:
    """Nunca devolve mais que `limite_linhas` linhas, e avisa que cortou."""
    if isinstance(resultado, (pd.DataFrame, pd.Series)):
        total = len(resultado)
        txt = resultado.head(limite_linhas).to_string()
        if total > limite_linhas:
            txt += f"\n[... mostrando {limite_linhas} de {total} linhas]"
        return txt
    return str(resultado)[:2000]


# ---------------------------------------------------------------- síntese
def sintetizar_resposta(model, pergunta: str, blocos: list[tuple], nome_tab: str) -> str:
    partes, usado = [], 0
    for titulo, texto in blocos:
        texto = texto[: max(0, MAX_CHARS_TOTAL - usado)]
        usado += len(texto)
        partes.append(f"### {titulo}\n{texto}")

    prompt = f"""
Você é um especialista em economia do clima do Observatório do Clima.
Pergunta do usuário na seção '{nome_tab}': "{pergunta}"

Resultados exatos calculados na base (use somente estes números, não invente):
{chr(10).join(partes)}

FORMATO:
1. Português do Brasil, tom técnico e direto.
2. Valores em R$ com separador de milhar (ex.: R$ 1.234.567,89) ou abreviados (R$ 1,2 bi).
3. Se a pergunta for de diagnóstico, organize por seções (uma por dimensão) e termine com uma conclusão.
4. Se algum bloco indicar "[... mostrando N de M linhas]", diga que é um recorte (top N).
"""
    return _gerar(model, prompt)


@st.cache_resource
def obter_modelo_gemini(api_key: str):
    """Mantém a instância do modelo Gemini em memória para todas as sessões."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODELO)

# ---------------------------------------------------------------- UI
def renderizar_chat_ia(chave_tab: str, nome_tab: str, df_secao: pd.DataFrame,
                       bucket_destinatario: str = None, prefixo_arquivo: str = None):
    with st.expander(f"🤖 Chat Assistente IA — Interpretação dos Dados de {nome_tab}", expanded=True):
        api_key = obter_api_key()
        if not api_key:
            st.info("💡 Adicione a chave `GEMINI_API_KEY_VIDTEK` no seu arquivo `.env` ou `.streamlit/secrets.toml`.")
            return

        genai.configure(api_key=api_key)
        model = obter_modelo_gemini(api_key)
        key_historico = f"chat_messages_{chave_tab}"

        if key_historico not in st.session_state:
            st.session_state[key_historico] = [{
                "role": "assistant",
                "content": f"Olá! Sou seu assistente focado na seção **{nome_tab}**. "
                           "Posso consultar cidades, estados, regiões, anos e tipos de desastre. O que gostaria de saber?",
            }]

        chat_container = st.container()

        if prompt_user := st.chat_input(f"Pergunte sobre qualquer dado de {nome_tab}...", key=f"chat_input_{chave_tab}"):
            st.session_state[key_historico].append({"role": "user", "content": prompt_user})

            with st.spinner("🎲 Consultando base de dados..."):
                try:
                    esquema = montar_esquema(df_secao)
                    consultas = planejar_consultas(model, prompt_user, esquema)

                    blocos, falhas = [], []
                    for c in consultas:
                        res, erro = executar_query_local(c["codigo"], df_secao)
                        if erro:
                            falhas.append(c.get("titulo", c["codigo"]))
                        else:
                            blocos.append((c.get("titulo", "Resultado"), resumir_para_prompt(res)))

                    if not blocos:
                        resposta = ("Não consegui montar uma consulta válida para essa pergunta. "
                                    "Tente reformular de forma mais específica (ex.: 'top 10 municípios por prejuízo total').")
                    else:
                        resposta = sintetizar_resposta(model, prompt_user, blocos, nome_tab)
                        if falhas:
                            resposta += "\n\n_Não consegui calcular: " + "; ".join(falhas) + "._"

                    st.session_state[key_historico].append({"role": "assistant", "content": resposta})

                    if bucket_destinatario and prefixo_arquivo:
                        salvar_historico_no_bucket(
                            bucket_name=bucket_destinatario,
                            file_prefix=prefixo_arquivo,
                            historico=st.session_state[key_historico],
                            modulo_nome=nome_tab,
                        )
                except Exception as e:
                    st.session_state[key_historico].append(
                        {"role": "assistant", "content": f"Erro ao processar consulta: {e}"}
                    )

        with chat_container:
            for msg in st.session_state[key_historico]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])