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

# Colunas cujo nome indica valor em R$ (ajuste se os nomes do seu df forem outros)
REGEX_DINHEIRO = r"preju|dano.?s?_?material|dm_total|pepl|pepr|pe_pl|valor|\(r\$\)|custo"

# ---------------------------------------------------------------- VERDADES (fixas)
# Fatos conferidos da base completa. Entram em TODO prompt de redação como referência de escala.
# CONFIRME que valem para a base completa (sem filtro) e complete com o que conhecer do app.
VERDADES_FIXAS = """
VERDADES DA BASE (referência de escala; valem para a base completa, sem filtro de período):
- Prejuízo total nacional: R$ 444.248.102.989,92 (R$ 444,25 bilhões).
- Por região: Sul R$ 186,46 bilhões; Nordeste R$ 134,94 bilhões; Sudeste R$ 72,13 bilhões;
  Centro-Oeste R$ 31,19 bilhões; Norte R$ 19,53 bilhões (as cinco somam o total nacional).
- Todos os valores monetários estão em reais (R$). 1 bilhão = 1.000.000.000 (9 zeros).
- Nenhum valor de região, estado, município ou tipo de desastre pode ser maior que o total nacional;
  portanto "trilhão" NUNCA se aplica a valores desta base.
"""


def obter_api_key():
    api_key = os.getenv("GEMINI_API_KEY_VIDTEK")
    if api_key:
        return api_key
    try:
        if "GEMINI_API_KEY_VIDTEK" in st.secrets:
            return st.secrets["GEMINI_API_KEY_VIDTEK"]
        elif "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY_VIDTEK"]
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------- formatação pt-BR (causa do erro do "trilhão")
# O to_string() do pandas imprime 186456900000.0 como 1.864569e+11, e o Gemini lê o expoente errado.
# Aqui o CÓDIGO formata: nunca vai notação científica para o modelo.
def _num(v, casas=2):
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _dinheiro(v):
    if pd.isna(v):
        return "—"
    txt = f"R$ {_num(v)}"
    for lim, div, sufixo in ((1e12, 1e12, "tri"), (1e9, 1e9, "bi"), (1e6, 1e6, "mi")):
        if abs(v) >= lim:
            return f"{txt} (R$ {_num(v / div)} {sufixo})"
    return txt


def formatar_resultado(res, limite=MAX_LINHAS) -> str:
    if isinstance(res, (int, float)) and not isinstance(res, bool):
        return _num(res, 0 if float(res).is_integer() else 2)
    if isinstance(res, pd.Series):
        df = res.to_frame(name=res.name if res.name is not None else "valor")
    elif isinstance(res, pd.DataFrame):
        df = res.copy()
    else:
        return str(res)[:2000]

    total = len(df)
    df = df.head(limite)
    df.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            if re.search(r"(^|_)(ano|year|mes)s?($|_)", c, re.I):  # ano/mês: sem separador de milhar
                out[c] = col.map(lambda x: "—" if pd.isna(x) else str(int(x)))
            elif re.search(REGEX_DINHEIRO, c, re.I):
                out[c] = col.map(_dinheiro)
                if "count" in df.index.astype(str):  # describe(): 'count' é contagem, não R$
                    out.loc["count", c] = _num(col.loc["count"], 0)
            elif col.dropna().map(lambda x: float(x).is_integer()).all():
                out[c] = col.map(lambda x: "—" if pd.isna(x) else _num(x, 0))
            else:
                out[c] = col.map(lambda x: "—" if pd.isna(x) else _num(x))
        else:
            out[c] = col
    txt = out.to_string()
    if total > limite:
        txt += f"\n[... mostrando {limite} de {total} linhas]"
    return txt


def verdades_do_df(df: pd.DataFrame) -> str:
    """Total e quebra por região/tipo calculados AGORA no df da seção (respeita filtros)."""
    cols = {c.lower(): c for c in df.columns}
    val = next((cols[k] for k in ("prejuizo_total", "pe_plepr", "prejuizo") if k in cols), None)
    if val is None or not pd.api.types.is_numeric_dtype(df[val]):
        return ""
    linhas = [f"VERDADES CALCULADAS AGORA NO FILTRO ATUAL (use como referência de escala):",
              f"- {val}, soma do filtro atual: {_dinheiro(df[val].sum())} em {_num(len(df), 0)} registros."]
    for rot, cands in (("região", ("regiao", "região")), ("tipo de desastre", ("descricao_tipologia", "tipologia"))):
        col = next((cols[k] for k in cands if k in cols), None)
        if col:
            g = df.groupby(col)[val].sum().nlargest(6)
            linhas.append(f"- Por {rot}: " + "; ".join(f"{k} {_dinheiro(v)}" for k, v in g.items()) + ".")
    return "\n".join(linhas)


# ---------------------------------------------------------------- chamada ao modelo
def _gerar(model, prompt: str, tentativas: int = 3, **cfg) -> str:
    """generate_content com retry/backoff para 429/503."""
    for i in range(tentativas):
        try:
            if cfg:
                return model.generate_content(prompt, generation_config=genai.GenerationConfig(**cfg)).text
            return model.generate_content(prompt).text
        except Exception as e:
            if ("429" in str(e) or "503" in str(e)) and i < tentativas - 1:
                time.sleep(2 ** (i + 1))
                continue
            raise


# ---------------------------------------------------------------- esquema enxuto
def montar_esquema(df: pd.DataFrame) -> str:
    """Colunas, tipos e valores de exemplo. Tamanho fixo, não cresce com a base."""
    linhas = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            ex = df[c].dropna().astype(str).value_counts().head(5).index.tolist()
            linhas.append(f"- {c} (texto) ex.: {ex}")
        else:
            if re.search(REGEX_DINHEIRO, str(c), re.I):
                linhas.append(f"- {c} ({df[c].dtype}) [valor em R$] min={_num(df[c].min())} max={_num(df[c].max())}")
            else:  # ano, contagens etc.: sem separador de milhar (evita 'ano 2.019')
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
6. NUNCA divida, multiplique ou converta valores monetários (nada de /1e9, /1000, *100):
   devolva o valor em reais como está. A formatação é feita pelo sistema.
7. NUNCA renomeie colunas de valor (nada de agg(total=...) ou .rename): mantenha o nome original
   (ex.: df.groupby('regiao')['prejuizo_total'].sum()), pois é o nome que identifica o valor em R$.
"""
    bruto = _gerar(model, prompt)
    bruto = re.sub(r"^```(?:json)?|```$", "", bruto.strip(), flags=re.M).strip()
    consultas = json.loads(bruto)
    return [c for c in consultas if "codigo" in c][:MAX_CONSULTAS]


# ---------------------------------------------------------------- execução local
# Divisão por constante de escala (/1e9, /1000, .div(1000000)) muda a ordem de grandeza: proibido.
_RE_ESCALA = re.compile(r"(?:/|\.div\(|\.floordiv\()\s*\(?\s*(?:1e\d+|10\s*\*\*\s*\d+|\d[\d_]{3,})", re.I)


def executar_query_local(codigo: str, df: pd.DataFrame):
    if _RE_ESCALA.search(codigo):
        return None, "conversão de escala não é permitida"
    try:
        resultado = eval(codigo, {"__builtins__": {}}, {"df": df, "pd": pd})
        return resultado, None
    except Exception as e:
        return None, str(e)


def resumir_para_prompt(resultado, limite_linhas: int = MAX_LINHAS) -> str:
    return formatar_resultado(resultado, limite_linhas)


# ---------------------------------------------------------------- síntese
def sintetizar_resposta(model, pergunta: str, blocos: list[tuple], nome_tab: str, verdades_df: str = "") -> str:
    partes, usado = [], 0
    for titulo, texto in blocos:
        texto = texto[: max(0, MAX_CHARS_TOTAL - usado)]
        usado += len(texto)
        partes.append(f"### {titulo}\n{texto}")

    prompt = f"""
Você é um especialista em economia do clima do Observatório do Clima.
Pergunta do usuário na seção '{nome_tab}': "{pergunta}"

{VERDADES_FIXAS}
{verdades_df}

Resultados exatos calculados na base (use somente estes números, não invente):
{chr(10).join(partes)}

REGRAS DE NÚMEROS (inegociáveis):
1. Copie os valores LITERALMENTE dos resultados. Cada valor monetário vem como
   "R$ valor exato (R$ valor abreviado)": use um dos dois, sem alterar dígitos.
2. NUNCA converta unidades, some, subtraia, estime ou calcule valores ou percentuais.
   Para comparar, cite os dois valores como estão.
3. 1 bilhão = 1.000.000.000. NUNCA escreva "trilhão": nenhum valor desta base chega a isso e
   nenhuma parte pode superar o total nacional das VERDADES.
4. Ao abreviar, arredonde na segunda casa (exemplo ilustrativo: R$ 1.234.567.890,12 = R$ 1,23 bilhão).
5. Nunca use notação científica. Milhares com ponto, decimais com vírgula.
6. Associe cada valor ao seu rótulo (região, estado, município, tipo) exatamente como na linha do
   resultado. Não troque valores entre linhas.
7. Se um dado não estiver nos resultados, diga que não foi consultado. Não preencha.
8. Se os resultados do filtro atual diferirem das VERDADES da base completa, é porque há filtro
   de período: use os resultados e diga que se refere ao filtro atual.

FORMATO:
1. Português do Brasil, tom técnico e direto.
2. Se a pergunta for de diagnóstico, organize por seções (uma por dimensão) e termine com uma conclusão.
3. Se algum bloco indicar "[... mostrando N de M linhas]", diga que é um recorte (top N).
"""
    return _gerar(model, prompt, temperature=0.1)


def responder_pergunta(model, pergunta: str, df_secao: pd.DataFrame, nome_tab: str) -> str:
    consultas = planejar_consultas(model, pergunta, montar_esquema(df_secao))

    blocos, falhas = [], []
    for c in consultas:
        res, erro = executar_query_local(c["codigo"], df_secao)
        if erro:
            falhas.append(c.get("titulo", c["codigo"]))
        else:
            blocos.append((c.get("titulo", "Resultado"), resumir_para_prompt(res)))

    if not blocos:
        return ("Não consegui montar uma consulta válida para essa pergunta. "
                "Tente reformular de forma mais específica (ex.: 'top 10 municípios por prejuízo total').")

    resposta = sintetizar_resposta(model, pergunta, blocos, nome_tab, verdades_do_df(df_secao))
    if falhas:
        resposta += "\n\n_Não consegui calcular: " + "; ".join(falhas) + "._"
    return resposta


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
                    resposta = responder_pergunta(model, prompt_user, df_secao, nome_tab)
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
                    # o markdown do Streamlit trata $...$ como fórmula e apaga o "R$": escapa só na exibição
                    st.markdown(msg["content"].replace("\\$", "$").replace("$", "\\$"))