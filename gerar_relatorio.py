import os
import json
import time
from datetime import datetime, timezone, timedelta
from google.cloud import storage
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

CREDS_PATH = os.getenv("GCP_SERVICE_ACCOUNT")
BUCKET_ECONOMICO = "conversas-impacto-economico"
BUCKET_SOCIAL = "conversas-impacto-social"
BUCKET_RELATORIOS = "relatorio-diagnostico"

MODELO = "gemini-3.5-flash-lite"

# Orçamento de entrada (1 token ~ 3 caracteres em português).
# Free tier: 250 mil tokens/min de entrada. Mantemos folga grande.
MAX_CHARS_POR_MODULO = 60_000    # ~20 mil tokens por módulo no prompt final
MAX_CHARS_POR_BLOCO = 40_000     # tamanho de cada bloco quando é preciso pré-resumir
MAX_CHARS_MENSAGEM = 1_500       # corta respostas individuais muito longas
PAUSA_ENTRE_CHAMADAS = 2         # segundos, evita rajada de requisições

PREFIXOS_IGNORADOS = ("Olá! Sou seu assistente", "Erro ao processar consulta")


# ---------------------------------------------------------------- GCS
_client = None

def _storage():
    global _client
    if _client is None:
        _client = storage.Client.from_service_account_json(CREDS_PATH)
    return _client


def obter_todos_jsons_bucket(bucket_name, desde=None):
    """Localiza e baixa os arquivos JSON gravados no bucket.

    Sem `desde` (padrão) lê TODOS, como antes. Com `desde` (datetime com fuso),
    lê só os arquivos atualizados a partir dele, útil para um relatório diário.
    """
    try:
        bucket = _storage().bucket(bucket_name)
        blobs = list(bucket.list_blobs())

        if not blobs:
            print(f"⚠️ Nenhum arquivo encontrado no bucket {bucket_name}")
            return []

        todos_dados = []
        for blob in sorted(blobs, key=lambda b: b.updated or datetime.min.replace(tzinfo=timezone.utc)):
            if not blob.name.endswith(".json"):
                continue
            if desde and blob.updated and blob.updated < desde:
                continue
            print(f"📥 Lendo arquivo: {blob.name} do bucket {bucket_name}")
            try:
                todos_dados.append(json.loads(blob.download_as_text()))
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON inválido em {blob.name}, ignorado: {e}")

        return todos_dados
    except Exception as e:
        print(f"Erro ao buscar dados do bucket {bucket_name}: {e}")
        return []


def salvar_relatorio_no_bucket(bucket_name, file_name, conteudo_md):
    """Salva o relatório Markdown no bucket relatorio-diagnostico."""
    try:
        bucket = _storage().bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_string(conteudo_md, content_type="text/markdown")
        print(f"✅ Relatório salvo com sucesso: {bucket_name}/{file_name}")
        return True
    except Exception as e:
        print(f"Erro ao salvar relatório no bucket: {e}")
        return False


# ---------------------------------------------------------------- preparo dos logs
def _achar_mensagens(obj, saida):
    """Percorre qualquer estrutura JSON e coleta dicts {role, content}."""
    if isinstance(obj, dict):
        if "role" in obj and "content" in obj:
            saida.append((str(obj["role"]), str(obj["content"])))
        else:
            for v in obj.values():
                _achar_mensagens(v, saida)
    elif isinstance(obj, list):
        for v in obj:
            _achar_mensagens(v, saida)


def compactar_conversas(lista):
    """Transforma os JSONs brutos em texto enxuto: só perguntas e respostas úteis.

    Remove saudações automáticas e mensagens de erro, corta respostas gigantes e
    descarta perguntas repetidas (a mesma pergunta feita várias vezes no dia).
    """
    vistos, blocos = set(), []
    for conversa in lista:
        msgs = []
        _achar_mensagens(conversa, msgs)

        if not msgs:  # formato inesperado: mantém o conteúdo, mas compacto
            bruto = json.dumps(conversa, ensure_ascii=False, separators=(",", ":"))
            blocos.append(bruto[:MAX_CHARS_MENSAGEM * 2])
            continue

        pergunta = None
        for role, texto in msgs:
            texto = texto.strip()
            if not texto or texto.startswith(PREFIXOS_IGNORADOS):
                continue
            if len(texto) > MAX_CHARS_MENSAGEM:
                texto = texto[:MAX_CHARS_MENSAGEM] + " […]"
            if role == "user":
                pergunta = texto
            elif pergunta is not None:
                chave = (pergunta.lower(), texto[:200].lower())
                if chave not in vistos:
                    vistos.add(chave)
                    blocos.append(f"P: {pergunta}\nR: {texto}")
                pergunta = None
    return blocos


def _agrupar_em_blocos(itens, limite):
    grupos, atual, tam = [], [], 0
    for it in itens:
        if atual and tam + len(it) > limite:
            grupos.append("\n\n".join(atual))
            atual, tam = [], 0
        atual.append(it)
        tam += len(it) + 2
    if atual:
        grupos.append("\n\n".join(atual))
    return grupos


# ---------------------------------------------------------------- chamada ao modelo
def _gerar(model, prompt, tentativas=4, **cfg):
    """generate_content com retry/backoff em 429 e sem quebrar em resposta bloqueada."""
    for i in range(tentativas):
        try:
            resp = model.generate_content(
                prompt, generation_config=genai.GenerationConfig(**cfg) if cfg else None
            )
            return resp.text
        except ValueError:
            raise RuntimeError("O modelo não devolveu texto (resposta bloqueada ou vazia).")
        except Exception as e:
            if ("429" in str(e) or "503" in str(e)) and i < tentativas - 1:
                espera = 2 ** (i + 2)
                print(f"⏳ Limite/instabilidade da API, tentando de novo em {espera}s…")
                time.sleep(espera)
                continue
            raise


def preparar_modulo(model, itens, nome_modulo):
    """Devolve o texto do módulo dentro do orçamento.

    Cabe no orçamento -> usa as conversas como estão (nada é perdido).
    Estourou        -> resume cada bloco preservando números e, se preciso, resume de novo.
    """
    if not itens:
        return "Sem conversas registradas."

    texto = "\n\n".join(itens)
    if len(texto) <= MAX_CHARS_POR_MODULO:
        return texto

    print(f"📦 {nome_modulo}: {len(texto):,} caracteres, pré-resumindo em blocos…")
    while len(texto) > MAX_CHARS_POR_MODULO:
        resumos = []
        for bloco in _agrupar_em_blocos(texto.split("\n\n"), MAX_CHARS_POR_BLOCO):
            prompt = f"""
Você condensa registros de perguntas e respostas de um assistente sobre desastres climáticos
no Brasil ({nome_modulo}).

REGRAS:
1. Preserve TODOS os números (R$, quantidades, percentuais), nomes de municípios, UFs, regiões,
   tipos de desastre e períodos exatamente como aparecem. Não arredonde nem invente.
2. Una informações repetidas e remova conversa e cortesia.
3. Saída: tópicos curtos "- fato com números". No máximo {MAX_CHARS_POR_BLOCO // 4} caracteres.

REGISTROS:
{bloco}
"""
            resumos.append(_gerar(model, prompt, temperature=0.1))
            time.sleep(PAUSA_ENTRE_CHAMADAS)
        novo = "\n".join(resumos)
        if len(novo) >= len(texto):  # não encolheu: corta para garantir o orçamento
            novo = novo[:MAX_CHARS_POR_MODULO]
        texto = novo
    return texto


# ---------------------------------------------------------------- relatório
def gerar_diagnostico_executivo(desde=None):
    """Gera o diagnóstico. `desde=None` mantém o comportamento original (todos os logs).

    Para um relatório só do dia: gerar_diagnostico_executivo(
        desde=datetime.now(timezone.utc) - timedelta(days=1))
    """
    print("⏳ Gerando relatório consolidado do fim do dia a partir dos logs...")

    api_key = os.getenv("GEMINI_API_KEY_VIDTEK") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Defina GEMINI_API_KEY_VIDTEK ou GEMINI_API_KEY.")
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODELO)

    lista_econ = obter_todos_jsons_bucket(BUCKET_ECONOMICO, desde)
    lista_soc = obter_todos_jsons_bucket(BUCKET_SOCIAL, desde)

    itens_econ = compactar_conversas(lista_econ)
    itens_soc = compactar_conversas(lista_soc)

    # Sem nenhuma conversa não há o que diagnosticar: evita relatório inventado.
    if not itens_econ and not itens_soc:
        print("⚠️ Nenhuma conversa aproveitável nos buckets. Relatório não gerado.")
        return None

    str_econ = preparar_modulo(model, itens_econ, "módulo econômico")
    str_soc = preparar_modulo(model, itens_soc, "módulo social")

    timestamp_agora = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_formatada = datetime.now().strftime('%d/%m/%Y %H:%M')

    prompt_template = f"""
    Você é o Diretor Executivo de Análise Climática e Risco do Observatório do Clima.
    Sua tarefa é gerar um DIAGNÓSTICO DE IMPACTO CLIMÁTICO CONSOLIDADO com base na TOTALIDADE de análises e perguntas feitas ao assistente de IA durante o dia em todas as seções.

    CONTEXTO DOS DADOS: base do Atlas Digital de Desastres (S2iD/MDR), valores monetários
    corrigidos pelo IGP-DI (base dez/2022), dano autodeclarado pelos municípios.
    Prejuízo total = dano material + prejuízo público + prejuízo privado.

    DADOS DE TODAS AS CONVERSAS DO MÓDULO ECONÔMICO (conversas-impacto-economico):
    {str_econ}

    DADOS DE TODAS AS CONVERSAS DO MÓDULO SOCIAL (conversas-impacto-social):
    {str_soc}

    REGRAS DE CONTEÚDO (obrigatórias):
    1. Use SOMENTE números, municípios, estados e tipos de desastre que aparecem nas conversas
       acima. Nunca invente, estime ou complete valores. Se a informação não existir, escreva
       "Não houve consulta sobre este ponto" em vez de preencher.
    2. Mantenha os valores exatamente como nas conversas; formate dinheiro como R$ 1.234.567,89
       ou abreviado (R$ 1,2 bi), sempre com a unidade.
    3. Quando a mesma métrica aparecer com valores diferentes (filtros ou períodos distintos),
       cite o recorte (ex.: "no Nordeste", "entre 2015 e 2023") ou aponte a divergência.
    4. Ordene destaques do maior para o menor impacto e dê preferência a números concretos
       em vez de adjetivos.
    5. Valores autodeclarados: não os trate como gasto efetivamente executado em reconstrução.
    6. Recomendações: exatamente 3, cada uma ligada a um achado específico acima e com ação concreta.
    7. Se um módulo estiver sem conversas, diga isso na seção correspondente e não extrapole.
    8. Português do Brasil, tom técnico e direto, sem repetir o mesmo dado em seções diferentes.

    Estrutura obrigatória do relatório (em Markdown):

    # 📋 Diagnóstico Executivo de Impacto Climático
    **Data de Emissão:** {data_formatada}
    **Status:** Síntese Diária Consolidada

    ---

    ## 🎯 1. Resumo Executivo
    (Um parágrafo claro sintetizando os maiores destaques e emergências identificadas nas consultas do dia).

    ## 💰 2. Destaques de Impacto Econômico
    - **Perdas Financeiras Significativas:** (Citar os principais números e categorias discutidas).
    - **Setores mais Vulneráveis:** (Citar os setores do prejuízo privado e serviços públicos mais atingidos).
    (Parágrafo de visão geral com o prejuízo total e a divisão entre dano material, prejuízo público
    e prejuízo privado, quando citados.)
 
    ### 2.1 Recortes
    - **Por região:** (...)
    - **Por estado:** (...)
    - **Por município:** (...)
    - **Ao longo do tempo:** (...)
    - **Por tipo de desastre:** (...)
    - **Por setor:** (setores do prejuízo privado e serviços públicos mais atingidos, se citados.)
 

    ## 👥 3. Destaques de Impacto Social e Humano
    - **Populações em Risco:** (Citar achados sobre mortalidade, feridos e afetados).
    - **Crise de Deslocamento:** (Resumo dos dados sobre desabrigados e desalojados).
    (Parágrafo de visão geral: mortos, feridos, afetados, desabrigados e desalojados, quando citados.)
 
    ### 3.1 Recortes
    - **Por região:** (...)
    - **Por estado:** (...)
    - **Por município:** (...)
    - **Ao longo do tempo:** (...)
    - **Por tipo de desastre:** (...

     ## ✅ 4. Conclusão
    (Síntese dos principais achados econômicos e sociais, o que significam em conjunto.)

    ## 💡 5. Recomendações e Próximos Passos
    (3 recomendações estratégicas baseadas nas descobertas).

    ---
    *Relatório gerado automaticamente via Observatório do Clima IA Engine.*

    Responda apenas com o relatório em Markdown, sem texto antes ou depois.
    """

    try:
        relatorio_md = _gerar(model, prompt_template, temperature=0.2, max_output_tokens=4096)
    except Exception as e:
        print(f"❌ Falha ao gerar o relatório: {e}")
        return None

    nome_arquivo_relatorio = f"diagnostico_executivo_{timestamp_agora}.md"
    salvar_relatorio_no_bucket(BUCKET_RELATORIOS, nome_arquivo_relatorio, relatorio_md)
    return relatorio_md


if __name__ == "__main__":
    gerar_diagnostico_executivo()