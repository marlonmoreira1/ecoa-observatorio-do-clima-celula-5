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

MODELO = "gemini-3.5-flash"

# Orçamento de entrada
MAX_CHARS_POR_MODULO = 60_000
MAX_CHARS_POR_BLOCO = 40_000
MAX_CHARS_MENSAGEM = 1_500
PAUSA_ENTRE_CHAMADAS = 2
DEBUG = os.getenv("DEBUG_RELATORIO") == "1"

PREFIXOS_IGNORADOS = ("Olá! Sou seu assistente", "Erro ao processar consulta")

VERDADES = """
VERDADES DA BASE (conferidas; são a referência de escala):
- Prejuízo total nacional: R$ 444.248.102.989,92 (R$ 444,25 bilhões).
- Prejuízo total por região: Sul R$ 186,46 bilhões; Nordeste R$ 134,94 bilhões; Sudeste R$ 72,13 bilhões;
  Centro-Oeste R$ 31,19 bilhões; Norte R$ 19,53 bilhões (a soma das cinco dá o total nacional).
- Todos os valores estão em reais (R$). 1 bilhão = 1.000.000.000 (9 zeros).
- Nenhum valor de região, estado, município ou tipo de desastre pode ser maior que o total nacional.
  Portanto a palavra "trilhão" NUNCA se aplica a valores desta base.
"""

_client = None

def _storage():
    global _client
    if _client is None:
        _client = storage.Client.from_service_account_json(CREDS_PATH)
    return _client

def obter_todos_jsons_bucket(bucket_name, desde=None):
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
    try:
        bucket = _storage().bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_string(conteudo_md, content_type="text/markdown")
        print(f"✅ Relatório salvo com sucesso: {bucket_name}/{file_name}")
        return True
    except Exception as e:
        print(f"Erro ao salvar relatório no bucket: {e}")
        return False

def _achar_mensagens(obj, saida):
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
    vistos, blocos = set(), []
    for conversa in lista:
        msgs = []
        _achar_mensagens(conversa, msgs)

        if not msgs:
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

def _gerar(model, prompt, tentativas=4, **cfg):
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

def _resumir_bloco(model, bloco, nome_modulo):
    prompt = f"""
Você condensa registros de perguntas e respostas de um assistente sobre desastres climáticos
no Brasil ({nome_modulo}).

REGRAS:
1. COPIE todos os números (R$, quantidades, percentuais) exatamente como estão, com os mesmos dígitos
   e a mesma unidade ("bilhões" continua "bilhões"). NUNCA converta unidades, some, arredonde ou calcule.
2. Nenhum valor desta base chega a trilhão; se algo parecer isso, mantenha o texto original do registro.
3. Preserve nomes de municípios, UFs, regiões, tipos de desastre e períodos como aparecem.
4. Una informações repetidas e remova conversa e cortesia.
5. Saída: tópicos curtos "- fato com números". No máximo {MAX_CHARS_POR_BLOCO // 4} caracteres.

REGISTROS:
{bloco}
"""
    return _gerar(model, prompt, temperature=0.0)

def preparar_modulo(model, itens, nome_modulo):
    if not itens:
        return "Sem conversas registradas."

    texto = "\n\n".join(itens)
    if len(texto) <= MAX_CHARS_POR_MODULO:
        return texto

    print(f"📦 {nome_modulo}: {len(texto):,} caracteres, pré-resumindo em blocos…")
    while len(texto) > MAX_CHARS_POR_MODULO:
        resumos = []
        for bloco in _agrupar_em_blocos(texto.split("\n\n"), MAX_CHARS_POR_BLOCO):
            resumos.append(_resumir_bloco(model, bloco, nome_modulo))
            time.sleep(PAUSA_ENTRE_CHAMADAS)
        novo = "\n".join(resumos)
        if len(novo) >= len(texto): 
            novo = novo[:MAX_CHARS_POR_MODULO]
        texto = novo
    return texto

# ---------------------------------------------------------------- relatório

RELATORIO_BASE = """# **DIAGNÓSTICO EXECUTIVO INTEGRADO DE IMPACTO CLIMÁTICO NO BRASIL**

**Impactos econômicos, sociais, territoriais e por tipologia de desastre**

**Base:** BD_Atlas_Final.csv | 24.617 registros | Eventos de 02/01/2013 a 31/12/2025

**Objetivo.** Consolidar a análise dos prejuízos econômicos, danos materiais e impactos humanos registrados na base, cruzando tipologia de desastre com tempo, setor, serviço público, região, UF e município.

**Nota monetária.** Os valores são tratados como corrigidos pela inflação conforme a versão CSV do Atlas Digital de Desastres. A documentação oficial informa que a versão CSV contém os valores corrigidos e que a atualização monetária usa o IGP-DI com valores de referência em dezembro de 2022. Portanto, os montantes deste relatório estão em reais corrigidos segundo a base, não em preços atualizados. Fonte metodológica: Atlas Digital de Desastres, MIDR.

Data de corte. O arquivo contém registros até 31/12/2025; não representa o ano inteiro de 2026.

Definição dos horizontes. Histórico: todos os eventos da base. Últimos dez anos: eventos a partir de 22/10/2016 (primeiro evento observado nesse recorte: 23/10/2016). **Recente:** eventos a partir de 01/01/2020 (primeiro evento observado: 02/01/2020).

# **1. Resumo Executivo**

A análise dos 24.617 registros da base revela uma exposição econômica e social expressiva aos desastres no Brasil. A base reúne registros reconhecidos, referentes a eventos entre 2013 e 2025. Os prejuízos econômicos totais (incluindo danos materiais, prejuízos públicos e privados) somam R$ 444,25 bi no histórico, dos quais R$ 279,42 bi (62,9%) foram registrados desde 2020, evidenciando uma intensificação na frequência e severidade no curto prazo. Os danos materiais compõem isoladamente R$ 64,35 bi desse total, com R$ 40,05 bi desde 2020. O total de danos humanos alcança 207.201.897 registros no histórico, com 119.173.784 (57,5%) concentrados desde 2020. Foram registradas 1.776 mortes no histórico, incluindo 1.242 desde 2020 (69,9%).

O perfil de impacto é distinto por tipologia. O Brasil sofre duas crises paralelas e com perfis opostos: de um lado, a estiagem e seca, agindo de forma silenciosa e prolongada, lideram os prejuízos econômicos, com R$ 287,31 bilhões no histórico e R$ 183,81 bilhões desde 2020, fortemente concentrados na agricultura e pecuária, impactando a economia rural e o abastecimento hídrico. Do outro, chuvas intensas lideram mortes, desaparecimentos, feridos, deslocamentos e danos materiais nos centros urbanos, com 1.021 mortes e R$ 30,26 bilhões em danos materiais desde 2020.

Territorialmente, o Sul concentra a maior parcela do prejuízo econômico recente, sobretudo pela combinação de estiagem/seca e chuvas intensas; o Nordeste apresenta forte exposição histórica à seca; e o Sudeste concentra a maior mortalidade histórica e recente. O Rio Grande do Sul lidera prejuízos e deslocamentos acumulados; o Rio de Janeiro lidera mortes.

Interpretação: os registros demonstram impactos humanos e econômicos substanciais e uma concentração expressiva desde 2020. A base, isoladamente, não permite atribuir causalidade climática a cada ocorrência nem comprovar tendência anual sem ajustar por exposição, cobertura e número de registros, mas a aceleração pós-2020 confirma que a matriz de risco do país teve alteração estrutural.

## **Principais Indicadores Acumulados**

| Indicador de Impacto | Histórico (2013–2025) | Desde 2020 (Recente) | Proporção Recente (%) |
| :---- | :---- | :---- | :---- |
| **Prejuízo Econômico Total** | R$ 444,25 bi | R$ 279,42 bi | 62,9% |
| **Prejuízo Privado** | R$ 341,41 bi | R$ 225,12 bi | 65,9% |
| **Prejuízo Público** | R$ 38,49 bi | R$ 14,11 bi | 36,7% |
| **Danos Materiais** | R$ 64,35 bi | R$ 40,05 bi | 62,2% |
| **Total de Danos Humanos** | 207.201.897 | 119.173.784 | 57,5% |
| **Vítimas Fatais (Mortes)** | 1.776 | 1.242 | 69,9% |
| **Desaparecidos** | 708 | 615 | 86,9% |
| **Desabrigados + Desalojados** | 5.260.447 | 3.464.106 | 65,9% |

## **Escopo, conceitos e cautelas metodológicas**

* **Unidade de análise:** registro de desastre municipal presente no arquivo. Registros não equivalem necessariamente a eventos meteorológicos independentes; eventos correlacionados ou declarações administrativas podem ser representados separadamente.  
* **Prejuízo econômico total:** soma direta do prejuízo privado, prejuízo público e dano material. Danos materiais, prejuízos privados e públicos também são apresentados separadamente na base para detalhamento de composição.  
* **Valores monetários:** mantidos conforme a versão corrigida da base (IGP-DI, ref. dez/2022).  
* **Contagens humanas:** são somas dos registros por campo, não contagem deduplicada de pessoas únicas. Pessoas podem aparecer em eventos diferentes e campos diferentes.  
* **Total de danos humanos:** soma dos campos *DH_total_danos_humanos_diretos* e *DH_OUTROS AFETADOS*. Não devem ser somados às demais categorias individuais para evitar dupla contagem.  
* **Campos ambientais:** a base inclui indicadores de poluição/contaminação, diminuição/exaurimento hídrico e incêndios em áreas protegidas, mas os campos numéricos analisados aparecem sem valores positivos no arquivo (tratados como ausência de preenchimento, não prova de ausência de impacto).

# **2. Módulo 1: Impacto Econômico**

## **2.1 Visão geral dos prejuízos**

A escalada dos danos econômicos é evidente. Enquanto a última década concentra o núcleo dos eventos mais danosos, a partir de 2020 a aceleração dos prejuízos amplia a curva acumulada de perdas. O prejuízo privado responde pela maior fatia histórica (89,9% no histórico e 94,1% desde 2020 se considerarmos a relação público e privado) e recente do prejuízo econômico total. Isso reflete a predominância das perdas declaradas nos setores privados, em particular agricultura e pecuária.

* **Prejuízo econômico total:** Histórico: R$ 444,25 bi | Últimos 10 anos: R$ 363,11 bi | Desde 2020: R$ 279,42 bi (62,9%)  
* **Prejuízo privado:** Histórico: R$ 341,41 bi | Últimos 10 anos: R$ 287,05 bi | Desde 2020: R$ 225,12 bi (65,9%)  
* **Prejuízo público:** Histórico: R$ 38,49 bi | Últimos 10 anos: R$ 26,49 bi | Desde 2020: R$ 14,11 bi (36,7%)  
* **Danos materiais:** Histórico: R$ 64,35 bi | Últimos 10 anos: R$ 49,40 bi | Desde 2020: R$ 40,05 bi (62,2%)

## **2.2 Detalhamento setorial e serviços**

O impacto não ocorre de forma homogênea, afetando diretamente cadeias de suprimento e suporte populacional. O agronegócio absorve a parcela mais substancial das perdas informadas.

| Setores Privados Atingidos | Histórico (R$) | Desde 2020 (R$) |
| :---- | :---- | :---- |
| **1. Agricultura** | R$ 238,91 bi | R$ 170,12 bi |
| **2. Pecuária** | R$ 74,44 bi | R$ 42,56 bi |
| **3. Comércio** | R$ 12,97 bi | R$ 7,19 bi |
| **4. Indústria** | R$ 8,77 bi | R$ 3,43 bi |
| **5. Serviços** | R$ 6,32 bi | R$ 1,82 bi |

A infraestrutura básica e logística registram comprometimentos operacionais relevantes durante as ocorrências. A concentração de perdas em água e mobilidade aponta esses setores como eixos centrais para estratégias de adaptação.

| Serviços Públicos Afetados | Histórico (R$) | Desde 2020 (R$) |
| :---- | :---- | :---- |
| **1. Abastecimento de água** | R$ 21,01 bi | R$ 7,08 bi |
| **2. Transporte** | R$ 8,56 bi | R$ 2,94 bi |
| **3. Esgotamento sanitário** | R$ 2,56 bi | R$ 1,01 bi |
| **4. Assistência médica/emergencial** | R$ 2,00 bi | R$ 984,22 mi |
| **5. Limpeza urbana/coleta de lixo** | R$ 1,68 bi | R$ 948,08 mi |

## **2.3 Recortes de Impacto Econômico**

O perfil financeiro varia conforme a tipologia do evento. A partir de 2020, a maior parte dos valores registrados esteve associada a dois grupos principais: seca e estiagem respondem por aproximadamente 64% do prejuízo econômico histórico, enquanto chuvas intensas figuram na segunda posição, combinando perdas no setor produtivo e danos em infraestrutura.

| Tipologia do Evento | Prejuízo Histórico | Prejuízo Desde 2020 | Participação Recente (%) |
| :---- | :---- | :---- | :---- |
| **Estiagem e Seca** | R$ 287,31 bi | R$ 183,81 bi | 64,0% |
| **Chuvas Intensas** | R$ 91,30 bi | R$ 71,05 bi | 77,8% |
| **Inundações** | R$ 22,05 bi | R$ 7,09 bi | 32,2% |
| **Movimento de Massa** | R$ 9,09 bi | R$ 4,11 bi | 45,2% |
| **Enxurradas** | R$ 13,77 bi | R$ 3,40 bi | 24,6% |

### **Cruzamento: Setores x Tipologia**

* **Estiagem e Seca:** Prejuízos concentrados no setor primário. Desde 2020, somou R$ 137,08 bi na Agricultura e R$ 38,37 bi na Pecuária (representando 95,6% das perdas totais relativas a esse evento no período). No âmbito público, impacta o Abastecimento de Água.  
* **Chuvas Intensas:** Registrou R$ 71,05 bi desde 2020 (77,8% do total histórico da categoria). Atingiu Agricultura (R$ 24,80 bi), comércio (R$ 4,35 bi) e pecuária (R$ 3,04 bi), sendo a principal causa de danos declarados em Transportes.  
* **Movimento de Massa:** Ao contrário dos eventos agrícolas, o Comércio lidera os prejuízos privados nesta categoria, com R$ 1,66 bi registrados desde 2020, refletindo sua ocorrência em áreas urbanas.  
* **Danos Materiais:** Enquanto a seca predomina nos prejuízos privados produtivos, as chuvas intensas representam a maior parcela da destruição física (R$ 37,53 bi no histórico e R$ 30,26 bi no período recente), afetando Obras de Infraestrutura (R$ 18,63 bi recentes) e Unidades Habitacionais (R$ 18,81 bi recentes).

### **Distribuição Regional e Estadual**

A Região Sul concentra a maior parcela dos prejuízos econômicos acumulados, seguida pelo Nordeste e Sudeste.

* **Sul:** R$ 186,46 bi (Histórico) | R$ 154,87 bi (Desde 2020). Lidera em Estiagem/Seca (R$ 113,28 bi recentes) e em Chuvas Intensas (R$ 30,34 bi recentes).  
* **Nordeste:** R$ 134,94 bi (Histórico) | R$ 48,20 bi (Desde 2020). Apresenta histórico elevado em seca/estiagem (R$ 113,13 bi históricos).  
* **Sudeste:** R$ 72,13 bi (Histórico) | R$ 41,49 bi (Desde 2020). Registra prejuízos distribuídos entre seca (R$ 18,05 bi recentes) e chuvas intensas (R$ 20,10 bi recentes).

### **Principais UFs em Prejuízo Total**

1. **Rio Grande do Sul (RS):** R$ 128,62 bi Histórico | R$ 110,89 bi Desde 2020  
2. **Minas Gerais (MG):** R$ 46,06 bi Histórico | R$ 29,12 bi Desde 2020  
3. **Paraná (PR):** R$ 32,68 bi Histórico | R$ 26,15 bi Desde 2020  
4. **Bahia (BA):** R$ 31,34 bi Histórico | R$ 10,91 bi Desde 2020  
5. **Pernambuco (PE):** R$ 28,78 bi Histórico | R$ 14,48 bi Desde 2020

### **Top Municípios por Prejuízo Econômico Recente (2020+)**

| Município | UF | Prejuízo Recente (R$) | Principal Tipologia |
| :---- | :---- | :---- | :---- |
| **São Leopoldo** | RS | R$ 5,07 bi | Chuvas Intensas |
| **Maceió** | AL | R$ 3,60 bi | Colapso / Instabilidade Solo |
| **Sorriso** | MT | R$ 3,04 bi | Estiagem e Seca |
| **Cascavel** | PR | R$ 2,11 bi | Estiagem e Seca |
| **Petrópolis** | RJ | R$ 1,90 bi | Chuvas Intensas |
| **Januária** | MG | R$ 1,74 bi | Estiagem e Seca |
| **São Lourenço do Sul** | RS | R$ 1,74 bi | Chuvas Intensas / Inundações |
| **Palmeira das Missões** | RS | R$ 1,67 bi | Estiagem e Seca |

---

# **3. Módulo 2: Impacto Social e Humano**

## **3.1 Visão Geral dos Danos Humanos**

Os dados registram ocorrências de impacto humano em diferentes dimensões. Mais da metade dos registros acumulados de afetados ocorreu a partir de 2020.

* **Total Geral de Danos Humanos:** 207.201.897 registros (119.173.784 concentrados desde 2020, correspondendo a 57,5%).  
* **População Deslocada:** 5.260.447 pessoas. Desde 2020, os desalojados somam 2.990.381 (abrigados temporariamente com terceiros) e os desabrigados somam 473.725 (atendidos em abrigos públicos).  
* **Feridos:** 56.386 pessoas (38.897 no período recente).  
* **Vítimas Fatais (Mortes):** 1.776 mortes totais (com 1.242 óbitos registrados desde 2020, equivalente a 69,9% do total histórico).  
* **Desaparecidos:** 708 pessoas (615 no período recente).

A contagem de afetados por seca/estiagem atinge 116.079.443 registros históricos. Esse indicador retrata a extensão do impacto prolongado sobre acesso à água e subsistência rural, apresentando características dinâmicas distintas das ocorrências de impacto imediato. Os deslocamentos forçados concentram-se predominantemente nos eventos pluviais.

## **3.2 Análise por Tipologia de Desastre**

| Tipologia do Evento | Óbitos Recentes | Desaparecidos Recentes | Deslocados Recentes | Danos Materiais Recentes |
| :---- | :---- | :---- | :---- | :---- |
| **Chuvas Intensas** | 1.021 | 578 | 2.446.196 | R$ 30,26 bi |
| **Inundações** | -- | -- | 1.579.881 (Hist.) | R$ 7,09 bi |
| **Enxurradas** | 79 (Hist.) | -- | -- | R$ 3,40 bi |
| **Movimento de Massa** | 80 (Hist.) | -- | -- | R$ 4,11 bi |
| **Estiagem e Seca** | 15 | -- | Menor relevância | Negligenciável em estruturas |

### **Características dos Impactos Humano e Social**

* **Chuvas Intensas:** Constituem a principal causa de mortalidade e deslocamento nos dados analisados. Registraram 1.115 mortes no histórico e 1.021 no período recente (com 91,6% das mortes dessa tipologia ocorrendo a partir de 2020), além de 578 desaparecidos e 14.249 feridos recentes.  
* **Inundações, Enxurradas e Movimento de Massa:** Somam participações relevantes em mortalidade e perdas materiais em áreas com declividade ou proximidade a rios (180 mortes históricas por inundações, 80 por movimento de massa e 79 por enxurradas).  
* **Deslocamento Forçado (Desabrigados e Desalojados):** Chuvas intensas acumulam 2.682.350 registros históricos (2.446.196 recentes, representando 70,6% do total recente consolidado).  
* **Estiagem e Seca:** Lidera a contagem de enfermos (405.448 no histórico), com baixos índices de mortalidade direta imediata (112 mortes registradas no histórico, sendo 15 recentes).

## **3.3 Recortes Territoriais do Impacto Social**

* **Região/UF de Maior Mortalidade:** O Sudeste registra o maior volume histórico (839 mortes) e recente (631 mortes). O Estado do Rio de Janeiro contabiliza 356 mortes no histórico (304 desde 2020, com destaque para Petrópolis, que registrou 242 óbitos recentes). O Rio Grande do Sul registra 272 mortes totais, seguido por São Paulo (218) e Minas Gerais (199).  
* **Região/UF de Maior Deslocamento:** O Rio Grande do Sul soma 1.341.696 deslocados no histórico (1.033.387 desde 2020). Os estados do Amazonas e Pará também apresentam volumes expressivos de deslocamentos associados às dinâmicas fluviais regionais.  
* **Contingente de Afetados por Seca:** Bahia (25.775.838), Pernambuco (17.056.200) e Ceará (13.216.354). No recorte recente, o Rio Grande do Sul (9.681.402) soma-se aos estados com elevados registros de afetados por estiagem.

### **Top 10 Municípios em Deslocamentos Acumulados Recentes (2020+)**

| Município | UF | Deslocados Recentes | Principal Tipologia |
| :---- | :---- | :---- | :---- |
| **São Leopoldo** | RS | 209.135 | Chuvas Intensas |
| **Canoas** | RS | 187.025 | Chuvas Intensas |
| **Jaboatão dos Guararapes** | PE | 99.889 | Chuvas Intensas / Enxurradas |
| **Guaíba** | RS | 53.701 | Chuvas Intensas |
| **Governador Valadares** | MG | 43.636 | Inundações |
| **Angra dos Reis** | RJ | 40.595 | Chuvas Intensas / Deslizamentos |
| **Erechim** | RS | 40.400 | Chuvas Intensas / Granizo |
| **Eldorado do Sul** | RS | 38.116 | Chuvas Intensas |
| **Marabá** | PA | 34.770 | Inundações |
| **Barreirinha** | AM | 30.416 | Inundações / Estiagem |

---

# **4. Análise Comparativa Temporal**

A distribuição dos dados nos períodos definidos indica a concentração das perdas financeiras e humanas nos anos mais recentes. O ano de 2022 registrou o maior montante de prejuízo econômico anual da base (R$ 79,94 bilhões) e o maior volume de mortes (502).

## **Concentração de Impactos no Período 2020–2025**

Desaparecimentos (86,9%)     ██████████████████████████████████░░░░ 86,9%

Mortes Totais (69,9%)        ███████████████████████████░░░░░░░░░░░ 69,9%

Desalojados (68,0%)          ██████████████████████████░░░░░░░░░░░░ 68,0%

Prejuízo Chuvas (77,8%)      ██████████████████████████████░░░░░░░░ 77,8%

Prejuízo Seca (64,0%)        ███████████████████████░░░░░░░░░░░░░░░ 64,0%

Prejuízo Total (62,9%)       ████████████████████████░░░░░░░░░░░░░░ 62,9%

* **86,9%** dos registros de desaparecimentos (615 de 708).  
* **69,9%** de todas as mortes registradas na base (1.242 de 1.776).  
* **77,8%** do prejuízo econômico acumulado em Chuvas Intensas (R$ 71,05 bi de R$ 91,30 bi).  
* **64,0%** do prejuízo econômico acumulado em Estiagem e Seca (R$ 183,81 bi de R$ 287,31 bi).  
* **62,9%** de todo o prejuízo financeiro da base (R$ 279,42 bi de R$ 444,25 bi).  
* **68,0%** dos registros de pessoas desalojadas.

---

# **5. Conclusão e Diretrizes Estratégicas**

## **5.1 Síntese Integrada de Risco**

A análise dos dados evidencia a coexistência de diferentes perfis de risco no território nacional:

* **Risco Produtivo e Hídrico (Seca e Estiagem):** Concentra perdas financeiras expressivas no setor primário (agronegócio) e afeta o abastecimento hídrico em escala regional, sem apresentar elevada mortalidade direta imediata.  
* **Risco Humano e Urbano (Eventos Pluviais):** Chuvas intensas, enxurradas e deslizamentos representam a maior proporção de mortes, desaparecimentos, danos a habitações e desalojamentos populacionais.  
* **Risco de Infraestrutura:** As redes de abastecimento de água e a infraestrutura de transportes absorvem os maiores prejuízos no âmbito dos serviços públicos.  
* **Risco Territorial:** O Sul registra expressivas perdas econômicas e expressivo volume de deslocados; o Nordeste mantém histórico de afetados por estiagem; e o Sudeste concentra a maior mortalidade acumulada associada a eventos pluviométricos.

## **5.2 Recomendações Prioritárias**

1. **Gestão de Risco para Eventos Pluviais Intensos (Foco na Proteção Humana):**  
   Priorizar municípios com histórico recorrente de mortalidade e deslocamento nos estados do Rio de Janeiro, Rio Grande do Sul e Minas Gerais. Fortalecer sistemas de alerta antecipado, planos de evacuação, mapeamento de áreas suscetíveis a deslizamentos e proteção da infraestrutura de mobilidade urbana.  
2. **Adaptação para Segurança Hídrica e Setor Agrícola (Foco na Mitigação Financeira):**  
   Estruturar programas de mitigação contra secas em estados do Nordeste, Centro-Oeste e Sul (como BA, PE, CE, PB, MT e RS). Incentivar tecnologias de conservação de água, ampliação de reservatórios, seguro agrícola e adequação dos sistemas de distribuição de água.  
3. **Estruturação de Métricas Diferenciadas de Avaliação de Risco:**  
   Adotar metodologias que analisem separadamente indicadores de perda financeira e de impacto humano. Riscos à vida em encostas urbanas demandam ações de resposta rápida e habitação, enquanto perdas produtivas do setor agropecuário necessitam de instrumentos de gestão de risco financeiro e segurança hídrica.

## **5.3 Considerações Finais**

A distribuição dos dados demonstra que a alocação de recursos e a elaboração de políticas públicas devem considerar a tipologia do desastre e a especificidade territorial. Fatores como ocupação do solo, adensamento urbano e alteração na cobertura de dados influenciam os totais registrados. Contudo, os volumes consolidados no período 2020–2025 indicam elevada concentração de danos tanto no setor produtivo rural quanto na infraestrutura e população urbana.
"""

def _montar_prompt(str_econ, str_soc, data_formatada):
    return f"""
    Você é o Diretor Executivo de Análise Climática e Risco do Observatório do Clima.
    Sua tarefa é ATUALIZAR o nosso DIAGNÓSTICO EXECUTIVO BASE, incorporando de forma orgânica APENAS achados novos e relevantes extraídos das conversas do dia com a IA, preservando estritamente 100% do conteúdo original.

    CONTEXTO DOS DADOS: base do Atlas Digital de Desastres (S2iD/MDR), valores monetários
    corrigidos pelo IGP-DI (base dez/2022), dano autodeclarado pelos municípios.
    {VERDADES}

    DADOS DE TODAS AS CONVERSAS DO DIA - MÓDULO ECONÔMICO:
    {str_econ}

    DADOS DE TODAS AS CONVERSAS DO DIA - MÓDULO SOCIAL:
    {str_soc}

    === RELATÓRIO BASE (REFERÊNCIA INTOCÁVEL) ===
    {RELATORIO_BASE}

    REGRAS DE CONTEÚDO E FORMATAÇÃO (Obrigatórias):
    1. PRESERVAÇÃO ESTRITA: Você NÃO PODE remover, resumir ou alterar NENHUMA frase, número, tabela ou conclusão que já existe no RELATÓRIO BASE. Ele é a fundação do nosso diagnóstico e deve ser mantido integralmente.
    2. ADIÇÃO DE NOVIDADES: Analise os dados das CONVERSAS DO DIA. Se houver informações, recortes territoriais (ex: municípios específicos), ou achados que NÃO estão no relatório base, adicione-os nos locais adequados (ex: criando novos bullet points nas seções de "Recortes", adicionando linhas nas tabelas, ou criando um pequeno bloco "Novos Achados do Dia").
    3. SEM REDUNDÂNCIA: Se as conversas do dia apenas perguntaram coisas que já estão contempladas ou respondidas no Relatório Base, ignore-as completamente. Só acrescente o que for de fato uma nova descoberta.
    4. INTEGRIDADE MATEMÁTICA: Copie cada valor novo exatamente como aparece nas conversas do dia (ex: "R$ 12,34 bilhões"). NUNCA converta unidades, calcule percentuais da sua cabeça, ou use a palavra "trilhão". 
    5. DATA DE EMISSÃO: Logo abaixo do título principal do relatório, adicione a linha: "**Atualizado em:** {data_formatada}"
    
    Responda APENAS com o relatório final completo em Markdown, unindo perfeitamente o texto base intacto com as novas inserções orgânicas, sem nenhum aviso introdutório ou de fechamento.
    """


def gerar_diagnostico_executivo(desde=None):
    """Gera o diagnóstico consolidado baseado no histórico de chats."""
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

    timestamp_agora = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_formatada = datetime.now().strftime('%d/%m/%Y %H:%M')

    # CORREÇÃO 1: Se os buckets estiverem vazios (nenhuma conversa do dia), não aborte. 
    # Publique o relatório base para garantir que a tela no Streamlit jamais fique vazia.
    if not itens_econ and not itens_soc:
        print("⚠️ Nenhuma conversa nova registrada nos buckets. Publicando o relatório base diretamente.")
        relatorio_intacto = RELATORIO_BASE + f"\n\n*(Diagnóstico executivo carregado na íntegra. Atualizado em: {data_formatada}. Nenhuma nova consulta identificada nas últimas horas).*"
        nome_arquivo_relatorio = f"diagnostico_executivo_{timestamp_agora}.md"
        salvar_relatorio_no_bucket(BUCKET_RELATORIOS, nome_arquivo_relatorio, relatorio_intacto)
        return relatorio_intacto

    str_econ = preparar_modulo(model, itens_econ, "módulo econômico")
    str_soc = preparar_modulo(model, itens_soc, "módulo social")

    prompt = _montar_prompt(str_econ, str_soc, data_formatada)
    if DEBUG:
        with open("ultimo_prompt_relatorio.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

    try:
        # CORREÇÃO 2: max_output_tokens alterado de 4096 para 8192 para suportar a reescrita do relatório gigante inteiro
        relatorio_md = _gerar(model, prompt, temperature=0.1, max_output_tokens=8192)
    except Exception as e:
        print(f"❌ Falha ao gerar o relatório via IA: {e}")
        return None

    nome_arquivo_relatorio = f"diagnostico_executivo_{timestamp_agora}.md"
    salvar_relatorio_no_bucket(BUCKET_RELATORIOS, nome_arquivo_relatorio, relatorio_md)
    return relatorio_md


if __name__ == "__main__":
    gerar_diagnostico_executivo()