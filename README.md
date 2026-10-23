# 🌦️ Diagnóstico Executivo Integrado de Impacto Climático no Brasil

![Status](https://img.shields.io/badge/Status-Concluído-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Data](https://img.shields.io/badge/Dados-S2iD%20%7C%20MIDR-lightgrey)

## 📌 Sobre o Projeto
Este repositório contém a análise de dados, os scripts de tratamento e o aplicativo interativo desenvolvidos para o **Observatório do Clima**. O objetivo deste projeto é consolidar e analisar os prejuízos econômicos, danos materiais e impactos humanos decorrentes de desastres climáticos no Brasil, com base nos registros municipais do Atlas Digital de Desastres (24.617 registros reconhecidos entre 2013 e 2025).

O diagnóstico visa destravar decisões estratégicas de gestão pública, respondendo a três perguntas fundamentais: *Onde investir? O que mitigar? Quem proteger primeiro?*

---

## 🚨 A Dupla Crise Climática (O Ponto de Ruptura Pós-2020)

> *"Com **62,9% dos prejuízos** econômicos e **69,9% das mortes** históricas concentrados pós-2020, o país exige duas respostas imediatas: **segurança hídrica** para salvar o setor produtivo e **infraestrutura urbana** para proteger vidas."*

O perfil de impacto no Brasil é distinto por tipologia. O país sofre duas crises paralelas e com perfis opostos: de um lado, a **estiagem e seca**, agindo de forma silenciosa e prolongada, lideram os prejuízos econômicos, com R$ 287,31 bilhões no histórico e R$ 183,81 bilhões desde 2020, fortemente concentrados na agricultura e pecuária, impactando a economia rural e o abastecimento hídrico. Do outro, **chuvas intensas** lideram de forma súbita e violenta mortes, desaparecimentos, feridos, deslocamentos e danos materiais nos centros urbanos, com 1.021 mortes e R$ 30,26 bilhões em danos materiais apenas desde 2020.

---

## 🤖 Arquitetura de Inteligência Artificial: "Chat com os Dados"

Um dos grandes diferenciais desta aplicação é a sua camada de Inteligência Artificial Generativa integrada diretamente aos dados. Construímos uma arquitetura que não apenas exibe gráficos, mas permite que o usuário *converse* com a base, utilizando a API do Gemini.

### 💬 O Assistente Interativo (Text-to-Pandas)
Dentro do aplicativo Streamlit, implementamos um **Chat Assistente IA**. O motor LLM traduz perguntas em linguagem natural do usuário (ex: *"Qual o prejuízo total causado por seca no Nordeste em 2022?"*) em instruções e consultas Pandas seguras, executadas na base de dados no contexto da aplicação. Os resultados matematicamente exatos retornam para o modelo formatar a resposta técnica.

### 🧠 Memória Coletiva e Geração de Relatório
Para escalar o conhecimento, implementamos um pipeline de memória coletiva:
1. **Armazenamento:** Cada interação útil do assistente nos módulos Econômico e Social é salva no **Google Cloud Storage (GCS)** em formato JSON, atuando como um log da inteligência coletiva.
2. **Síntese Diária:** Um script assíncrono isolado (`gerar_relatorio.py`) faz a varredura desses buckets, compacta as conversas relevantes de todos os usuários, remove repetições e solicita ao LLM que atue como o "Diretor Executivo de Análise".
3. **Produto Final:** O modelo gera um *Diagnóstico Executivo Consolidado* em Markdown, apontando automaticamente os maiores riscos e tendências de todas as interações feitas. O relatório fica imediatamente disponível para visualização e download na página `📄 Relatorio Executivo`.

Essa arquitetura elimina a necessidade de emitir os relatórios manualmente, extraindo valor estratégico direto do uso orgânico da plataforma.

---

## 🔬 Metodologia de Análise

A análise foi desenhada para separar o problema em eixos de atuação distintos, evitando que o impacto financeiro ofuscasse a crise humanitária (e vice-versa). A metodologia foi estruturada em dois módulos principais:

1. **Módulo Econômico:** Focado em identificar *onde o dinheiro se perde*. Mapeia os prejuízos aos cofres públicos, ao setor privado (com destaque absoluto para agropecuária) e os danos materiais à infraestrutura.
2. **Módulo Social:** Focado em *quem precisa ser protegido*. Analisa perdas de vidas, feridos e os massivos deslocamentos urbanos (desabrigados e desalojados).

**Horizontes Temporais Analisados:**
- **Histórico Completo:** Todos os eventos da base (2013 a 2025).
- **Últimos 10 anos:** Eventos a partir de 22/10/2016.
- **Recente (Pós-2020):** Eventos a partir de 01/01/2020, evidenciando a recente aceleração estrutural do risco.

---

## 🛡️ Validação e Tratamento de Dados

Para garantir a acurácia do diagnóstico e evitar distorções (como dupla contagem), aplicamos regras estritas de negócio durante o ETL:

* **Prejuízo Econômico Total:** Diferente da base original que separava as rubricas, criamos uma métrica agregada somando o *Prejuízo Privado*, *Prejuízo Público* e *Danos Materiais* para refletir o custo global do evento.
* **Métricas Humanas Seguras:** Substituímos a soma cega de colunas pelo uso estrito dos campos `DH_total_danos_humanos_diretos` e `DH_OUTROS AFETADOS`. Isolamos Desabrigados e Desalojados como "População Deslocada", evitando inflar artificialmente o número de indivíduos únicos afetados.
* **Ajuste Monetário:** Os valores financeiros utilizados mantêm a correção inflacionária nativa da versão CSV do Atlas Digital de Desastres (ajustados pelo IGP-DI com referência em dezembro de 2022).

---

## 📈 Principais Achados

- **Concentração Recente Extrema:** R$ 279,42 bilhões de prejuízos econômicos totais (62,9% do histórico) e 1.242 mortes (69,9% do histórico) ocorreram apenas a partir de 2020.
- **Risco Produtivo:** 95,6% das perdas econômicas totais com a seca desde 2020 atingiram diretamente a Agricultura (R$ 137,08 bi) e a Pecuária (R$ 38,37 bi).
- **Risco à Vida:** 91,6% das mortes causadas por Chuvas Intensas registradas na base ocorreram pós-2020. A tipologia também lidera a destruição de infraestrutura, com R$ 30,26 bilhões em danos materiais no mesmo período.
- **Diferenças Territoriais:** O Sul concentra a maior fatia do prejuízo econômico (R$ 154,87 bi recentes, puxados pelo RS); o Nordeste mantém a maior contagem de pessoas afetadas cronicamente pela seca; e o Sudeste lidera a mortalidade (puxado pelas chuvas intensas no RJ).

---

## 💡 Recomendações Estratégicas

Com base nos dados, direcionamos a estratégia pública para três frentes:

1. **Gestão de Risco Pluvial (Foco na Proteção da Vida):**
   Priorizar municípios de encostas e várzeas no RS, RJ e MG. Exige integração forte de alertas meteorológicos, mapeamento de áreas de deslizamento e proteção da infraestrutura de drenagem urbana.
2. **Adaptação para Segurança Hídrica (Foco na Mitigação Financeira):**
   Tratar a estiagem no BA, PE, CE, PB, MT e RS como problema estrutural, não sazonal. Exige investimento pesado em retenção de água, reservatórios avançados e seguro rural para evitar o colapso logístico das cadeias do agronegócio.
3. **Métricas de Risco Desacopladas:**
   Abandono da nota de risco unificada. Riscos à vida em chuvas exigem ações emergenciais da Defesa Civil; perdas de capital no campo por seca exigem resposta de crédito e financiamento.

---

## 🔗 Links e Recursos

- 📊 **App Interativo (Streamlit):** [App](https://ecoa-observatorio-do-clima.streamlit.app/)
- 📑 **Apresentação Executiva (Google Slides):** [Apresentação](https://docs.google.com/presentation/d/1mJwzfK6pDothryH3QxM6u7rQ-Hc6U-xwPP1WYuUv4J0/edit?slide=id.p1#slide=id.p1)
- 📑 **Relatório Executiva (PDF):** [Relatório](https://docs.google.com/document/d/1tQVyeJ9vZPLlrEMqHzg_g3QjfDA7rT_pQmbJBy5KbC0/edit?tab=t.0#heading=h.ou3std61sv8z)
- 📄 **Base de Dados Original:** S2iD / Ministério da Integração e do Desenvolvimento Regional (MIDR).

---

## 👥 Equipe

Projeto desenvolvido como parte da Residência de Dados (DataConnect).
- **Márlon Moreira de Almeida** 
- **Luiza Rasbold** 
- **Magno Moraes** 
- **Fabíola Loterio**
- **Maxxi Soares**
- **Tulani Souza**

---
*Feito com análise crítica, dados públicos e Python.* 🐍
