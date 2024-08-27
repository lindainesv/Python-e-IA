# Importação das libs
# import json
import os
from datetime import datetime
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
import yfinance as yf
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
# from langchain_openai import ChatOpenAI


async def main():
    # Criando Função: fetch_stock_price
    def fetch_stock_price(ticket):
        # Metodo: stock
        stock = yf.download(ticket, start="2024-01-01", end=datetime.now().strftime(f"%Y-%m-%d"))
        return stock

    yft = Tool(  # metodo Tools
        name="Yahoo Finance Tool",
        description="""Obtém preços de ações para {ticket} do último ano sobre uma
ação específica da API do Yahoo Finance""",  # descricao detalhada do que a função executa
        func=lambda ticket: fetch_stock_price(ticket),
    )

    # Importando OpenAI LLM - GPT
    os.environ["GOOGLE_API_KEY"] = 'AIzaSyDFKkcFy6OD7OmdPW8kZtv9HmL0j4GSbcs'
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    # Criando o agente stockPriceAnalyst
    # DOC.: https://docs.crewai.com/core-concepts/Agents/#what-is-an-agent
    stockPriceAnalyst = Agent(
        role="Analista Sênior de Preço de Ações",
        goal="Encontre o preço das ações {ticket} e analise tendências",  # Objetivo
        backstory="""Você tem muita experiência em analisar o preço de uma ação específica e fazer previsões sobre seu preço futuro.""",
        verbose=True,
        llm=llm,
        max_iter=5,
        memory=True,
        tools=[yft],
        allow_delegation=False,
    )

    # Criando tarefa para o agente executar
    getStockPrice = Task(
        description="""Analise o histórico de preços de ações {ticket} e crie análises de tendências de alta, baixa ou lateral""",
        expected_output="""Especifique a tendência atual do preço das ações - para cima, para baixo ou
para os lados.
por exemplo, ações para 'AAPL, preço PARA CIMA'""",
        agent=stockPriceAnalyst,
    )

    # Importando a tool de pesquisa
    search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)

    # Criando agente de noticias
    news_analyst = Agent(
        role="Analista de notícias de ações",
        goal="""Crie um breve resumo das notícias do mercado releted para a empresa de ações
{ticket}. Especifique a tendência atual - para cima, para baixo ou para os lados com
o contexto das notícias. Para cada ativo de ação solicitado, especifique um número entre 0
e 100, quando 0 é medo extremo e 100 é ambição extrema.""",
        backstory="""Você é altamente experiente em analisar tendências de mercado e notícias
e acompanha ativos há mais de 10 anos.
Você também é analista de nível mestre em mercados tradicionais e tem profundo
entendimento da psicologia humana.
Você entende notícias, seus títulos e informações, mas olha para
aqueles com uma boa dose de ceticismo.
Você também considera a fonte dos artigos de notícias.""",
        verbose=True,
        llm=llm,
        max_iter=5,
        memory=True,
        tools=[search_tool],
        allow_delegation=False,
    )

    # Criando tarefa do agente de noticias
    get_news = Task(
        description=f"""Pegue o estoque e sempre inclua BTC nele
(se não for solicitado).
Use a ferramenta de busca para pesquisar cada um individualmente.

A data atual é {datetime.now()}.

Componha os resultados em um relatório útil.""",
        expected_output="""Um resumo do mercado geral e um resumo de uma frase
para cada ativo solicitado.
Inclua uma pontuação de medo\ambição para cada ativo com base nas notícias. Use o
formato:
<ATIVO DE AÇÕES>
<RESUMO COM BASE NAS NOTÍCIAS>
<PREVISÃO DE TENDÊNCIAS>
<PONTUAÇÃO DE MEDO/AMBIÇÃO>""",
        agent=news_analyst,
    )

# Agente que irá escrever a analise de fato
    stock_analyst_white = Agent(
        role="Escritor Analista Sênior de Ações",
        goal="""Analise as tendências de preços e notícias e escreva um boletim informativo, perspicaz, atraente e de 3 parágrafos com base no relatório de ações e na tendência de preços.""",
        backstory="""Você é amplamente aceito como o melhor analista de ações do
mercado. Você entende conceitos complexos e cria histórias e
narrativas atraentes que ressoam com públicos mais amplos.
Você entende fatores macro e combina múltiplas teorias - por exemplo, teoria do
ciclo e análises fundamentais. Você é capaz de sustentar múltiplas opiniões
ao analisar qualquer coisa.Você é amplamente aceito como o melhor analista de ações do
mercado. Você entende conceitos complexos e cria histórias e
narrativas atraentes que ressoam com públicos mais amplos.
Você entende fatores macro e combina múltiplas teorias - por exemplo, teoria do
ciclo e análises fundamentais. Você é capaz de sustentar múltiplas opiniões
ao analisar qualquer coisa.""",
        verbose=True,
        llm=llm,
        max_inter=5,
        memory=True,
        allow_delegation=True,
    )

# Tarefa do agente stock_analyst_white
    writeAnalyses = Task(
        description="""Use a tendência do preço das ações e o relatório de notícias sobre ações para
criar uma análise e publicar o boletim informativo sobre a empresa {ticket}
que seja breve e destaque os pontos mais importantes.
Concentre-se na tendência do preço das ações, notícias e pontuação de medo/ambição. Quais são as considerações futuras próximas?
Inclua as análises anteriores da tendência das ações e o resumo das notícias.""",
        expected_output="""Um boletim informativo eloquente de 3 parágrafos formatado como
markdown de uma maneira fácil de ler. Ele deve conter:

- resumo executivo de 3 marcadores
- Introdução - define o quadro geral e aumenta o interesse
- A parte principal fornece o cerne da análise, incluindo o resumo de notícias
e pontuações de feed/greed
- Resumo - fatos principais e previsão concreta de tendências futuras - para cima, para baixo ou
para os lados.""",
        agent=stock_analyst_white,
        context=[getStockPrice, get_news],
    )

    # Criando o grupo de IAs
    crew = Crew(
        agents=[stockPriceAnalyst, news_analyst, stock_analyst_white],
        tasks=[getStockPrice, get_news, writeAnalyses],
        verbose=2,
        process=Process.hierarchical,
        full_output=True,
        share_crew=False,
        manager_llm=llm,
        max_inter=15,
    )

    # results = crew.kickoff(inputs={"ticket": "AAPL"})
    # Colocando a aplicação para rodar em front-end utilizando o streamlit
    with st.sidebar:

        st.header('Enter the Stock to Research')

        with st.form(key='research_form'):
            topic = st.text_input('Select the ticket')
            submit_button = st.form_submit_button(label='Run Research')

    if submit_button:
        if not topic:
            st.error('Please fill the ticket field')
        else:
            results = crew.kickoff(inputs={"ticket": topic})

            st.subheader('Results of your research:')
            st.write(results["final_output"])

asyncio.run(main())
