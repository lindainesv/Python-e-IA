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
        stock = yf.download(ticket, start="2024-01-01", end=datetime.now().strftime(f"%d-%m-%Y"))
        return stock

    yft = Tool(  # metodo Tools
        name="Yahoo Finance Tool",
        description="""Fetches stock prices for {ticket} from the last year about a 
        specific stock from Yahoo Finance API""",  # descricao detalhada do que a função executa
        func=lambda ticket: fetch_stock_price(ticket),
    )

    # Importando OpenAI LLM - GPT
    os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']
    # "AIzaSyAM2W1hHvAigykhwbR8KqdVZyctvKq6pFo"
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    # Criando o agente stockPriceAnalyst
    # DOC.: https://docs.crewai.com/core-concepts/Agents/#what-is-an-agent
    stockPriceAnalyst = Agent(
        role="Senior stock price Analyst",
        goal="Find the {ticket} stock price and analyses trends",  # Objetivo
        backstory="""You're a highly experienced in analyzing the price of an
        specific stock
    and make prediction about its future price.""",
        verbose=True,
        llm=llm,
        max_iter=5,
        memory=True,
        tools=[yft],
        allow_delegation=False,
    )

    # Criando tarefa para o agente executar
    getStockPrice = Task(
        description="""Analyze the stock {ticket} price history and create a trend
        analyses of up, down or sideways""",
        expected_output="""Specify the current trend stock price - up, down or
        sideways.
        eg. stock for 'AAPL, price UP'""",
        agent=stockPriceAnalyst,
    )

    # Importando a tool de pesquisa
    search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)

    # Criando agente de noticias
    news_analyst = Agent(
        role="Stock News Analyst",
        goal="""Create a short summary of the market news releted to the stock
        {ticket} company. Specify the current trend - up, down or sideways with
        the news context. For each request stock asset, specify a number between 0
        and 100, when 0 is extreme fear and 100 is extreme greed.""",
        backstory="""You're highly experient analyzing the market trends and news
        and have tracked assest for more then 10 years.
        You're also master level analyts in the tradicional markets and  have deep
        understanding of human psychology.
        You're understand news, theirs tittles and information, but you look ate
        those with a healt dose of skeepticism.
        You consider also the sourse of the news articles.""",
        verbose=True,
        llm=llm,
        max_iter=5,
        memory=True,
        tools=[search_tool],
        allow_delegation=False,
    )

    # Criando tarefa do agente de noticias
    get_news = Task(
        description=f"""Take the stock and always include BTC to it
        (if not request).
        Use  the search tool to search each one individually.

        The current date is {datetime.now()}.

        Compose the results into a helpfull report.""",
        expected_output="""A summary of the overall market and one sentence
        summary for each request asset.
        Include a fear\greed score for each asset based on the news. Use the
        format:
        <STOCK ASSET>
        <SUMMARY BASED ON THE NEWS>
        <TREND PREDICTION>
        <FEAR/GREED SCORE>""",
        agent=news_analyst,
    )

# Agente que irá escrever a analise de fato
    stock_analyst_white = Agent(
        role="Senior Stock Analyts Writer",
        goal="""Analyze the trends price and news and write an insighfull
        compelling and informative 3 paragraph long newsletter based on the stock
        report and price trend.""",
        backstory="""You're widely accepted as the best stock analyst in the
        market. You understand comples concepts and create compelling stories and
        narratives that ressonate with wider audiences.
        You understand macro factors and combine multiple theories - eg. cycle
        theory and fundamental analyses. You're able to hold multiple opinions
        when analyzing anything.""",
        verbose=True,
        llm=llm,
        max_inter=5,
        memory=True,
        allow_delegation=True,
    )

# Tarefa do agente stock_analyst_white
    writeAnalyses = Task(
        description="""Use the stock price trend and the stock news report to
        create an analyses and white the newsletter about the {ticket} company
        that is brief and highlights the most important points.
        Focus on the stock price trend, news and fear/greed score. What are near
        future considerations?
        Include the previous analyses of stock trend and news summary.""",
        expected_output="""An enloquent 3 paragraphs newsletter formated as
        markdown in an easy readable manner. It shold contain:

        - 3 bullets executive summary
        - Introduction - set the overall picture and spike up the interest
        - Main part provides the meat  of  the analysis including  the news
        summary and feed/greed scores
        - Summary - key facts and concrete future trend prediction - up, dows or
        sideways.""",
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
