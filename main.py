import discord
from discord.ext import commands
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import io
import requests
import pandas as pd
from datetime import datetime, timedelta

class APIAdapter:
    '''API adapter'''
    def get_price_data(self, coin_id, days):
        raise NotImplementedError

    def get_funding_rate_history(self, symbol, start_time, end_time, limit=1000):
        raise NotImplementedError

class CoinGeckoAdapter(APIAdapter):
    '''adapter for fetching prices from coingecko'''
    def get_price_data(self, coin_id, days):
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
        headers = {
            "accept": "application/json",
            "x-cg-demo-api-key": "CG-token"
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        prices = [price[1] for price in data["prices"]]
        return prices

    def get_funding_rate_history(self, symbol, start_time, end_time, limit=1000):
        raise NotImplementedError

class BinanceAdapter(APIAdapter):
    '''adapter for fetching prices and funding rates from binance exchange'''
    def get_price_data(self, coin_id, days):
        raise NotImplementedError

    def get_funding_rate_history(self, symbol, start_time, end_time, limit=1000):
        '''fundtion to call binance public api and get the funding rate of a specifici symbol from start to end date'''
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={start_time}&endTime={end_time}&limit={limit}"
        response = requests.get(url)
        data = response.json()
        return data


class BotSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.bot = None
            cls._instance.price_adapter = CoinGeckoAdapter()
            cls._instance.funding_adapter = BinanceAdapter()
        return cls._instance

    def initialize(self, bot_token):
        '''initializes the bot token'''
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        self._initialize_commands()
        self.bot.run(bot_token)

    def _initialize_commands(self):
        '''initializes the bot commands'''
        @self.bot.command(name='correlation')
        async def correlation(ctx):
            '''function to get the correlation between n amount of coins, returns a graph'''
            await ctx.send("Please enter the coins you want to analyze (separated by commas):")

            def check(msg):
                return msg.author == ctx.author and msg.channel == ctx.channel

            msg = await self.bot.wait_for('message', check=check)
            assets = [asset.strip().lower() for asset in msg.content.split(',')]

            if len(assets) < 2:
                await ctx.send("Please provide at least two assets to calculate the correlation.")
                return

            time_periods = ['7', '30', '90', '180', '365']
            options = [discord.SelectOption(label=f"{period} days", value=period) for period in time_periods]
            select = discord.ui.Select(placeholder="Select time period", options=options)

            async def callback(interaction):
                days = int(select.values[0])
                await interaction.response.defer()

                correlations = np.zeros((len(assets), len(assets)))
                sharpe_ratios = []

                for i, coin in enumerate(assets):
                    coin_prices = get_price_data(coin, days)
                    returns = np.diff(coin_prices) / coin_prices[:-1]
                    sharpe_ratio = calculate_sharpe_ratio(returns, days=days)
                    sharpe_ratios.append(sharpe_ratio)

                    for j, coin2 in enumerate(assets):
                        if i == j:
                            correlations[i, j] = 1.0
                        else:
                            coin2_prices = get_price_data(coin2, days)
                            min_length = min(len(coin_prices), len(coin2_prices))
                            coin_prices_trimmed = coin_prices[:min_length]
                            coin2_prices_trimmed = coin2_prices[:min_length]
                            correlation = calculate_correlation(coin_prices_trimmed, coin2_prices_trimmed)
                            correlations[i, j] = correlation
                            correlations[j, i] = correlation

                fig = generate_heatmap(correlations, assets, sharpe_ratios)

                buffer = io.BytesIO()
                fig.savefig(buffer, format='png')
                buffer.seek(0)

                await interaction.followup.send(file=discord.File(buffer, 'correlation_heatmap.png'))

            select.callback = callback
            view = discord.ui.View()
            view.add_item(select)

            await ctx.send("Select the time period:", view=view)

        @self.bot.command(name='funding')
        async def funding(ctx, symbol: str):
            '''command to get the funding rate form x data to x date'''
            symbol = symbol.upper() + "USDT"
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

            try:
                data = self.funding_adapter.get_funding_rate_history(symbol, start_time, end_time)
                if len(data) == 0:
                    await ctx.send(f"No funding rate data found for {symbol} in the last 7 days.")
                    return

                buffer = generate_funding_rate_graph(data)
                await ctx.send(file=discord.File(buffer, 'funding_rate_graph.png'))

            except Exception as e:
                await ctx.send(f"An error occurred while retrieving funding rate data: {str(e)}")

        @self.bot.command(name='price')
        async def prices(ctx, symbol: str):
            '''Get a the prices of an asset with date time format'''
            symbol = symbol.lower()
            price_date = get_price_data_with_timestamps(symbol)

            try:
                if len(price_date) == 0:
                    await ctx.send(f"No chart for coin name {symbol}")
                    return

                buffer = generate_coin_price_graph(price_date, symbol)
                await ctx.send(file=discord.File(buffer, 'price_chart.png'))

            except Exception as e:
                await ctx.send(f"An error occurred while retrieving price data: {str(e)}")

        @self.bot.command(name='ma')
        async def prices_with_ma(ctx, symbol: str, ma_periods: commands.Greedy[int]):
            '''returns moving average of an asset'''
            symbol = symbol.lower()
            price_data = get_price_data_with_timestamps(symbol)

            try:
                if len(price_data) == 0:
                    await ctx.send(f"No chart for coin name {symbol}")
                    return

                buffer = generate_coin_price_graph_with_ma(price_data, symbol, ma_periods)
                await ctx.send(file=discord.File(buffer, 'price_chart_with_ma.png'))

            except Exception as e:
                await ctx.send(f"An error occurred while retrieving price data: {str(e)}")

def get_price_data(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG token"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    prices = [price[1] for price in data["prices"]]
    return prices

def calculate_correlation(coin1_prices, coin2_prices):
    '''calculates correlation between two assets'''
    return pearsonr(coin1_prices, coin2_prices)[0]

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, days=365):
    '''get the sharp ratio of a an asset'''
    mean_return = np.mean(returns) * days
    std_dev = np.std(returns) * np.sqrt(days)
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev
    return sharpe_ratio

def generate_heatmap(correlations, coins, sharpe_ratios):
    '''heat map of correlations'''
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlations, cmap='viridis', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(coins)))
    ax.set_yticks(np.arange(len(coins)))
    ax.set_xticklabels(coins)
    ax.set_yticklabels(coins)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(coins)):
        for j in range(len(coins)):
            ax.text(j, i, f"{correlations[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_title("Correlation Heatmap")

    sharpe_text = "Sharpe Ratios:\n"
    for coin, sharpe_ratio in zip(coins, sharpe_ratios):
        sharpe_text += f"{coin}: {sharpe_ratio:.2f}\n"
    ax.text(1.05, 0.5, sharpe_text, transform=ax.transAxes, fontsize=12, verticalalignment='center')

    fig.tight_layout()
    return fig

def get_funding_rate_history(symbol, start_time, end_time, limit=1000):
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={start_time}&endTime={end_time}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    return data

def generate_funding_rate_graph(data):
    '''returns funding rate graph'''
    df = pd.DataFrame(data)
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['fundingRate'] = df['fundingRate'].astype(float)
    df['fundingRateAnnualized'] = df['fundingRate'] * 3 * 365 * 100  # Annualize the funding rate

    plt.figure(figsize=(10, 6))
    plt.plot(df['fundingTime'], df['fundingRateAnnualized'])
    plt.xlabel('Funding Time')
    plt.ylabel('Annualized Funding Rate (%)')
    plt.title('Annualized Funding Rate - Last 7 Days')
    plt.grid(True)
    plt.xticks(rotation=45)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

def get_price_data_with_timestamps(coin_id):
    '''get price data from coingecko api'''
    to_date = datetime.now().timestamp()
    from_date = (datetime.now() - timedelta(days=120)).timestamp()
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency=usd&from={from_date}&to={to_date}"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG-token"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    return data

def generate_coin_price_graph(price_data, symbol):
    '''get price graph from price and time'''
    prices = [price[1] for price in price_data["prices"]]
    timestamps = [datetime.fromtimestamp(price[0] / 1000) for price in price_data["prices"]]
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, prices, marker='.', linestyle='-')
    plt.title(f'60D price of {symbol}')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.xticks(rotation=45)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

def generate_coin_price_graph_with_ma(price_data, symbol, ma_periods):
    '''add moving averages to the graph'''
    prices = [price[1] for price in price_data["prices"]]
    timestamps = [datetime.fromtimestamp(price[0] / 1000) for price in price_data["prices"]]

    df = pd.DataFrame({'Price': prices}, index=timestamps)

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Price'], marker=',', linestyle='dotted', label='Price')

    for period in ma_periods:
        ma = df['Price'].rolling(window=period).mean()
        plt.plot(df.index, ma, label=f'{period}-day MA')

    plt.title(f'60D price of {symbol} with Moving Averages')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

if __name__ == '__main__':
    bot_token = 'BOT_TOKEN'
    bot_singleton = BotSingleton()
    bot_singleton.initialize(bot_token)
