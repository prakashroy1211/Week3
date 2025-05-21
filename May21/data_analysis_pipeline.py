import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

#Mock Gemini LLM and GroupChat classes

class GeminiLLM:
    def __init__(self, model="gemini-1.5-flash"):
        self.model = model

    async def generate(self, prompt):
        # Simulate LLM async call with a dummy response
        await asyncio.sleep(0.5)
        return f"LLM({self.model}) response to: {prompt[:50]}..."

class Agent:
    def __init__(self, name, llm):
        self.name = name
        self.llm = llm

    async def step(self, message):
        # To be implemented in subclasses
        pass

class RoundRobinGroupChat:
    def __init__(self, agents):
        self.agents = agents
        self.index = 0

    async def next_turn(self, message):
        agent = self.agents[self.index]
        self.index = (self.index + 1) % len(self.agents)
        return await agent.step(message)

#Tools

class PandasTool:
    async def load_csv(self, path_or_url: str) -> pd.DataFrame:
        # If URL, fetch async, else read local file synchronously
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(path_or_url) as resp:
                    resp.raise_for_status()
                    csv_text = await resp.text()
            from io import StringIO
            df = pd.read_csv(StringIO(csv_text))
        else:
            df = pd.read_csv(path_or_url)
        return df

    async def describe(self, df: pd.DataFrame) -> str:
        return df.describe().to_string()

class MatplotlibTool:
    async def plot_histogram(self, df: pd.DataFrame, column: str) -> bytes:
        plt.figure(figsize=(8,6))
        df[column].hist(bins=20)
        plt.title(f"Histogram of '{column}'")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.read()

#Agents

class DataFetcherAgent(Agent):
    def __init__(self, pandas_tool, llm):
        super().__init__("DataFetcher", llm)
        self.pandas_tool = pandas_tool
        self.df = None

    async def step(self, message):
        print(f"[{self.name}] Fetching data from {message} ...")
        self.df = await self.pandas_tool.load_csv(message)
        print(f"[{self.name}] Data fetched with shape {self.df.shape}")
        return f"Data fetched with {len(self.df)} rows and {len(self.df.columns)} columns."

class AnalystAgent(Agent):
    def __init__(self, pandas_tool, matplotlib_tool, llm):
        super().__init__("Analyst", llm)
        self.pandas_tool = pandas_tool
        self.matplotlib_tool = matplotlib_tool

    async def step(self, message):
        df, column = message
        print(f"[{self.name}] Analyzing column '{column}' ...")
        description = await self.pandas_tool.describe(df)
        print(f"[{self.name}] Description ready.")
        histogram_png = await self.matplotlib_tool.plot_histogram(df, column)
        filename = f"histogram_{column}.png"
        with open(filename, "wb") as f:
            f.write(histogram_png)
        print(f"[{self.name}] Histogram saved as '{filename}'.")
        return description

#Main async program

async def main():
    pandas_tool = PandasTool()
    matplotlib_tool = MatplotlibTool()
    llm = GeminiLLM()

    data_fetcher = DataFetcherAgent(pandas_tool, llm)
    analyst = AnalystAgent(pandas_tool, matplotlib_tool, llm)

    chat = RoundRobinGroupChat(agents=[data_fetcher, analyst])

    #local CSV file path
    csv_path_or_url = "house_price.csv"  

    # Step 1: Fetch data
    fetch_response = await chat.next_turn(csv_path_or_url)

    # Step 2: Analyze first numeric column
    df = data_fetcher.df
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        print("No numeric columns found for analysis.")
        return
    column = numeric_cols[0]

    analysis_response = await chat.next_turn((df, column))

    print("\n--- Analysis Description ---")
    print(analysis_response)
    print(f"\nHistogram saved for column '{column}' as 'histogram_{column}.png'")

if __name__ == "__main__":
    asyncio.run(main())