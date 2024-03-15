import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from args import Args


tickers = ['BTC', 'ETH', 'USDT']
agents = ['dqn', 'ddqn']

for ticker in tickers:
    for agent in agents:
        args = Args(ticker=ticker, agent=agent)
        val_path = os.path.join('./results', ticker, agent, 'val_results.csv')
        df = pd.read_csv(val_path)
        df['date'] = pd.date_range(start='2023-01-01', periods=args.trading_days)
        
        train_path = os.path.join('./results', ticker, agent, 'train_results.csv')
        df_t = pd.read_csv(train_path)
        print("ticker=", ticker, "agent=", agent, "average final training loss (10-day avg)=", 
              df_t['Losses'].tail(10).mean(), "average final excess return (50-day avg)=", df_t['Asset'].tail(50).mean())
        print("ticker=", ticker, "agent=", agent, "average final training profit (10-day avg)=", df['asset'].tail(10).mean())

        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['price'], color='black', lw=2.0, alpha=1)

        colors = ['red', 'blue', 'green'] #sell, hold, buy
        cmap = ListedColormap(colors)

        for action in range(3):
            df_a = df[df['action'] == action]
            s = 35
            if action == 2:
                s = 15
            plt.scatter(df_a['date'], df_a['price'], color=colors[action], s=s, alpha=.8)

            plt.xlabel('Date')
            plt.ylabel('Daily Closing Price')
            plt.title('Action Taken on Each Trading Day during Evaluation')

            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(os.path.join('./results', ticker, agent, 'val_action_graph.png'), dpi=500)



