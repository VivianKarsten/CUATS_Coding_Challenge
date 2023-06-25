# Modified from https://quantpedia.com/strategies/low-volatility-factor-effect-in-stocks-long-only-version/
# 
# The investment universe is high momentum US mid-cap stocks
# Stocks are ranked on past year volatility, calculated weekly
# Monthly: Lowest volatility N stocks minus exited stocks are invested in equally
#          Buy US T notes with remaining funds
# Daily: Position in a stock is liquidated if it falls fraction x below its high
#        These stocks are filtered out until they recover fraction y from their low
# 
# Team: Coders love cookies
# Members: Vivian Karsten, Yu Chen Lim

from AlgorithmImports import *
import numpy as np

class LowVolatilityFactorEffect(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2007, 1, 1)
        self.SetEndDate(2014, 1, 1)
        self.SetCash(100_000) 

        self.data = {}              # tracks data for each stock
        self.long = []              # stocks selected by the uni filter
        self.exited = []            # tracks liquidated stocks due to drawdown
        self.selection_state = 0    # ensure uni filter (state 1) runs before investing (state 2)

        self.period = 12*21         # 12 month rolling window
        self.fine_count = 100       # N, max stocks from uni filter
        self.exit_thresh = 0.1      # x, max drawdown before liquidation
        self.reenter_thresh = 0.25  # y, min recovery before allowing back onto portfolio

        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

        # monthly rebalancing
        self.Schedule.On(self.DateRules.MonthStart(), self.TimeRules.At(9,5), self.Selection)
        
        # 10 year US Treasury Note futures
        self.bond = self.AddData(QuantpediaFutures, "CME_TY1", Resolution.Daily).Symbol
    
    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            security.SetLeverage(10)
            
    def CoarseSelectionFunction(self, coarse):
        # update the rolling window daily
        for stock in coarse:
            symbol = stock.Symbol
            if symbol in self.data:
                self.data[symbol].update(stock.AdjustedPrice)

        # run filter monthly
        if self.selection_state != 1:
            return Universe.Unchanged
        self.selection_state = 2

        selected = [x.Symbol for x in coarse if x.HasFundamentalData and x.Market == "usa"]

        for symbol in selected:
            if symbol in self.data:
                continue
        
            # warmup rolling windows
            self.data[symbol] = SymbolData(self.period, self.exit_thresh, self.reenter_thresh)
            history = self.History(symbol, self.period, Resolution.Daily)
            if history.empty:
                self.Log(f"Not enough data for {symbol} yet.")
                continue
            closes = history.loc[symbol].close
            for _, close in closes.iteritems():
                self.data[symbol].update(close)
        
        return [x for x in selected if self.data[x].is_ready()]

    def FineSelectionFunction(self, fine):
        fine = [x for x in fine if x.MarketCap != 0]
        
        # filter for mid caps
        fine = [x for x in fine if 2e9 <= x.MarketCap <= 2e10]

        # filter for high momentum
        fine.sort(key = lambda x: self.data[x.Symbol].momentum_12_1m())
        fine = fine[int(len(fine)/2):]
        
        # filter for low volatility (bottom quartile)
        weekly_vol = {x.Symbol : self.data[x.Symbol].volatility() for x in fine}
        sorted_by_vol = sorted(weekly_vol.items(), key = lambda x: x[1])
        quartile = min(int(len(sorted_by_vol) / 4), self.fine_count)
        long = [x[0] for x in sorted_by_vol[:quartile]]
        
        # remove recovered stocks from exited list
        self.exited = [x for x in self.exited if x in self.data and not self.data[x].reenter()]
        
        # remove exited stocks
        self.long = [x for x in long if x not in self.exited]

        return self.long
        
    def OnData(self, data):
        # liquidate position in stock if its drawdown exceeds threshold
        stocks_invested = [x.Key for x in self.Portfolio if x.Value.Invested]
        exit_list = []
        for symbol in stocks_invested:
             if symbol in data and symbol in self.data and self.data[symbol].exit():
                if symbol in self.exited:
                    self.Log(f"ERROR: exit {symbol}")
                self.Liquidate(symbol)
                exit_list.append(symbol)
        self.exited += exit_list

        # remove exited stocks
        stocks_invested = [x for x in stocks_invested if x not in exit_list]
        self.long = [x for x in self.long if x not in exit_list]

        # run rebalancing monthly
        if self.selection_state != 2:
            return
        self.selection_state = 0

        # portfolio rebalancing
        for symbol in stocks_invested:
            if symbol not in self.long + [self.bond]:
                self.Liquidate(symbol)
        
        # funds not invested in stocks is put into US T Notes
        if len(self.long) == self.fine_count:
            self.Liquidate(self.bond)
        else:
            self.SetHoldings(self.bond, 1 - len(self.long) / self.fine_count)

        for symbol in self.long:
            if symbol in data and data[symbol]:
                self.SetHoldings(symbol, 1 / self.fine_count)
                
                # reset when entering a position
                if not self.Portfolio[symbol].Invested:
                    self.data[symbol].reset_high()
        
        self.long.clear()
        
    def Selection(self):
        self.selection_state = 1

# class for calculating volatility, momentum, drawdown etc for the stock
class SymbolData():
    def __init__(self, period, exit_thresh, reenter_thresh):
        self.period = period
        self.price = RollingWindow[float](period)
        self.exit_thresh = exit_thresh
        self.reenter_thresh = reenter_thresh
        self.high = 0
        self.low = 0
    
    def update(self, value):
        self.price.Add(value)
        if value > self.high:
            self.low = value
        else:
            self.low = min(self.low, value)
        self.high = max(self.high, value)
    
    def is_ready(self) -> bool:
        return self.price.IsReady
        
    def volatility(self) -> float:
        closes = [x for x in self.price]
        
        # Weekly volatility calc.
        separate_weeks = [closes[x:x+5] for x in range(0, len(closes), 5)]
        weekly_returns = [(x[0] - x[-1]) / x[-1] for x in separate_weeks]
        return np.std(weekly_returns)
    
    # 12 month minus 1 month momentum
    def momentum_12_1m(self) -> float:
        return self.price[21]/self.price[self.period-1] - 1

    def reset_high(self):
        self.high = self.low = self.price[0]
    
    # drawdown from previous high
    def drawdown(self) -> float:
        return 1 - self.price[0]/self.high
    
    # exit condition
    def exit(self) -> bool:
        return self.drawdown() > self.exit_thresh
    
    # reentry condition
    def reenter(self) -> bool:
        return self.price[0]/self.low - 1 > self.reenter_thresh or self.price[0] == self.high

# Quantpedia data reader for CME_TY1
# NOTE: IMPORTANT: Data order must be ascending (datewise)
class QuantpediaFutures(PythonData):
    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource("data.quantpedia.com/backtesting_data/futures/{0}.csv".format(config.Symbol.Value), SubscriptionTransportMedium.RemoteFile, FileFormat.Csv)

    def Reader(self, config, line, date, isLiveMode):
        data = QuantpediaFutures()
        data.Symbol = config.Symbol
        
        if not line[0].isdigit(): return None
        split = line.split(";")
        
        data.Time = datetime.strptime(split[0], "%d.%m.%Y") + timedelta(days=1)
        data["back_adjusted"] = float(split[1])
        data["spliced"] = float(split[2])
        data.Value = float(split[1])

        return data