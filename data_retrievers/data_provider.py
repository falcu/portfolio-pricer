import datetime
from abc import ABC, abstractmethod
import fix_yahoo_finance as yf
import pandas as pd
from pandas_datareader import data
from decorators import computeBefore


class PriceDataProvider(ABC):
    def __init__(self):
        self._dataRetrieved = False
        self._marketData = None

    def preCompute(self):
        if not self._dataRetrieved:
            self.preComputeAction()
            self._dataRetrieved = True

    @abstractmethod
    def preComputeAction(self):
        pass

    @computeBefore
    def prices(self):
        return self._marketData

    @abstractmethod
    def tickers(self):
        pass

class OnlinePriceDataProvider(PriceDataProvider):

    def __init__(self, tickers, start=datetime.date(2016,1,1), end=None):
        self._tickers = tickers
        self.start = start
        self.end   = end or datetime.date.today()
        super().__init__()

    def preComputeAction(self):
        dateToStr = lambda d: d.strftime("%Y-%m-%d")
        yf.pdr_override()
        self._marketData = data.get_data_yahoo(self._tickers, start=dateToStr(self.start), end=dateToStr(self.end))

    def tickers(self):
        return self._tickers

class OfflinePriceDataProvider(PriceDataProvider):
    def __init__(self, fileName, tickers):
        self._fileName = fileName
        self._tickers = tickers
        super().__init__()

    def preComputeAction(self):
        self._marketData = pd.read_pickle(self._fileName)

    def tickers(self):
        return self._tickers

class EnhancedValuesProvider:

    def __init__(self, dataProvider ):
        self.dataProvider = dataProvider

    def closePrices(self):
        return self.dataProvider.prices()['Close'][self.dataProvider.tickers()].fillna(method='backfill').fillna(method='ffill')

    def returns(self):
        prices = self.closePrices()
        zeroToTMinus1Prices = prices.loc[prices.index[0:-1]]
        oneToTPrices = prices.loc[prices.index[1:]].set_index(zeroToTMinus1Prices.index)
        return oneToTPrices.sub(zeroToTMinus1Prices).divide(zeroToTMinus1Prices)