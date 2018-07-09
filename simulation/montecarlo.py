import datetime
import math
import numpy as np
import pandas as pd


class SimulationParameters:
    def __init__(self, initialPricesVector, meanVector, volatilityVector, corrMatrix):
        self.initialPricesVector=initialPricesVector
        self.meanVector=meanVector
        self.volatilityVector=volatilityVector
        self.corrMatrix=corrMatrix

    def show(self):
        print('expected return {}'.format(self.meanVector * 100))
        print('volatility {}'.format(self.volatilityVector * 100))
        print('Corr {}'.format(self.corrMatrix))


class MontecarloParameterProvider:

    def __init__(self,stocksDataProvider, ccyDataProvider, referenceDate=None, historyInDays=360, daysInYear=252):
        self.stocksDataProvider = stocksDataProvider
        self.ccyDataProvider = ccyDataProvider
        self.referenceDate = referenceDate or datetime.date.today()
        self.historyInDays = historyInDays
        self.daysInYear = daysInYear

    def _pricesSubset(self):
        fromDate = self.referenceDate - datetime.timedelta(days = self.historyInDays)
        fromDateStr = self._toDateStr( fromDate )
        toDateStr = self._toDateStr( self.referenceDate + datetime.timedelta(days=1) )
        stockData = self.stocksDataProvider.closePrices().loc[fromDateStr:toDateStr]
        ccyData = self.ccyDataProvider.closePrices().loc[stockData.index]

        return stockData, ccyData

    def _toDateStr(self, d):
        return d.strftime("%Y-%m-%d")

    def _dolarizedStockValues(self):
        stockData, ccyData = self._pricesSubset()
        return pd.DataFrame(stockData.values*ccyData.values, columns=stockData.columns, index=stockData.index)

    def _computeReturns(self, prices):
        zeroToTMinus1Prices = prices.loc[prices.index[0:-1]]
        oneToTPrices = prices.loc[prices.index[1:]].set_index(zeroToTMinus1Prices.index)
        return oneToTPrices.sub(zeroToTMinus1Prices).divide(zeroToTMinus1Prices)

    def _meanReturnsAnnualized(self, stockReturns):
        return (stockReturns.mean(axis=0)*self.daysInYear).values

    def _volatilityReturnsAnnualized(self, stockReturns):
        return (stockReturns.var(axis=0).apply( lambda val : np.sqrt(val))*(np.sqrt(self.daysInYear))).values

    def _referenceDatePrices(self, prices):
        return prices.loc[prices.index[-1]].values

    def _correlatinMatrix(self, returnsMatrix):
        return returnsMatrix.corr().values

    def montecarloParameters(self):
        dolarizedValues = self._dolarizedStockValues()
        dolarizedReturns = self._computeReturns(dolarizedValues)
        meanVector = self._meanReturnsAnnualized(dolarizedReturns)
        volatilityVector = self._volatilityReturnsAnnualized(dolarizedReturns)
        initialPricesVector = self._referenceDatePrices(dolarizedValues)
        corrMatrix = self._correlatinMatrix(dolarizedReturns)

        return SimulationParameters( initialPricesVector, meanVector, volatilityVector, corrMatrix )


class MontecarloMultipleStocks:
    def __init__(self, priceVector, driftVector, volatlityVector, corrMatrix, deltaTime=1.0/252.0):
        self.priceVector = np.array(priceVector)
        self.driftVector = np.array(driftVector)
        self.volatlityVector = np.array(volatlityVector)
        self.corrMatrix = np.array(corrMatrix)
        self.deltaTime = deltaTime
        self.stocksCount = len(priceVector)
        self.choleskyDecomposition = np.linalg.cholesky(corrMatrix)

    def simulate(self, periods, trajectories):
        result = np.zeros((trajectories, periods+1, len(self.priceVector)))
        for t in range(0,trajectories):
            result[t][0]=self.priceVector.copy()
            for p in range(1,periods+1):
                result[t][p] = self.nextPrice(result[t][p-1])

        return result


    def nextPrice(self, price):
        expVector = ((self.driftVector- (0.5*self.volatlityVector**2))*self.deltaTime) + (self.volatlityVector*math.sqrt(self.deltaTime)*self._correlatedRV())
        return price * np.apply_along_axis(lambda val : np.exp(val),0,expVector)

    def _correlatedRV(self):
        return np.matmul( self.choleskyDecomposition, self._randomNormal().transpose())

    def _randomNormal(self):
        return np.random.standard_normal( self.stocksCount )


class PortfolioSimulation:
    def __init__(self, montecarloSimulator):
        self.montecarloSimulator = montecarloSimulator

    def simulateReturns(self, periods, trajectories, weights=None):
        pricesSim = self.montecarloSimulator.simulate( periods, trajectories )
        weights = self._weights(weights)
        pricesSimTranspose = np.transpose(pricesSim,(0,2,1)) #Not transposing trajectory dimension
        returnsSim = np.diff(np.log(pricesSimTranspose))
        returnsSimTransposed = np.transpose(returnsSim,(0,2,1))
        return np.sum(returnsSimTransposed*weights,2)

    def _weights(self, weights=None):
        return 1.0/self._stocksNumber() if weights is None else weights

    def _stocksNumber(self):
        return len(self.montecarloSimulator.priceVector)

    def oneDayVar(self, periods, trajectories, weights=None, alpha=0.05):
        portfolioReturns = self.simulateReturns( periods, trajectories, weights)
        indexingByPeriod = np.sort( np.transpose(portfolioReturns,(1,0)), 1)
        alphaVarIndex = int(trajectories*alpha)
        return indexingByPeriod[0:,alphaVarIndex]

    def expectedReturns(self, weights=None):
        weights = self._weights(weights)
        return np.sum(self.montecarloSimulator.driftVector*weights)


