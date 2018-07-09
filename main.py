from data_retrievers.data_provider import  EnhancedValuesProvider, OfflinePriceDataProvider, OnlinePriceDataProvider
import data_retrievers.market_constants as market_constants
from simulation.montecarlo import MontecarloParameterProvider, MontecarloMultipleStocks, PortfolioSimulation
import datetime
import numpy as np

def testVarOfflineData():
    import os
    mainPath = os.path.dirname(os.path.realpath(__file__))
    indexesPath = os.path.join(mainPath,'data','indexes')
    fxPath = os.path.join(mainPath,'data','fx')
    stockProvider = EnhancedValuesProvider(
        OfflinePriceDataProvider(indexesPath,
                                 market_constants.INDEXES_TICKERS))
    ccyProvider = EnhancedValuesProvider(
        OfflinePriceDataProvider(fxPath, market_constants.CCY))
    montecarloParams = MontecarloParameterProvider(stockProvider, ccyProvider,
                                                   referenceDate=datetime.date(2018, 6, 26)).montecarloParameters()

    montecarloSimulation = MontecarloMultipleStocks(montecarloParams.initialPricesVector,
                                                    montecarloParams.meanVector, montecarloParams.volatilityVector, montecarloParams.corrMatrix)
    portfolioSimulation = PortfolioSimulation(montecarloSimulation)

    return portfolioSimulation.oneDayVar(1,10000)

def testVarOnlineData():
    stockProvider = EnhancedValuesProvider(OnlinePriceDataProvider(market_constants.INDEXES_TICKERS))
    ccyProvider = EnhancedValuesProvider(OnlinePriceDataProvider(market_constants.CCY))
    montecarloParams = MontecarloParameterProvider(stockProvider, ccyProvider,
                                                   referenceDate=datetime.date(2018, 6, 26)).montecarloParameters()

    montecarloSimulation = MontecarloMultipleStocks(montecarloParams.initialPricesVector,
                                                    montecarloParams.meanVector, montecarloParams.volatilityVector, montecarloParams.corrMatrix)
    portfolioSimulation = PortfolioSimulation(montecarloSimulation)

    return portfolioSimulation.oneDayVar(1,10000)

def testHarcodedParamsVar():
    priceVector = np.array([  9999.99838895,  11168.84721966,   4619.252541  , 413.67599052, 508.6407107 ])
    meanVector = np.array([ 0.04735843, -0.0595617 ,  0.05191102,  0.00915321,  0.05615462])
    volatVector = np.array([ 0.11076801,  0.13717778,  0.11199026,  0.09955415,  0.11671932])
    corrMatrix = [[ 1.        ,  0.60022832,  0.35898327,  0.38086786,  0.32925194],
       [ 0.60022832,  1.        ,  0.35096846,  0.33808315,  0.30583728],
       [ 0.35898327,  0.35096846,  1.        ,  0.14312697,  0.28641867],
       [ 0.38086786,  0.33808315,  0.14312697,  1.        ,  0.33888931],
       [ 0.32925194,  0.30583728,  0.28641867,  0.33888931,  1.        ]]

    montecarloSimulation = MontecarloMultipleStocks(priceVector, meanVector, volatVector, corrMatrix)
    portfolioSimulation = PortfolioSimulation(montecarloSimulation)

    return portfolioSimulation.oneDayVar(1, 10000)

def testOs():
    import os
    print(os.path.dirname(os.path.realpath(__file__)))