from data_retrievers.data_provider import  EnhancedValuesProvider, OfflinePriceDataProvider, OnlinePriceDataProvider
import data_retrievers.market_constants as market_constants
from simulation.montecarlo import MontecarloParameterProvider, MontecarloMultipleStocks, PortfolioSimulation
import datetime
import numpy as np
import scipy.stats

def testPortfolioOfflineData():
    import os
    daysForHistoricalCalc = 180*2
    referenceDate = datetime.date(2018, 6, 22)

    mainPath = os.path.dirname(os.path.realpath(__file__))
    indexesPath = os.path.join(mainPath,'data','indexes')
    fxPath = os.path.join(mainPath,'data','fx')
    stockProvider = EnhancedValuesProvider(OfflinePriceDataProvider(indexesPath, market_constants.INDEXES_TICKERS))
    ccyProvider = EnhancedValuesProvider( OfflinePriceDataProvider(fxPath, market_constants.CCY))
    montecarloParams = MontecarloParameterProvider(stockProvider, ccyProvider,
                                                   referenceDate=referenceDate, historyInDays=daysForHistoricalCalc).montecarloParameters()

    montecarloSimulation = MontecarloMultipleStocks(montecarloParams.initialPricesVector,
                                                    montecarloParams.meanVector, montecarloParams.volatilityVector, montecarloParams.corrMatrix)
    portfolioSimulation = PortfolioSimulation(montecarloSimulation)
    montecarloParams.show()
    print('-----------------------------------------------------------------------')
    print('Equal Weights(not efficient frontier):')
    print("1 day var {}%".format(portfolioSimulation.oneDayVar(1, 10000, weights=None)*100))
    print("Expected Return {}%".format(portfolioSimulation.expectedReturns(weights=None) * 100))
    print('-----------------------------------------------------------------------')
    print('7% return efficeint frontier:')
    weights = np.array([21.12,0,29.2,15.83,33.85])/100
    print('Weights {}'.format(weights))
    print("1 day var {}%".format(portfolioSimulation.oneDayVar(1, 10000, weights=weights)*100))
    print("Expected Return {}%".format(portfolioSimulation.expectedReturns(weights=weights)*100))
    print('-----------------------------------------------------------------------')
    print('Lowest return efficeint frontier:')
    weights = np.array([17.04, 0, 26.82, 35.62, 20.52]) / 100
    print('Weights {}'.format(weights))
    print("1 day var {}%".format(portfolioSimulation.oneDayVar(1, 10000, weights=weights) * 100))
    print("Expected Return {}%".format(portfolioSimulation.expectedReturns(weights=weights) * 100))
    print('-----------------------------------------------------------------------')
    print('Highest return efficeint frontier:')
    weights = np.array([0,0,0,0,100])/100
    print('Weights {}'.format(weights))
    print("1 day var {}%".format(portfolioSimulation.oneDayVar(1, 10000, weights=weights)*100))
    print("Expected Return {}%".format(portfolioSimulation.expectedReturns(weights=weights)*100))

def testPortfolioOnline():
    start = datetime.datetime(2015,1,1)
    daysForHistoricalCalc = 180 * 2
    referenceDate = datetime.date(2018, 6, 22)


    stockProvider = EnhancedValuesProvider(OnlinePriceDataProvider(market_constants.INDEXES_TICKERS, start=start))
    ccyProvider = EnhancedValuesProvider(OnlinePriceDataProvider(market_constants.CCY, start=start),)
    montecarloParams = MontecarloParameterProvider(stockProvider, ccyProvider,
                                                   referenceDate=referenceDate, historyInDays=daysForHistoricalCalc).montecarloParameters()

    montecarloSimulation = MontecarloMultipleStocks(montecarloParams.initialPricesVector,
                                                    montecarloParams.meanVector, montecarloParams.volatilityVector, montecarloParams.corrMatrix)
    portfolioSimulation = PortfolioSimulation(montecarloSimulation)

    montecarloParams.show()
    print('-----------------------------------------------------------------------')
    print('Equal Weights(not efficient frontier):')
    print("1 day var {}%".format(portfolioSimulation.oneDayVar(1, 10000, weights=None)*100))
    print("Expected Return {}%".format(portfolioSimulation.expectedReturns(weights=None) * 100))
    print('-----------------------------------------------------------------------')
    print('7% return efficeint frontier:')
    weights = np.array([21.12,0,29.2,15.83,33.85])/100
    print('Weights {}'.format(weights))
    print("1 day var {}%".format(portfolioSimulation.oneDayVar(1, 100000, weights=weights)*100))
    print("Expected Return {}%".format(portfolioSimulation.expectedReturns(weights=weights)*100))
    print('-----------------------------------------------------------------------')
    print('Highest return efficeint frontier:')
    weights = np.array([0,0,0,0,100])/100
    print('Weights {}'.format(weights))
    print("1 day var {}%".format(portfolioSimulation.oneDayVar(1, 100000, weights=weights)*100))
    print("Expected Return {}%".format(portfolioSimulation.expectedReturns(weights=weights)*100))

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

def testMarcowitz():
    minMarkowitz = (0.0579 / 252, 0.0735 / np.sqrt(252))
    maxMarkowitz = (0.0889 / 252, 0.1145 / np.sqrt(252))
    print('One day var lowest {}%'.format(scipy.stats.norm(minMarkowitz[0],minMarkowitz[1]).ppf(0.05)*100))
    print('One day var highest {}%'.format(scipy.stats.norm(maxMarkowitz[0],maxMarkowitz[1]).ppf(0.05)*100))
