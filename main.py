from data_retrievers.data_provider import  EnhancedValuesProvider, OfflinePriceDataProvider, OnlinePriceDataProvider
import data_retrievers.market_constants as market_constants
from simulation.montecarlo import MontecarloParameterProvider, MontecarloMultipleStocks, PortfolioSimulation
from optimizer.markowitz import MarkowitzWeightsCalculator, PortfolioOptimizer
import datetime
import numpy as np
import scipy.stats

def ej1(get_data_mode='offline'):
    daysForHistoricalCalc = 360
    referenceDate = datetime.date(2018, 6, 22)
    indexesPriceProvider = _makePriceProvider(get_data_mode, market_constants.INDEXES_TICKERS, 'indexes')
    fxPriceProvider = _makePriceProvider(get_data_mode, market_constants.CCY, 'fx')

    portfolioSimulator = _makePortfolioSimulator(indexesPriceProvider, fxPriceProvider, referenceDate, daysForHistoricalCalc)
    print('Ejercicio 1')
    print('Invertido Partes Iguales')
    print("Var 1 dia {}%".format(portfolioSimulator.oneDayVar(1, 100000)*100))
    print("Rendimiento Esperado {}%".format(portfolioSimulator.expectedReturns() * 100))

def ej2(get_data_mode='offline'):
    daysForHistoricalCalc = 360
    referenceDate = datetime.date(2018, 6, 22)
    indexesPriceProvider = _makePriceProvider(get_data_mode, market_constants.INDEXES_TICKERS, 'indexes')
    fxPriceProvider = _makePriceProvider(get_data_mode, market_constants.CCY, 'fx')
    portfolioSimulator = _makePortfolioSimulator(indexesPriceProvider, fxPriceProvider, referenceDate, daysForHistoricalCalc)

    adversionRiskValues = np.linspace(0.1, 300, 5000)

    #MARKOWITZ NO SHORT SELLING
    markowitzNoShortSelling = MarkowitzWeightsCalculator()
    markowitzNoShortSelling.addPortfolioSegment((0.0, 0.43), [0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 1.0])
    markowitzNoShortSelling.addPortfolioSegment((0.43, 1.61), [0.0, 0.0, -0.2272, 0.0, 0.2272],
                                                [0.0, 0.0, 0.5290, 0.0, 0.4710])
    markowitzNoShortSelling.addPortfolioSegment((1.61, 11.07), [-0.5448, 0.0, 0.0667, 0.0, 0.4781],
                                                [0.3380, 0.0, 0.3466, 0.0, 0.3154])
    markowitzNoShortSelling.addPortfolioSegment((11.07, np.inf), [1.4181, 0.0, 0.7086, -4.2147, 2.0880],
                                                [0.1606, 0.0, 0.2886, 0.3808, 0.1699])
    optimizerNoShortSelling = PortfolioOptimizer(portfolioSimulator, markowitzNoShortSelling)
    print("Optimizando Portfolio Sin venta en corto permitida")
    weights, oneDayVar = optimizerNoShortSelling.optimize(adversionRiskValues, maxOeDayVar=-0.015, periods=1, trajectories=100000)
    print("Var 1 dia {}%".format(oneDayVar * 100))
    weightsMsg = ['{}: {}'.format(ticker,weight) for ticker,weight in zip(market_constants.INDEXES_TICKERS,weights)]
    print("Composcion {}".format(', '.join(weightsMsg)))
    print("Rendimiento Esperado {}%".format(portfolioSimulator.expectedReturns(weights)*100))

    # MARKOWITZ SHORT SELLING
    markowitzWithShortSelling = MarkowitzWeightsCalculator()
    markowitzWithShortSelling.addPortfolioSegment((0.0, np.inf), [6.5669, -9.0299, 1.9549, -3.4759, 3.9840],
                                          [0.1672, -0.0026, 0.2657, 0.3801, 0.1895])
    optimizerWithShortSelling = PortfolioOptimizer(portfolioSimulator, markowitzWithShortSelling)

    print('------------------------------------------------------')
    print("Optimizando Portfolio Con venta en corto permitida")
    weights, oneDayVar = optimizerWithShortSelling.optimize(adversionRiskValues, maxOeDayVar=-0.015, periods=1, trajectories=100000)
    print("Var 1 dia {}%".format(oneDayVar * 100))
    weightsMsg = ['{}: {}'.format(ticker,weight) for ticker,weight in zip(market_constants.INDEXES_TICKERS,weights)]
    print("Composcion {}".format(', '.join(weightsMsg)))
    print("Rendimiento Esperado {}%".format(portfolioSimulator.expectedReturns(weights)*100))


def _filePath(*args):
    import os
    mainPath = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(mainPath,*args)

def _makePriceProvider(get_data_mode, tickers, providerName):
    if get_data_mode=='offline':
        return OfflinePriceDataProvider(_filePath('data',providerName), tickers)
    elif get_data_mode=='online':
        return OnlinePriceDataProvider(tickers, start=datetime.datetime(2015,1,1))

def _makeDataProvider(priceProvider):
    return EnhancedValuesProvider(priceProvider)

def _makePortfolioSimulator(indexesPriceProvider, fxPriceProvider, referenceDate, daysForHistoricalCalc):
    indexesDataProvider = _makeDataProvider(indexesPriceProvider)
    fxDataProvider = _makeDataProvider(fxPriceProvider)
    montecarloParams = MontecarloParameterProvider(indexesDataProvider, fxDataProvider, referenceDate=referenceDate,
                                                   historyInDays=daysForHistoricalCalc).montecarloParameters()
    montecarloSimulation = MontecarloMultipleStocks(montecarloParams.initialPricesVector,
                                                    montecarloParams.meanVector, montecarloParams.volatilityVector,
                                                    montecarloParams.corrMatrix)
    portfolioSimulation = PortfolioSimulation(montecarloSimulation)

    return portfolioSimulation



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
    print('-----------------------------------------------------------------------')
    print('8% return efficeint frontier with short selling:')
    weights = np.array([28.62, -16.62, 30.12, 31.71, 26.17]) / 100
    print('Weights {}'.format(weights))
    print("1 day var {}%".format(portfolioSimulation.oneDayVar(1, 10000, weights=weights) * 100))
    print("Expected Return {}%".format(portfolioSimulation.expectedReturns(weights=weights) * 100))
    print('-----------------------------------------------------------------------')
    print('Highest return efficeint frontier with short selling:')
    weights = np.array([91.10, -103.65, 48.70, 0, 63.85]) / 100
    print('Weights {}'.format(weights))
    print("1 day var {}%".format(portfolioSimulation.oneDayVar(1, 10000, weights=weights) * 100))
    print("Expected Return {}%".format(portfolioSimulation.expectedReturns(weights=weights) * 100))

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

def testPortfolio():
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

    return portfolioSimulation

def optimizePorfolio():
    import os
    daysForHistoricalCalc = 180 * 2
    referenceDate = datetime.date(2018, 6, 22)

    mainPath = os.path.dirname(os.path.realpath(__file__))
    indexesPath = os.path.join(mainPath, 'data', 'indexes')
    fxPath = os.path.join(mainPath, 'data', 'fx')
    stockProvider = EnhancedValuesProvider(OfflinePriceDataProvider(indexesPath, market_constants.INDEXES_TICKERS))
    ccyProvider = EnhancedValuesProvider(OfflinePriceDataProvider(fxPath, market_constants.CCY))
    montecarloParams = MontecarloParameterProvider(stockProvider, ccyProvider,
                                                   referenceDate=referenceDate,
                                                   historyInDays=daysForHistoricalCalc).montecarloParameters()

    montecarloSimulation = MontecarloMultipleStocks(montecarloParams.initialPricesVector,
                                                    montecarloParams.meanVector, montecarloParams.volatilityVector,
                                                    montecarloParams.corrMatrix)
    portfolioSimulation = PortfolioSimulation(montecarloSimulation)

    weightsCalculator = MarkowitzWeightsCalculator()
    weightsCalculator.addPortfolioSegment((0.0, np.inf), [6.5669, -9.0299, 1.9549, -3.4759, 3.9840],
                                          [0.1672, -0.0026, 0.2657, 0.3801, 0.1895])

    optimizer = PortfolioOptimizer( portfolioSimulation, weightsCalculator)
    weights, oneDayVar = optimizer.optimize(np.linspace(300,7,5000))
    print("Maximized Return {}".format(portfolioSimulation.expectedReturns(weights)))
