import numpy as np

class MarkowitzWeightsCalculator:

    def __init__(self):
        self._segments = []

    def addPortfolioSegment(self, coeffAInterval, c, d):
        self._segments.append((coeffAInterval,c,d))

    def _computeWeight(self, a, c,d ):
        return ((1.0/a)*c) + d

    def weights(self, relativeRiskAdversion):
        segment = self._findSegment(relativeRiskAdversion)
        return [self._computeWeight(relativeRiskAdversion,c,d) for c,d in zip(segment[1],segment[2])]

    def _findSegment(self, relativeRiskAdversion):
        for aSegment in self._segments:
            interval = aSegment[0]
            if relativeRiskAdversion>=interval[0] and relativeRiskAdversion<interval[1]:
                return aSegment

        raise Exception('No segment found for risk adversion {}'.format(relativeRiskAdversion))

class PortfolioOptimizer:
    def __init__(self,portfolioSimulator, markowitzCalculator):
        self.portfolioSimulator = portfolioSimulator
        self.markowitzCalculator = markowitzCalculator

    def optimize(self, riskAdversionValues, maxOeDayVar=-0.015, holdSimulation=True, periods=1, trajectories=100000):

        self.portfolioSimulator.holdSimulation=holdSimulation
        for riskAdversion in riskAdversionValues:
            weights = self.markowitzCalculator.weights(riskAdversion)
            oneDayVarsSim = self.portfolioSimulator.oneDayVar(periods, trajectories, weights=weights)
            if any( oneDayVarsSim>maxOeDayVar):
                #print("Risk adversion {}, weights {}, one day var {}".format(riskAdversion, weights, oneDayVarsSim))
                return weights, oneDayVarsSim

        #print("Risk adversion {}, weights {}, one day var {}".format(riskAdversion, weights, oneDayVarsSim))
        return weights, oneDayVarsSim


def shortSellingMarkowitz():
    #Values obtained with external program
    weightsCalculator = MarkowitzWeightsCalculator()
    weightsCalculator.addPortfolioSegment( (0.0,np.inf), [6.5669,-9.0299,1.9549,-3.4759,3.9840], [0.1672, -0.0026, 0.2657, 0.3801, 0.1895] )

    return weightsCalculator
