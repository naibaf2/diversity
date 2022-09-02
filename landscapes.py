import utils
import numpy as np

class FitnessLandscape:

    def __init__(self, parameters):
        self.numberOfAgentGroups = parameters['numberOfAgentGroups']
        self.N_NKmodel = parameters['N_NKmodel']
        self.K_NKmodel = parameters['K_NKmodel']
        self.scaleNKFitness = parameters['scaleNKFitness']
        self.landscape = self.getLandscape()
        self.allItems, self.itemWithMaximumPayoff = self.getAllItemsFromLandscape()
        self.numberOfItems = len(self.allItems)


    def getAllItemsFromLandscape(self):

        allItems = []
        numberOfItems = self.landscape.shape[1]

        for itemIdx in range(numberOfItems):

            newItem = utils.Item(itemIdx, self.landscape[:, itemIdx])
            allItems.append(newItem)

        return allItems, np.max(self.landscape, axis=0)

    def getLandscape(self):

        NKPayoffs = np.zeros((self.numberOfAgentGroups, 2**self.N_NKmodel))

        for agentGroupIdx in range(self.numberOfAgentGroups):

            NKPayoffIdx = utils.getStatesPayoffsNKmodel(self.N_NKmodel, self.K_NKmodel, utils.getSubcombinationPayoffValues(self.N_NKmodel, self.K_NKmodel))
            NKPayoffIdx /= np.max(NKPayoffIdx)
            NKPayoffIdx = NKPayoffIdx**8
            NKPayoffs[agentGroupIdx, :] = NKPayoffIdx

        return NKPayoffs


