import numpy as np
import networksSocialLearning
import landscapes
import utils


class Simulation:

    def __init__(self, parameters):

        self.parameters = parameters
        self.numberOfAgents = parameters['numberOfAgents']
        self.distributeAgentsRandomly = parameters['distributeAgentsRandomly']
        self.numberOfAgentGroups = parameters['numberOfAgentGroups']
        self.groupSizeAgents = int(self.numberOfAgents/self.numberOfAgentGroups)
        self.groupSizeCompensation = self.numberOfAgents%self.numberOfAgentGroups
        self.numberOfTimeSteps = parameters['numberOfTimeSteps']
        self.fitnessLandscape = landscapes.FitnessLandscape(self.parameters)
        self.networkSocialLearning = networksSocialLearning.NetworkSocialLearning(self.parameters)
        self.payoffClassesAgents = np.zeros(self.numberOfAgents, dtype=np.int)
        self.payoffsAgentsOverTime = np.zeros((self.numberOfAgents, self.numberOfTimeSteps+1))
        self.itemsAgentsOverTime = np.zeros((self.numberOfAgents, self.numberOfTimeSteps+1), dtype=np.int)
        self.uniqueItemsOverTime = np.zeros(self.numberOfTimeSteps+1, dtype=np.int)
        self.arrivalTimesMaximumSolution = np.ones(self.numberOfAgents, dtype=np.int)*-1
        self.agentsNotArrived = list(range(self.numberOfAgents))
        self.N_NK = parameters['N_NKmodel']
        self.results = dict()

    def setInitialItemsAndPayoffsForAgents(self):

        for agentIdx in range(self.numberOfAgents):
            initialItem = int(np.random.choice(range(self.fitnessLandscape.numberOfItems)))
            self.itemsAgentsOverTime[agentIdx, 0] = initialItem
            self.payoffsAgentsOverTime[agentIdx, 0] = self.fitnessLandscape.landscape[self.payoffClassesAgents[agentIdx], initialItem]

        uniqueItems = np.unique(self.itemsAgentsOverTime[:, 0])
        self.uniqueItemsOverTime[0] = uniqueItems.shape[0]

    def distributeAgentsToGroups(self):

        if self.distributeAgentsRandomly:
            allAgents = list(range(self.numberOfAgents))

            for groupIdx in range(self.numberOfAgentGroups):

                if groupIdx == 0:
                    groupSize =  self.groupSizeAgents+self.groupSizeCompensation
                else:
                    groupSize = self.groupSizeAgents

                    for i in range(groupSize):
                        chosenIdx = np.random.choice(allAgents)
                        self.payoffClassesAgents[chosenIdx] = int(groupIdx)
                        allAgents.remove(chosenIdx)

        else:
            for groupIdx in range(self.numberOfAgentGroups):
                self.payoffClassesAgents[groupIdx*self.groupSizeAgents + self.groupSizeCompensation:(groupIdx+1)*self.groupSizeAgents + self.groupSizeCompensation] = groupIdx

    def getNeighboringItemEfficient(self, agentIdx, t):

        neighborsIdcs = np.nonzero(self.networkSocialLearning.networkSocialLearningNumpy[agentIdx, :])[0]
        itemsNeighbors = self.itemsAgentsOverTime[:,t][neighborsIdcs]
        focalPayoffsOfneighboringItems = self.fitnessLandscape.landscape[self.payoffClassesAgents[agentIdx], itemsNeighbors]
        return itemsNeighbors[np.argmax(focalPayoffsOfneighboringItems)], np.max(focalPayoffsOfneighboringItems)


    def innovateNKmodel(self, intState):

        binState = utils.reformatBinState(bin(intState), self.N_NK)
        randInt = np.random.choice(range(self.N_NK))
        flippedBinState = utils.spinFlipBinState(binState, randInt)
        flippedIntState = int(flippedBinState, 2)
        return flippedIntState

    def runEfficientSL(self):

        self.distributeAgentsToGroups()
        self.setInitialItemsAndPayoffsForAgents()

        for t in range(self.numberOfTimeSteps):

            for agentIdx in range(self.numberOfAgents):

                itemWithMaxFocalPayoff, maxFocalPayoff  = self.getNeighboringItemEfficient(agentIdx, t)

                if maxFocalPayoff > self.payoffsAgentsOverTime[agentIdx, t]:
                    self.itemsAgentsOverTime[agentIdx, t+1] = itemWithMaxFocalPayoff
                    self.payoffsAgentsOverTime[agentIdx, t+1] = maxFocalPayoff

                else:
                    innovatedItem = self.innovateNKmodel(self.itemsAgentsOverTime[agentIdx, t])
                    innovatedPayoff = self.fitnessLandscape.landscape[self.payoffClassesAgents[agentIdx], innovatedItem]

                    if innovatedPayoff > self.payoffsAgentsOverTime[agentIdx, t]:
                        self.itemsAgentsOverTime[agentIdx, t+1] = innovatedItem
                        self.payoffsAgentsOverTime[agentIdx, t+1] = innovatedPayoff
                    else:
                        self.itemsAgentsOverTime[agentIdx, t+1] = self.itemsAgentsOverTime[agentIdx, t]
                        self.payoffsAgentsOverTime[agentIdx, t+1] = self.payoffsAgentsOverTime[agentIdx, t]

                if (self.arrivalTimesMaximumSolution[agentIdx] < 0) and np.isclose(self.payoffsAgentsOverTime[agentIdx, t+1], 1.):
                    self.arrivalTimesMaximumSolution[agentIdx] = t+1

            uniqueItems = np.unique(self.itemsAgentsOverTime[:, t+1])
            self.uniqueItemsOverTime[t+1] = uniqueItems.shape[0]

        totalUniqueItems = np.unique(self.itemsAgentsOverTime)
        numberOfUniqueItemsAgents = self.computeNumberOfUniqueItemsAgents()

        self.results['payoffsAgentsOverTime'] = self.payoffsAgentsOverTime
        self.results['arrivalTimesMaximumSolution'] = self.arrivalTimesMaximumSolution
        self.results['uniqueItemsOverTime'] = self.uniqueItemsOverTime
        self.results['numberOfUniqueItemsAgents'] = numberOfUniqueItemsAgents
        self.results['totalNumberOfUniqueItems'] = totalUniqueItems.shape[0]
        self.results['numberOfTimeSteps'] = self.numberOfTimeSteps


    def computeNumberOfUniqueItemsAgents(self):

        numberOfUniqueItemsAgents = np.zeros(self.numberOfAgents, dtype=int)
        for agentIdx in range(self.numberOfAgents):
            itemsAgent = self.itemsAgentsOverTime[agentIdx, :]
            uniqueItems = np.unique(itemsAgent)
            numberOfUniqueItemsAgents[agentIdx] = uniqueItems.shape[0]

        return numberOfUniqueItemsAgents


