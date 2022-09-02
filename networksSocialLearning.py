import networkx as nx
import numpy as np
import utils
import matplotlib.pyplot as plt




class NetworkSocialLearning:

    def __init__(self, parameters):

        self.parameters = parameters
        self.numberOfAgents = parameters['numberOfAgents']
        self.numberOfAgentGroups = parameters['numberOfAgentGroups']
        self.numberOfNeighbors = parameters['numberOfNeighbors']
        self.typeOfNetworkSocialLearning = parameters['typeOfNetworkSocialLearning']
        self.k_wsNetwork = parameters['k_wsNetwork']
        self.p_wsNetwork = parameters['p_wsNetwork']
        self.p_erNetwork = parameters['p_erNetwork']
        self.config_delta_deg = parameters['config_delta_deg']
        self.networkSocialLearningNetworkX = self.getNetworkSocialLearning()
        self.networkSocialLearningNumpy = np.array(nx.to_numpy_array(self.networkSocialLearningNetworkX)) # numpy array


    def getNetworkSocialLearning(self):

        if self.typeOfNetworkSocialLearning == 'ws':
            network = nx.watts_strogatz_graph(self.numberOfAgents, self.k_wsNetwork, self.p_wsNetwork)
            return network

        if self.typeOfNetworkSocialLearning == 'er':
            network = nx.erdos_renyi_graph(self.numberOfAgents, self.p_erNetwork)
            while not nx.is_connected(network):
                network = nx.erdos_renyi_graph(self.numberOfAgents, self.p_erNetwork)
            return network

        if self.typeOfNetworkSocialLearning == 'config_delta':
            deg = [self.config_delta_deg] * self.numberOfAgents
            network = nx.configuration_model(deg)
            network.remove_edges_from(nx.selfloop_edges(network))
            while not nx.is_connected(network):
                network = nx.configuration_model(deg)
                network.remove_edges_from(nx.selfloop_edges(network))
            return network

        if self.typeOfNetworkSocialLearning == 'ba':
            network = nx.barabasi_albert_graph(self.numberOfAgents, m=3)
            return network

        if self.typeOfNetworkSocialLearning == 'line':
            network = nx.watts_strogatz_graph(self.numberOfAgents, k=2, p=0)
            network.remove_edge(self.numberOfAgents-2,self.numberOfAgents-1)
            return network

        if self.typeOfNetworkSocialLearning == 'completeGraph':
            network = nx.complete_graph(self.numberOfAgents)
            return network
