def parameters():

    p = {}
    p['numberOfAgents'] = 50
    p['distributeAgentsRandomly'] = True
    p['numberOfAgentGroups'] = 1
    p['numberOfNeighbors'] = 3
    p['numberOfTimeSteps'] = 200
    p['strategySocialLearning'] = 'bestMember'
    p['N_NKmodel'] = 8
    p['K_NKmodel'] = 0
    p['scaleNKFitness'] = True
    p['typeOfNetworkSocialLearning'] = 'ws' # ['er', 'ws', 'config_delta', 'completeGraph', 'line']
    p['k_wsNetwork'] = 4
    p['p_wsNetwork'] = 0.1
    p['p_erNetwork'] = .005
    p['config_delta_deg'] = 4

    return p