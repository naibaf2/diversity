import matplotlib.pyplot as plt
import numpy as np
import simulation
import params


p = params.parameters()

sim = simulation.Simulation(p)
sim.runEfficientSL()

for agentIdx in range(sim.numberOfAgents):
    plt.plot(sim.payoffsAgentsOverTime[agentIdx, :], linewidth=0.2, color='k')

plt.plot(np.mean(sim.payoffsAgentsOverTime, axis=0), color='red', linewidth=2.5, label='Collective Performance')



plt.xlabel('Time')
plt.ylim([0,1.05])
plt.ylabel('Performance')
plt.legend()

plt.show()