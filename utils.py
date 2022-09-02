import numpy as np
import itertools


class Item:

    def __init__(self, coordinates, payoffs):
        self.coordinates = coordinates
        self.payoffs = payoffs

    def __lt__(self, other):
        return self.payoffs[0] < other.payoffs[0]


def spinFlipBinState(binState, positionOfSpinFlip):

    listBinState = list(binState)

    if listBinState[positionOfSpinFlip] == '0':
        listBinState[positionOfSpinFlip] = '1'
        binState = "".join(listBinState)
        return binState

    else:
        listBinState[positionOfSpinFlip] = '0'
        binState = "".join(listBinState)
        return binState


def reformatBinState(binState, N_NK):

    binState = binState.split('b')[1]
    lenBinState = len(binState)
    assert lenBinState <= N_NK, 'check: something must be wrong'

    binState = '0'*(N_NK-lenBinState) + binState
    return binState


def getStatesPayoffsNKmodel(N_NK, K_NK, dictSubcombinationPayoffValues):

    interactionMatrixNKmodel = getRandomInteractionMatrixNKmodel(N_NK, K_NK)

    payoffsNK = np.zeros(2**N_NK)

    for intState in range(2**N_NK):

        payoffsSingleStates = np.zeros(N_NK)
        binState = reformatBinState(bin(intState), N_NK)

        for digit_idx, digit in enumerate(binState):
            combinationIdcs = np.nonzero(interactionMatrixNKmodel[digit_idx,:])[0]
            string = digit
            for combinationIdx in combinationIdcs:
                string += binState[combinationIdx]
            payoffsSingleStates[digit_idx] = dictSubcombinationPayoffValues[string]

        payoffsNK[intState] = np.mean(payoffsSingleStates)

    return payoffsNK



def getSubcombinationPayoffValues(N_NK, K_NK):

    assert 0<= K_NK < N_NK

    dictCombinationsPayoffs = dict()
    combinations = list(itertools.product('01', repeat=K_NK+1))

    for combination in combinations:
        string = ''
        for digit in combination:
            string += digit
        dictCombinationsPayoffs[string]  = np.random.uniform(0,1)

    return dictCombinationsPayoffs



def getRandomInteractionMatrixNKmodel(N_NK, K_NK):

    idcs_mat = np.zeros((N_NK, N_NK), dtype=np.int)
    for i in np.arange(N_NK):
        idcs = list(range(N_NK))
        idcs.remove(i)
        chosen_ones = np.random.choice(idcs, size=K_NK, replace=False)
        for j in chosen_ones:
            idcs_mat[i, j] = 1

    return idcs_mat



