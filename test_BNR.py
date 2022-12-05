import itertools
import numpy as np
import copy
from BNReasoner import BNReasoner

def test_marginalDistribution1(BN):
    ### Test marginalDistribution
    # Take combinations of all variables 
    variable_set = BN.bn.get_all_variables()
    perms = list(itertools.permutations(variable_set))
    choices = np.arange(1, len(variable_set))

    # Take splits of the variable set
    splits = set()
    for perm in perms:
        for choice in choices:
            splits.add(tuple([frozenset(perm[:choice]), frozenset(perm[choice:])]))

    splits = list(splits)
    
    # Test marginalDistribution for all splits
    for split in splits:
        BN_test = copy.deepcopy(BN)
        Q = list(split[0])
        e = dict(zip(list(split[1]), np.random.choice([True, False], size=len(split[1]))))

        try:
            BN_test.marginalDistribution(Q, e)
        except:
            print('Error in marginalDistribution with Q = {} and e = {}'.format(Q, e))
            return False

    return True
     

def test_marginalDistribution2(BN):
    ### Test marginalDistribution
    # Take combinations of all variables 
    variable_set = BN.bn.get_all_variables()
    combs = list(itertools.combinations(variable_set, 2))
    choices = np.arange(1, 2)

    # Take splits of the variable set
    splits = set()
    for comb in combs:
        for choice in choices:
            splits.add(tuple([frozenset(comb[:choice]), frozenset(comb[choice:])]))

    splits = list(splits)
    
    # Test marginalDistribution for all splits
    for split in splits:
        
        BN_test = BNReasoner('testing/lecture_example.BIFXML')
        Q = list(split[0])
        e = dict(zip(list(split[1]), np.random.choice([True, False], size=len(split[1]))))
        print("Test:" ,Q, e)
        BN_test.marginalDistribution(Q, e)
        # try:
        #     BN_test.marginalDistribution(Q, e)
        # except:
        #     print('Error in marginalDistribution with Q = {} and e = {}'.format(Q, e))
        #     return False

    return True
     


