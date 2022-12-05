import itertools
import numpy as np
import copy

def test_marginalDistribution(BN):
    variable_set = BN.bn.get_all_variables()
    perms = list(itertools.permutations(variable_set))
    choices = np.arange(1, len(variable_set))

    splits = set()
    for perm in perms:
        for choice in choices:
            splits.add(tuple([frozenset(perm[:choice]), frozenset(perm[choice:])]))

    splits = list(splits)
    
    
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
     



