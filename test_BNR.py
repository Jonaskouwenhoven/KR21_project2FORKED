import itertools
import numpy as np
import pandas as pd
import copy
import networkx as nx
from BNReasoner import BNReasoner
from pgmpy.inference import CausalInference
from VariableEliminate import VariableElimination
from pgmpy.inference import EliminationOrder
from pgmpy.readwrite import XMLBIFReader
import pgmpy

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
        
        BN_test = BNReasoner('testing/dog_problem.BIFXML')
        Q = list(split[0])
        e = dict(zip(list(split[1]), np.random.choice([True, False], size=len(split[1]))))
    
        try:
            BN_test.marginalDistribution(Q, e)

        except:
            print('Error in marginalDistribution with Q = {} and e = {}'.format(Q, e))
            return False

    return True

def test_marginalDistribution3(BN):
    variable_set = BN.bn.get_all_variables()

    ### Test marginalDistribution
    # Test for prior distribution
    perms = []
    for i in range(1,len(variable_set)):
        comb = itertools.combinations(variable_set, i)
       
        perms.append([list(c) for c in comb])
    

    # Test conditionalDistribution for all splits
    for perm in perms:
        for Q in perm:
            BN_test = copy.deepcopy(BN)
            
            BN_test.marginalDistribution(Q)
            
            try:
                BN_test.marginalDistribution(Q)
            except:
                print('Error in marginalDistribution with Q = {} and e = {}'.format(Q, None))
                return False

    return True
     
def test_dsep(BN):
    assert (BN.dSeperation(['Winter?'], ['Rain?'], ['Slippery Road?']) == (nx.d_separated(BN.bn.structure, {'Winter?'},{'Rain?'}, {'Slippery Road?'})))
    assert (BN.dSeperation(['Slippery Road?'], ['Rain?'], ['Winter?'])  ==  (nx.d_separated(BN.bn.structure, {'Rain?'},{'Slippery Road?'}, {'Winter?'})))
    assert (BN.dSeperation(['Sprinkler?'], ['Slippery Road?'], ['Winter?']) ==  (nx.d_separated(BN.bn.structure, {'Sprinkler?'},{'Slippery Road?'}, {'Winter?'})))

def test_ind(BN):

    assert (BN.independence(['Winter?'], ['Rain?'], ['Slippery Road?']) == (nx.d_separated(BN.bn.structure, {'Winter?'},{'Rain?'}, {'Slippery Road?'})))



def test_prune(BN):
    # TODO: HOW can we test this?
    pass

def test_marg(BN):

    newDf = (BN.marginalization('Winter?', BN.bn.get_cpt('Rain?')))
    assert newDf['p'].to_list()[0] == 1.1
    assert newDf['p'].to_list()[1] == 0.9
    
def test_maxingout(BN):

    maxed_out = (BN.maxingOut('Rain?'))
    assert maxed_out.loc[maxed_out['Winter?'] == True]['p'].to_list()[0] == 0.8
    assert maxed_out.loc[maxed_out['Winter?'] == False]['p'].to_list()[0] == 0.9

def test_fact_mult(BN):
    ## Check for commutative property
    assert (BN.factorMultiplication(BN.bn.get_cpt('Rain?'), BN.bn.get_cpt('Winter?'))['p'].to_list() ==  BN.factorMultiplication(BN.bn.get_cpt('Winter?'), BN.bn.get_cpt('Rain?'))['p'].to_list())
    assert BN.factorMultiplication(BN.bn.get_cpt('Rain?'), BN.bn.get_cpt('Winter?'))['p'].to_list()[-1] == 0.48
    
def test_ordering(BN):
    variables = ['Rain?', 'Winter?', 'Slippery Road?']
    evidence = {'Wet Grass?': True}
    elimination_order = 'MinFill'
    # Not quite sure how to test
    pass

def test_MAP(BN):
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
        
        BN_test = BNReasoner('testing/dog_problem.BIFXML')
        Q = list(split[0])
        e = dict(zip(list(split[1]), np.random.choice([True, False], size=len(split[1]))))
    
        try:
            BN_test.MAP(Q, e, 'min_degree')

        except:
            print('Error in marginalDistribution with Q = {} and e = {}'.format(Q, e))
            assert False

    assert True
   

def test(BN):
    #test_marginalDistribution(BN)
    #test_marginalDistribution2(BN)
    # test_marginalDistribution3(BN)
    # test_dsep(BN) ## Correct
    # test_ind(BN) ## Correct
    # test_prune(BN) ## Not sure
    # test_marg(BN) ## Works
    # test_maxingout(BN) ## Works
    # test_fact_mult(BN) ## Works
    # test_ordering(BN) ## Not Sure
    test_MAP(BN)
    pass
    
if __name__ == "__main__":
    BN = BNReasoner('testing/dog_problem.BIFXML')
    # BN.bn.draw_structure()
    test(BN)