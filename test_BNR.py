import itertools
import numpy as np
import copy
import networkx as nx
from BNReasoner import BNReasoner
from pgmpy.inference import CausalInference
from VariableEliminate import VariableElimination
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
        
        BN_test = BNReasoner('testing/lecture_example.BIFXML')
        Q = list(split[0])
        e = dict(zip(list(split[1]), np.random.choice([True, False], size=len(split[1]))))

        reader = XMLBIFReader("testing/lecture_example.BIFXML")
        model = reader.get_model()
        res1 = VariableElimination(model).query(['Winter?', 'Rain?', "Wet Grass?"], evidence={'Slippery Road?': 'True'})
        print(res1)
        #res1['p']  = res1['p'] / float(res1['p'].sum())
        #print(res1)
        #print(Q,e)
        res2 = BN_test.marginalDistribution(['Winter?', 'Rain?', 'Wet Grass?'], {'Slippery Road?': False})
        res2['p'] = res2['p']/float(res2['p'].sum())
        print(res2)
        
        # try:
        #     BN_test.marginalDistribution(Q, e)
        # except:
        #     print('Error in marginalDistribution with Q = {} and e = {}'.format(Q, e))
        #     return False

    return True

def test_marginalDistribution3(BN):
    reader = XMLBIFReader("testing/lecture_example.BIFXML")
    model = reader.get_model()

    print(model)
    # infer = CausalInference(model)
    # print(infer.query(['Rain?', 'Sprinkler?']))
    print(VariableElimination(model).query(['Rain?', 'Sprinkler?'], evidence={'Winter?': 'False'}))

    ### Test marginalDistribution
    # Test for prior distribution
    
    print(BN.marginalDistribution(['Rain?', 'Sprinkler?'], {'Winter?': True}))


    # perms = []
    # for i in range(1,len(variable_set)):
    #     comb = itertools.combinations(variable_set, i)
       
    #     perms.append([list(c) for c in comb])
    

    # # Test conditionalDistribution for all splits
    # for perm in perms:
    #     for Q in perm:
    #         BN_test = copy.deepcopy(BN)
            
    #         BN_test.marginalDistribution(Q)
            
    #         # try:
    #         #     BN_test.marginalDistribution(Q)
    #         # except:
    #         #     print('Error in marginalDistribution with Q = {} and e = {}'.format(Q, None))
    #         #     return False

    return True
     
def test_dsep(BN):
    reader = XMLBIFReader("testing/lecture_example.BIFXML")
    model = reader.get_model()
    # assert BN.dSeperation(['Winter?'], ['Rain?'], ['Slippery Road?']) == (nx.d_separated(BN.bn.structure, {'Rain?'},{'Winter?'}, {'Slippery Road?'}))
    # assert BN.dSeperation(['Slippery Road?'], ['Rain?'], ['Winter?']) == (nx.d_separated(BN.bn.structure, {'Rain?'},{'Slippery Road?'}, {'Winter?'}))
    print(BN.dSeperation(['Sprinkler?'], ['Slippery Road?'], ['Winter?']) , (nx.d_separated(BN.bn.structure, {'Sprinkler?'},{'Slippery Road?'}, {'Winter?'})))
if __name__ == "__main__":
    BN = BNReasoner('testing/lecture_example.BIFXML')
    # BN.bn.draw_structure()
    test_dsep(BN)