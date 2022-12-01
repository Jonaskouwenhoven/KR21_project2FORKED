from typing import Union
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net
            
        self.inDict = None
        self.outDict = None

    # TODO: This is where your methods should go
    
    def getding(self):
        """Store the in and out degree of each node in the network"""

        vars = BN.bn.get_all_variables()
        inDict, outDict = {}, {}
        for var in vars:
            neighbors = list(BN.bn.structure.neighbors(var))

            if len(list(neighbors)) < 1:
                continue
            else:
                for n in list(neighbors):
                    if n in inDict:
                        inDict[n].append(var)
                    else:
                        inDict[n] = [var]
                    
                    if var in outDict:
                        outDict[var].append(n)
                    else:
                        outDict[var] = [n]
                        
        self.inDioc = inDict
        self.outDict = outDict

    
    def netPrune(self):
        #TODO: Network Pruning: Given a set of query variables Q and evidence e, node- and edge-prune the Bayesian network s.t. queries of the form P(Q|E) can still be correctly calculated
        pass

        
    def dSeperation(self, X, Y, Z):
        #TODO: d-Separation: Given three sets of variables X, Y, and Z, determine whether X is d-separated of Y given Z. (4pts)
        if X == Y:
            # In the situation that X is equal to Y, X is not d-separated from Y given Z
            return False
        
        
        pass
    
    def independence(self):
        #TODO: Independence: Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z. (Hint: Remember the connection between d-separation and independence) (1.5pt)
        if self.dSeperation():
            return True
        
        return False
    
    def marginalization(self):
        #TODO: Marginalization: Given a factor and a variable X, compute the CPT in which X is summed-out. (3pts)
        pass
    
    def maxingOut(self):
        #TODO: Maxing-out: Given a factor and a variable X, compute the CPT in which X is maxed-out. Remember to also keep track of which instantiation of X led to the maximized value. (5pts)
        pass
    
    def factorMultiplication(self, f, g):
        #TODO: Factor multiplication: Given two factors f and g, compute the multiplied factor h=fg. (5pts) HALLLLOOOOOTEST
        
        pass
    
    def ordering(self):
        #TODO: Ordering: Given a set of variables X in the Bayesian network, compute a good ordering for the elimination of X based on the min-degree heuristics (2pts) and the min-fill heuristics (3.5pts). (Hint: you get the interaction graph ”for free” from the BayesNet class.)
        pass
    
    def variableElimination(self):
        #TODO: Variable Elimination: Sum out a set of variables by using variable elimination. (5pts)
        pass
    
    def marginalDistribution(self):
        #TODO: Marginal Distributions: Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). Note that Q is a subset of the variables in the Bayesian network X with Q ⊂ X but can also be Q = X. (2.5pts)
        pass
    
    def MAP(self):
        #TODO: Compute the maximum a-posteriory instantiation + value of query variables Q, given a possibly empty evidence e. (3pts)
        pass
        
    def MEP(self):
        #TODO: Compute the most probable explanation given an evidence e. (1.5pts)
        pass
        
        



if __name__ == '__main__':
    
    BN = BNReasoner('testing/dog_problem.BIFXML')
    # BN.getding()
    vars = BN.bn.get_all_variables()
    inDict, outDict = {}, {}
    for var in vars:
        neighbors = list(BN.bn.structure.neighbors(var))
        print(neighbors, var)
        
    print(BN.bn.all_simple_paths('hear-bark', 'family-out'))
    BN.bn.draw_structure()
    exit()
    # vars = BN.bn.get_all_variables()
    # for var in vars:
    #     print(var)
    #     print(BN.bn.get_all_cpts()[var])
    #     exit()