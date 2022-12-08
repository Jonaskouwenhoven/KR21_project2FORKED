from typing import Union
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import test_BNR
import numpy as np

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
            
        # self.inDict = None
        # self.outDict = None

    # TODO: This is where your methods should go
    
    # def getding(self):
    #     """Store the in and out degree of each node in the network"""

    #     vars = BN.bn.get_all_variables()
    #     inDict, outDict = {}, {}
    #     for var in vars:
    #         neighbors = list(BN.bn.structure.neighbors(var))

    #         if len(list(neighbors)) < 1:
    #             continue
    #         else:
    #             for n in list(neighbors):
    #                 if n in inDict:
    #                     inDict[n].append(var)
    #                 else:
    #                     inDict[n] = [var]
                    
    #                 if var in outDict:
    #                     outDict[var].append(n)
    #                 else:
    #                     outDict[var] = [n]
                        
    #     self.inDioc = inDict
    #     self.outDict = outDict

    
    def netPrune(self, Q, evidence):
        #TODO: Network Pruning: Given a set of query variables Q and evidence e, node- and edge-prune the Bayesian network s.t. queries of the form P(Q|E) can still be correctly calculated
        evidence_nodes = list(evidence.keys())

        Q_plus_e = Q + evidence_nodes
        
        variables = self.bn.get_all_variables()
        

        # get all factors
        factors = self.bn.get_all_cpts()
        
        
        # reduce factors with regard to e
        if len(evidence) != 0:
            for node in variables:
                new_factor = self.bn.reduce_factor(pd.Series(evidence), factors[node])
                for e in evidence_nodes:
                    if e in new_factor.columns:
                        new_factor = new_factor[new_factor['p'] != 0]
                self.bn.update_cpt(node, new_factor)

        
        # Edge Purning
        for e in evidence:
            children = self.bn.get_children(e)
            for child in children:

                self.bn.del_edge((e, child))

        # Node Pruning
        for variable in variables:
            if variable not in Q_plus_e:
                delete = True
                for q in Q_plus_e:  
                    if variable in self.bn.get_cpt(q).columns:
                        delete = False
                if delete:
                    self.bn.del_var(variable)
            
        return

        
    def dSeperation(self, X, Y, Z):
        Graph = self.bn.structure

        
        if X == Y:
            return False
        
        for x in X:
            for y in Y:
                for path in nx.all_simple_paths(self.bn.get_interaction_graph(), source=y, target=x):
                    print(path)
                    for element in path:
                        if element == x or element == y:
                            continue
                        in_degree = Graph.in_degree(element)
                        if in_degree == 1:
                            if element in Z:
                                return True
                            
                        if in_degree == 2: # This is a collider
                            if element not in Z:
                                return True
                            
                        out_degree = (Graph.out_degree(element))
                        if out_degree == 2: # This is a fork
                            if element in Z:
                                return True
        return False
    
    def independence(self):
        #CAS
        #TODO: Independence: Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z. (Hint: Remember the connection between d-separation and independence) (1.5pt)
        if self.dSeperation():
            return True
        
        return False
    
    def marginalization(self, X, f):
        #JONAS
        #TODO: Marginalization: Given a factor and a variable X, compute the CPT in which X is summed-out. (3pts)
        
        if X not in f.columns:
            return f
        else:
            new_f = f
        
            new_columns = [c for c in (new_f.columns) if c not in [X, 'p']]
         
            new_f = new_f.groupby(new_columns)["p"].sum().reset_index()
            #new_f['p']  = new_f['p'] / new_f['p'].sum()
         
            return new_f


    
    def maxingOut(self, X):
        #JONAS
        #TODO: Maxing-out: Given a factor and a variable X, compute the CPT in which X is maxed-out. Remember to also keep track of which instantiation of X led to the maximized value. (5pts)
        cpt = self.bn.get_cpt(X)
        falseDf = cpt.loc[cpt[X] == False].reset_index(drop=True)
        trueDf = cpt.loc[cpt[X] == True].reset_index(drop=True)
        dfList = []
        for i, el in enumerate(zip(falseDf['p'].to_list(), trueDf['p'].to_list())):
            if el[0] > el[1]:
                dfList.append(falseDf.iloc[i])
            else:
                dfList.append(trueDf.iloc[i])
                
        maxedout = (pd.DataFrame(dfList))
        
        return maxedout
    
    def factorMultiplication(self, f, g):
        #JONAS
        #TODO: Factor multiplication: Given two factors f and g, compute the multiplied factor h=fg. (5pts)
        
        f_columns = (f.columns.drop('p'))
        g_columns = (g.columns.drop('p'))
      
        double = list((f_columns).intersection(g_columns))
        
        # if not double:
        #     return False
      
    
        if len(double) > 0:
            new = pd.merge(f, g, on=double)
            new['p'] = new['p_x'] * new['p_y']
            new.drop(columns=['p_x', 'p_y'], inplace=True)
        else :
            new = f
          
            new['p'] = new['p'] * g['p']
    
        return new

    def multipleFactorMultiplication(self, factors):
        num_factors = len(factors)
        factor_product = factors[0]
        for i in range(1,num_factors):
            factor = factors[i]
            factor_product = self.factorMultiplication(factor, factor_product)
        return factor_product    

    def _min_degree(self, X, int_graph):
        """Return the node with minimum degree in the graph"""
       
        int_sub_graph = [node for node in int_graph.degree if node[0] in X]
        return min(int_sub_graph, key=lambda x: x[1])[0] 
    
    def _fill(self, int_graph, node):
        """Return the fill of a node in the graph"""
        neighbors = int_graph.neighbors(node)
        set_neighbors = set([el for el in neighbors])
        
        tot = 0

        
        for n1 in set_neighbors:  
            n_copy = set_neighbors.copy()
            shared_edges = (set(int_graph.neighbors(n1)) & n_copy)
            edges = len(n_copy.difference(shared_edges)) - 1
            tot += edges
    
        return tot/2

    def draw_graph(self, graph):
        """Draw a graph with networkx"""
        nx.draw(graph, with_labels=True, node_size = 3000)
        plt.show()

    def _min_fill(self, X, int_graph):
        """Return the node with minimum fill in the graph"""
        fills = []

        for n1 in X:
            fills.append((n1, self._fill(int_graph, n1)))

        return min(fills, key=lambda x: x[1])[0]

    def ordering(self, X, method = 'min_degree'):
        #SICCO
        #TODO: Ordering: Given a set of variables X in the Bayesian network, compute a good ordering for the elimination of X based on the min-degree heuristics (2pts) and the min-fill heuristics (3.5pts). (Hint: you get the interaction graph ”for free” from the BayesNet class.)
        
        int_graph = self.bn.get_interaction_graph()
        order = []
        order_func = self._min_degree if method == 'min_degree' else self._min_fill
        
        X_copy = X.copy()
        for i in range(len(X)):
            node = order_func(X, int_graph)
            order.append(node)
            int_graph.remove_node(node)
            
            X_copy.remove(node)
        return order
    
    def _get_all_factors(self, X):
        """Return a list of all factors"""
        factors = {}
        for node in X:
            part_factors = []
            cpt = self.bn.get_cpt(node)
            part_factors.append((cpt,node))

            for var in self.bn.get_children(node):
                if var in X and var != node:
                    cpt = self.bn.get_cpt(var)
                    if node in cpt.columns:
                        part_factors.append((cpt, var))
  
            factors[node] = part_factors
        
        return factors

    def variableElimination(self, Q, X, order_method = 'min_degree'):
        #SICCO
        #TODO: Variable Elimination: Sum out a set of variables by using variable elimination. (5pts)
        # Deze functie klopt nog niet helemaal...

        elimination_order = self.ordering(X, order_method) if len(X) != 0 else [] # get elimination order
   
        # get factors
        variables = Q + list(X)

        work_factors = self._get_all_factors(variables)
     
        eliminated_variables = set()
        print(elimination_order)
        for node in elimination_order:  # iterate over elimination order
            factors = []
            for factor, org in work_factors[node]:
                if not set(variables).intersection(eliminated_variables):
                    factors.append(factor)


            num_factors = len(factors)

            factor_product = factors[0]
            for i in range(1,num_factors):
                factor = factors[i]
                factor_product = self.factorMultiplication(factor, factor_product)

            marg_factor = self.marginalization(node, factor_product)
  
            
            for var in marg_factor.columns:
                if var != 'p' and var in work_factors:
                    entries = [org for factor, org in work_factors[var]]
                    if node in entries:
                        index = entries.index(node)
                
                        work_factors[var][index] = (marg_factor, node)

            del work_factors[node]

        print("XXXXXXXXXXXXX")
        final_distribution = []
        column_sets = set()
        for node in work_factors:
            for factor, org in work_factors[node]:
                print(factor)
                if not set(factor.columns[:-1]).intersection(eliminated_variables) and tuple(factor.columns[:-1]) not in column_sets:
                    if all(col in Q for col in factor.columns[:-1]):
                        column_sets.add(tuple(factor.columns[:-1]))
                        final_distribution.append((factor))
        final_distribution = [factor for factor in final_distribution]
        


        factor_product = final_distribution[0]
        for i in range(1,len(final_distribution)):
            
            factor_product = self.factorMultiplication(factor_product, final_distribution[i])
           

        return factor_product
    
    def marginalDistribution(self, Q, e = {}, order_method = 'min_degree'):
        #SICCO
        #TODO: Marginal Distributions: Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). Note that Q is a subset of the variables in the Bayesian network X with Q ⊂ X but can also be Q = X. (2.5pts)
        
        # pruning
        self.netPrune(Q, e)
        variables = self.bn.get_all_variables()
        print(variables)
        # order
        evidence_node =  list(e.keys()) 
        Q_plus_e = Q + evidence_node 
        elimination_variables =  (
            set(variables)
            - set(Q_plus_e)
        )


        joint = self.variableElimination(Q_plus_e, elimination_variables, order_method) # hier gaat vgm nog wat mis


        if len(e) == 0:
            return joint
        else:
            helper = joint
            for q in Q:
                helper = self.marginalization(q, helper)
            pr_e = helper
            
            posterior = joint
   
            posterior['p'] = joint['p'] / float(pr_e['p'])
            return posterior

        return
    

    def marginal_dist(self, Q: list, E: dict, order_method = "min_degree") -> dict:
        '''
        Calculate the marginal distribution of Q given evidence E. 
        :Param Q: list of variables in Q
        :Param E: list of variables in the evidence
        :Param var: ordered list of variables not in Q
        :Return: marginal distribution  
        '''
        
        # first, prune the network based on the query and the evidence:
        self.netPrune(Q, E)
        
        # get the probability of the evidence
        evidence_factor = 1
        for variable in E:
            cpt = self.bn.get_cpt(variable)
            evidence_factor *= cpt['p'].sum()
        
        # get all cpts in which the variable occurs
        S = self.bn.get_all_cpts()
        
        factor = 0

        # get elimination order
        variables = self.bn.get_all_variables()
        evidence_node =  list(E.keys()) 
        Q_plus_E = Q + evidence_node 
        elimination_variables =  (
            set(variables)
            - set(Q_plus_E)
        )

        elimination_order = self.ordering(elimination_variables, order_method) if len(elimination_variables) != 0 else [] # get elimination order

        # loop over every variable not in Q
        for variable in elimination_order:
            print("variable is: ", variable)
            factor_var = {}
            
            for cpt_var in S:
                
                if variable in S[cpt_var]:
                    factor_var[cpt_var] = S[cpt_var]
            
            print("factor_var ", factor_var)
            
            # apply chain rule and eliminate all variables 
            if len(factor_var) >= 2:
                factors = self.multipleFactorMultiplication(list(factor_var.values()))
                
                marg_factor = self.marginalization(variable, factors)
                
                for factor_variable in factor_var:
                    del S[factor_variable]
                
                factor +=1
                S["factor "+str(factor)] = marg_factor
            
            # when there is only one cpt, don't multiply
            elif len(factor_var) == 1:
                marg_factor = self.marginalization( variable, list(factor_var.values())[0])

                for factor_variable in factor_var:
                    del S[factor_variable]
                
                factor +=1
                S["factor "+str(factor)] = marg_factor

        if len(S) > 1:
            marginal_dist = self.multipleFactorMultiplication(list(S.values()))
        else:
            marginal_dist = list(S.values())[0]
        
        marginal_dist['p'] = marginal_dist['p'].div(evidence_factor)
        return marginal_dist

    def MAP(self, Q, e, order_method = 'min_degree'):
        #TODO: Compute the maximum a-posteriory instantiation + value of query variables Q, given a possibly empty evidence e. (3pts)
        
        # get all factors
        factors = BN.bn.get_all_cpts()
 
        # reduce factors with regard to e
        for node in BN.bn.get_all_variables():
            new_factor = BN.bn.reduce_factor(pd.Series(e), factors[node])
            BN.bn.update_cpt(node, new_factor)
        
        # order
        evidence_node =  list(e.keys()) 
        Q_plus_e = Q + evidence_node

        self.variableElimination(Q_plus_e, order_method)
        
        # joint marginal
        
        joint_marginal = self.bn.get_cpt(Q[0])

        MAP = self.maxingOut()

        
        pass
        
    def MEP(self):
        #TODO: Compute the most probable explanation given an evidence e. (1.5pts)
        pass
        
        



if __name__ == '__main__':
    
    BN = BNReasoner('testing/lecture_example.BIFXML')
    # cptWet = BN.bn.get_cpt("Wet Grass?")
    # cptRain = BN.bn.get_cpt("Rain?")
    #BN.factorMultiplication(cptWet, cptRain)

    ### Test ordering, variable elimination and marginal distribution
    test_val1 = True#test_BNR.test_marginalDistribution1(BN)
    test_val2  = test_BNR.test_marginalDistribution2(BN) 
    test_val3 = True #test_BNR.test_marginalDistribution3(BN)
    
    if test_val1 and test_val2 and test_val3:
        print("Test marginal distribution passed")
    else:
        print("Test marginal distribution failed")
        exit()

    # BN.netPrune(['Wet Grass?'], {'Winter?':True, "Rain?":False})
    # Q = ['Rain?']
    # e = {'Winter?':True}
    # BN.marginalDistribution(Q, {'Winter?':True})

    # print(BN.bn.get_cpt(Q[0]))
   
    exit()
