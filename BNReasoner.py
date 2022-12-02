from typing import Union
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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

    
    def netPrune(self,Q, evidence):
        #TODO: Network Pruning: Given a set of query variables Q and evidence e, node- and edge-prune the Bayesian network s.t. queries of the form P(Q|E) can still be correctly calculated
        
        variables = self.bn.get_all_variables()
        for key in evidence.keys():
            variables.remove(key)

        
        for key, value in zip(evidence.keys(), evidence.values()):
            for var in variables:
                cpt = self.bn.get_cpt(var)
                if key not in cpt.columns:
                    continue
                instantation = pd.Series({key:value})
                cit = self.bn.get_compatible_instantiations_table(instantation, cpt)
                # crt = self.bn.reduce_factor(instantation, cpt)
                self.bn.update_cpt(key, cit)
                print(cpt)
                print(cit)
            self.bn.del_var(key)
            
            

        self.bn.draw_structure()
                # print(crt)
        pass

        
    def dSeperation(self, X, Y, Z):
        #CAS
        #TODO: d-Separation: Given three sets of variables X, Y, and Z, determine whether X is d-separated of Y given Z. (4pts)
        if X == Y:
            # In the situation that X is equal to Y, X is not d-separated from Y given Z
            return False
        
        
        pass
    
    def independence(self):
        #CAS
        #TODO: Independence: Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z. (Hint: Remember the connection between d-separation and independence) (1.5pt)
        if self.dSeperation():
            return True
        
        return False
    
    def marginalization(self, X):
        #JONAS
        #TODO: Marginalization: Given a factor and a variable X, compute the CPT in which X is summed-out. (3pts)
        #NOTE: Hier moet nog wat aan gedaan worden. Er staat given a "factor" and a varaible X, maar nu gebruik je die factor niet. je moet ook de andere cpt aanpassen
        cpt = self.bn.get_cpt(X)
        newCpt = cpt.loc[cpt[X] == False].reset_index(drop=True)
        new_p = [1.0 for i in range(len(newCpt))]
        newCpt['p'] = new_p
        newCpt.drop(columns=[X], inplace=True)
        print(newCpt, "\n\n")
        ## MOET ik de CPT ook updaten? Onduidelijk

    
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
        
        print(maxedout,"\n\n", cpt)
        ## Werkt, wat nu?
    
    def factorMultiplication(self, f, g):
        #JONAS
        #TODO: Factor multiplication: Given two factors f and g, compute the multiplied factor h=fg. (5pts)
        f_columns = (f.columns.drop('p'))
        g_columns = (g.columns.drop('p'))
        double = (f_columns).intersection(g_columns)[0]
        if not double:
            return False
        
        else:
            new = pd.merge(f, g, on=double)
            new['p'] = new['p_x'] * new['p_y']
            new.drop(columns=['p_x', 'p_y'], inplace=True)
            
        print(new)
        # Werkt wat moet ik ermee!?

        
    def _min_degree(self, X, int_graph):
        """Return the node with minimum degree in the graph"""
        
        int_sub_graph = [node for node in int_graph if node in X]
        return min(int_sub_graph, key=lambda x: x[1])
    
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
        
        for i in range(len(X)):
            node = order_func(X, int_graph)
            order.append(node)
            int_graph.remove_node(node)
            X.remove(node)
        return order
    
    def variableElimination(self, X, order_method = 'min_degree'):
        #SICCO
        #TODO: Variable Elimination: Sum out a set of variables by using variable elimination. (5pts)
        
        order = self.ordering(X, order_method)

        for node in order:
            children = self.bn.get_children(node)
            prob = self.bn.get_cpt(node)
            
            for child in children:
                cpt = self.bn.get_cpt(child)
                
                cpt.loc[cpt[node] == False,'p'] = cpt.loc[cpt[node] == False,'p'] * float(prob.loc[prob[node] == False,'p'])
                cpt.loc[cpt[node] == True,'p'] = cpt.loc[cpt[node] == True,'p'] * float(prob.loc[prob[node] == True,'p'])
                

                self.marginalization(cpt, node)

                self.bn.update_cpt(child, cpt)
                        
        pass
    
    def marginalDistribution(self):
        #SICCO
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
    # cptWet = BN.bn.get_cpt("Wet Grass?")
    # cptRain = BN.bn.get_cpt("Rain?")
    #BN.factorMultiplication(cptWet, cptRain)

    # BN.netPrune(['Wet Grass?'], {'Winter?':True, "Rain?":False})
    print(BN.variableElimination(['light-on', 'bowel-problem', 'hear-bark']))
    exit()
