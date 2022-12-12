from typing import Union
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
from pgmpy.readwrite import XMLBIFReader

import matplotlib.pyplot as plt
import test_BNR
import numpy as np
import pgmpy
from VariableEliminate import VariableElimination


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
        # print(evidence, "THISSSS")
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

                if nx.is_simple_path(Graph, [y, x]) == False and nx.is_simple_path(Graph, [x, y]) == False:

                    return True
                for path in nx.all_simple_paths(Graph, source=x, target=y):

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
    
    def independence(self, X, Y, Z):
        #CAS
        #TODO: Independence: Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z. (Hint: Remember the connection between d-separation and independence) (1.5pt)
        if self.dSeperation(X, Y, Z):
            return True
        
        return False
    
    def marginalization(self, X, f):
        #JONAS
        #TODO: Marginalization: Given a factor and a variable X, compute the CPT in which X is summed-out. (3pts)
        if X not in list(f.columns):
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

    def variableElimination(self, Q, elimination_variables, order_method = 'min_degree'):
        var = self.bn.get_all_variables()


        elimination_variables = self.ordering(elimination_variables, order_method)
        cpts = self.bn.get_all_cpts()
        fac = 0

        for var in elimination_variables:
            temp = {}
            for cpt in cpts:
                if var in cpts[cpt]:
                    temp[cpt] = cpts[cpt]
                    
            if len(temp) > 1:

                mat_cpt = self.factorMultiplication(temp[list(temp.keys())[0]], temp[list(temp.keys())[1]])
                ding = (mat_cpt)

                updated = self.marginalization(var, ding)
                for factor in temp:
                    cpts.pop(factor)
                
                fac += 1
                cpts["factor" + str(fac)] = updated
                
            elif len(temp) == 1:
                ding = (temp[list(temp.keys())[0]])
                updated = self.marginalization(var, ding)
                for factor in temp:
                    cpts.pop(factor)
                    
                fac += 1
                cpts['factor' + str(fac)] = updated

        return cpts

    
    def marginalDistribution(self, Q, e = {}, order_method = 'min_degree'):        
        
        self.netPrune(Q, e)
        evidence_node =  list(e.keys()) 
        Q_plus_e = Q + evidence_node 
        elimination_variables =  (
            set(self.bn.get_all_variables())
            - set(Q_plus_e)
        )

        eliminationOrder = self.ordering(elimination_variables, order_method)
        totalCpts= []
        for elim in eliminationOrder:
            for node in self.bn.get_all_variables():
                if elim  == node or node not in Q_plus_e:
                    continue
                
                elimCpt = self.bn.get_cpt(elim)
                nodeCpt = self.bn.get_cpt(node)
                
                if elim in nodeCpt.columns.to_list()[:-1]:
                    finalCpt = self.factorMultiplication(elimCpt, nodeCpt)
                    resultCpt = self.marginalization(elim, finalCpt)
                    totalCpts.append(resultCpt)
                    self.bn.update_cpt(node, resultCpt)
                    
        for cpt in self.bn.get_all_cpts():
            print(self.bn.get_all_cpts()[cpt])  #### HIER KOMEN VM DE GOEIE WAARDES UIT!!!!!
            
        return 

        
        
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
    reader = XMLBIFReader("testing/lecture_example.BIFXML")
    model = reader.get_model()
    BN = BNReasoner('testing/Russia.BIFXML')
    BN.bn.draw_structure()
    exit()
    margDist2 = VariableElimination(model).query(['Winter?', 'Rain?', "Wet Grass?"], evidence={'Slippery Road?': 'True'})
    print(margDist2)
    margDist = BN.marginalDistribution(['Winter?', 'Rain?', "Wet Grass?"], e={'Slippery Road?': 'False'})
    # print(margDist)
    exit()
