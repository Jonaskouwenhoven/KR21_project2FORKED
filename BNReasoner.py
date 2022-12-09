from typing import Union
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
#import test_BNR
import numpy as np
import itertools
from itertools import combinations
import pgmpy

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
            

    # TODO: This is where your methods should go
    
    
    def netPrune(self, Q, evidence):
        #TODO: Network Pruning: Given a set of query variables Q and evidence e, node- and edge-prune the Bayesian network s.t. queries of the form P(Q|E) can still be correctly calculated
        evidence_nodes = list(evidence.keys()) if len(evidence) > 0 else []

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

        
        ### Edge Pruning
        for e in evidence:
            children = self.bn.get_children(e)
            for child in children:

                self.bn.del_edge((e, child))

        ### Node Pruning
        subgraph = self.bn.structure.subgraph(Q).copy()
        
        # Get dependent variables
        for node in Q_plus_e:
            dependent_nodes = list(nx.algorithms.dag.ancestors(self.bn.structure, node))
            dependent_nodes.append(node)
            dependent_nodes_subgraph = self.bn.structure.subgraph(dependent_nodes).copy()
            subgraph = nx.algorithms.operators.binary.compose(subgraph, dependent_nodes_subgraph)
        
        # Drop all irrelevant variables
        for node in self.bn.get_all_variables():
            if node not in list(subgraph.nodes()) and node not in evidence_nodes:
                self.bn.del_var(node)

        return self.bn

        
    def dSeparation(self, X, Y, Z):
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

    def maxingOut(self, X, f):
        # Deze functie werkt nog niet goed
        #TODO: Marginalization: Given a factor and a variable X, compute the CPT in which X is summed-out. (3pts)
        if X not in list(f.columns):
            return f
        
        else:
            new_columns = [c for c in (f.columns) if c not in [X, 'p']]
            max_index = f.loc[f.groupby(new_columns)["p"].idxmax(), X].reset_index(drop=True)
            maxed = f.groupby(new_columns)["p"].max().reset_index()
            new_f = pd.concat([max_index, maxed], axis=1)
            
            return new_f

    
    def factorMultiplication(self, f, g):
        #JONAS
        #TODO: Factor multiplication: Given two factors f and g, compute the multiplied factor h=fg. (5pts)
        
        f_columns = (f.columns.drop('p'))
        g_columns = (g.columns.drop('p'))
      
        double = list((f_columns).intersection(g_columns))
 
        if len(double) > 0:
            new = pd.merge(f, g, on=double)
            new['p'] = new['p_x'] * new['p_y']
            new.drop(columns=['p_x', 'p_y'], inplace=True)
        else:
            # merge two pd.DataFrames conditional probability tables without common columns            
            
            ### Get column length
            c_len_g = len(g.columns)-1
            c_len_f = len(f.columns)-1

            ### Get final shape
            table = pd.DataFrame(list(itertools.product([False, True], repeat=c_len_g+c_len_f)))
            
            ### Separate chape over variables
            g_choices = table.iloc[:, :c_len_g]
            f_choices = table.iloc[:, c_len_g:]

            g_choices.columns = g.columns[:-1]
            f_choices.columns = f.columns[:-1]

            ### Get the probability values of the variables
            g_ordered = pd.merge(g_choices, g, on=g_choices.columns.tolist(), how='left')
            f_ordered = pd.merge(f_choices, f, on=f_choices.columns.tolist(), how='left')

            ### Merge the two tables
            new = pd.concat([g_ordered.iloc[:,:-1], f_ordered.iloc[:,:-1]], axis=1)
            new['p'] = f_ordered['p']*g_ordered['p']

            ### Remove NaN values which correspond to the unavailable instances of the evidence nodes
            new = new.dropna().reset_index(drop = True)
        return new

    def factorsMultiplication(self, factors):
        """Extension of factor Multiplication"""

        factor_prod = factors[0]
        if len(factors) > 0:
            for factor in factors[1:]:
                factor_prod = self.factorMultiplication(factor_prod, factor)
    
        return factor_prod
    

    def _min_degree(self, X, int_graph):
        """Return the node with minimum degree in the graph"""
       
        int_sub_graph = [node for node in int_graph.degree if node[0] in X]
        return min(int_sub_graph, key=lambda x: x[1])[0] 

    def _fill(self, int_graph, node):
        """Return the fill of a node in the graph"""
        
        return len(list(combinations(nx.neighbors(int_graph, node), 2))) 

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
            node = order_func(X_copy, int_graph)
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

    def get_relevant_factors(self, node, factor_dict):
        """Return a list of factors that contain the node and a list that don't"""
        selected_factors = []
        selected_keys = [] # Only non relevant for this selection. Non relevant factors will be used later on
        
        # iterate over stored factors
        for key in factor_dict:
            factor = factor_dict[key]
            if node in factor.columns:
                selected_factors.append(factor)
                selected_keys.append(key)

        # remove the used selected_factors
        for key in selected_keys:
            factor_dict.pop(key)
        
        return selected_factors, factor_dict 

    def variableElimination(self, elimination_variables, order_method = 'min_degree'):
        #SICCO
        #TODO: Variable Elimination: Sum out a set of variables by using variable elimination. (5pts)
        # Deze functie klopt nog niet helemaal...

        ### 1. Get elimination order
        elimination_order = self.ordering(elimination_variables, order_method) if len(elimination_variables) != 0 else [] # get elimination order

        ### 3. Get all cpts of variables
        factor_dict = self.bn.get_all_cpts()
  
        ### 4. Eliminate variables
        for idx, node in enumerate(elimination_order):
            # get relevant factors
            working_factors, factor_dict = self.get_relevant_factors(node, factor_dict) # gets all the factors with node in the columns, and return factors without the used factors
            
            # multiply factors
            factor_prod = self.factorsMultiplication(working_factors)

            # margninalize
            marg_factor = self.marginalization(node, factor_prod)
      
            # update work_factors
            factor_dict[f'f{idx+1}'] = marg_factor

        # multiply all factors
        final_factors = []
        for key in factor_dict:
            final_factors.append(factor_dict[key])
        
        joint_marginal = self.factorsMultiplication(final_factors)

        return joint_marginal 

    def variableEliminationMaxedOut(self, elimination_variables, order_method = 'min_degree'):
        #SICCO
        #TODO: Variable Elimination: Sum out a set of variables by using variable elimination. (5pts)
        # Deze functie klopt nog niet helemaal...

        ### 1. Get elimination order
        elimination_order = self.ordering(elimination_variables, order_method) if len(elimination_variables) != 0 else [] # get elimination order

        ### 3. Get all cpts of variables
        factor_dict = self.bn.get_all_cpts()
  
        ### 4. Eliminate variables
        for idx, node in enumerate(elimination_order):
            # get relevant factors
            working_factors, factor_dict = self.get_relevant_factors(node, factor_dict) # gets all the factors with node in the columns, and return factors without the used factors
            
            # multiply factors
            factor_prod = self.factorsMultiplication(working_factors)

            # margninalize
            marg_factor = self.marginalization(node, factor_prod)
      
            # update work_factors
            factor_dict[f'f{idx+1}'] = marg_factor

        # multiply all factors
        final_factors = []
        for key in factor_dict:
            final_factors.append(factor_dict[key])
        
        joint_marginal = self.factorsMultiplication(final_factors)

        return joint_marginal 


    def marginalDistribution(self, Q, e = {}, order_method = 'min_degree'):
        #SICCO
        #TODO: Marginal Distributions: Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). Note that Q is a subset of the variables in the Bayesian network X with Q ⊂ X but can also be Q = X. (2.5pts)
        
        ### 1. Reduce factors with regard to the evidence
        if len(e) > 0 :
            # get all factors
            factors = self.bn.get_all_cpts()
    
            # reduce factors with regard to e
            for node in self.bn.get_all_variables():
                new_factor = self.bn.reduce_factor(pd.Series(e), factors[node])
                self.bn.update_cpt(node, new_factor)


        ### 2. Prune network
        self.netPrune(Q, e)
        variables = self.bn.get_all_variables()
        
        ### 3. Reduce evidence factor
        ev_fac = 1
        for ev in e:
            ev_fac *= self.bn.get_cpt(ev)['p'].sum()

        ### 4. Set variable lists            
        evidence_node =  list(e.keys()) 
        self.evidence_node = evidence_node
        Q_plus_e = Q + evidence_node 
        elimination_variables =  (
            set(variables)
            - set(Q_plus_e)
        )

        ### 5. Compute joint marginal
        joint = self.variableElimination(elimination_variables, order_method) 
        
        if len(e) == 0:
            return joint # if no evidence is provided, return the joint marginal = prior = posterior
        else:
            ### 6. Compute prior
            helper = joint
            for q in Q:
                helper = self.marginalization(q, pd.DataFrame(helper))
            pr_e = helper
            
            ### 7. Compute posterior
            posterior = joint
            posterior['p'] = joint['p'] / float(pr_e['p'])
            return posterior

        return

    def MAP(self, Q, e, order_method = 'min_degree'):
        #TODO: Compute the maximum a-posteriory instantiation + value of query variables Q, given a possibly empty evidence e. (3pts)
        
        ### 1. Set all variable lists
        evidence_node =  list(e.keys()) 
        Q_plus_e = Q + evidence_node

        ### 2. Get joint marginal
        joint = self.marginalDistribution(Q_plus_e, order_method= order_method)
        
        ### 3. Get MAP by maxing out all variables in Q
    
        helper = joint
        for q in Q:
            helper = self.maxingOut(q, helper)

        MAP = helper

        ### 4. Get MAP value
        for evidence in e:
            evidence_value = e[evidence]
            
            MAP = MAP[MAP[evidence] == evidence_value]

        return MAP 
        
    def MPE(self, Q, e = {}, order_method = 'min_degree'):
        #TODO: Compute the most probable explanation given an evidence e. (1.5pts)
         ### 1. Reduce factors with regard to the evidence
        if len(e) > 0 :
            # get all factors
            factors = self.bn.get_all_cpts()
    
            # reduce factors with regard to e
            for node in self.bn.get_all_variables():
                new_factor = self.bn.reduce_factor(pd.Series(e), factors[node])
                self.bn.update_cpt(node, new_factor)


        ### 2. Prune network
        self.netPrune(Q, e)
        variables = self.bn.get_all_variables()
        
        ### 3. Reduce evidence factor
        ev_fac = 1
        for ev in e:
            ev_fac *= self.bn.get_cpt(ev)['p'].sum()

        ### 4. Set variable lists            
        evidence_node =  list(e.keys()) 
        self.evidence_node = evidence_node
        Q_plus_e = Q + evidence_node 
        elimination_variables =  (
            set(variables)
            - set(Q_plus_e)
        )

        ### 5. Compute joint marginal
        joint = self.variableEliminationMaxedOut(elimination_variables, order_method) 
 
        ### 6. Compute MPE
        helper = joint
        for q in Q:
            helper = self.maxingOut(q, pd.DataFrame(helper))
        MPE = helper
     
        return MPE

        
        
if __name__ == '__main__':
    
    BN = BNReasoner('testing/dog_problem.BIFXML')
    # cptWet = BN.bn.get_cpt("Wet Grass?")
    # cptRain = BN.bn.get_cpt("Rain?")
    #BN.factorMultiplication(cptWet, cptRain)

    ### Test ordering, variable elimination and marginal distribution
    test_val1 = test_BNR.test_marginalDistribution1(BN)
    test_val2  = test_BNR.test_marginalDistribution2(BN) 
    test_val3 = test_BNR.test_marginalDistribution3(BN)
    
    if test_val1 and test_val2 and test_val3:
        print("Test marginal distribution passed")
    else:
        print("Test marginal distribution failed")
        exit()

    exit()
