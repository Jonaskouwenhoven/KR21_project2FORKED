from typing import Union
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy 


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
                self.bn.update_cpt(key, cit)
            self.bn.del_var(key)
            
            

        # self.bn.draw_structure()
                # print(crt)
        pass

        
    def dSeperation(self, X, Y, Z):
        Graph = self.bn.structure

        
        if X == Y:
            return False
        
        for x in X:
            for y in Y:
                for path in nx.all_simple_paths(Graph, source=y, target=x):
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
    
    def marginalization(self, X, cpt):
        #JONAS
        #TODO: Marginalization: Given a factor and a variable X, compute the CPT in which X is summed-out. (3pts)
        #NOTE: Hier moet nog wat aan gedaan worden. Er staat given a "factor" and a varaible X, maar nu gebruik je die factor niet. je moet ook de andere cpt aanpassen
        #cpt = self.bn.get_cpt(X)
       
        if X not in cpt.columns:
            return cpt
        else:
            new_columns = [c for c in (cpt.columns) if c not in [X, 'p']]
            cpt = cpt.groupby(new_columns)["p"].sum().reset_index()
            return cpt

    
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
                

                marg_factor = self.marginalization( node, cpt)

                self.bn.update_cpt(child, marg_factor)

        return
    
    def marginalDistribution(self, Q, e = None, order_method = 'min_degree'):
        #SICCO
        #TODO: Marginal Distributions: Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). Note that Q is a subset of the variables in the Bayesian network X with Q ⊂ X but can also be Q = X. (2.5pts)
        
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

        # posterior
        joint_marginal = self.bn.get_cpt(Q[0])
        prior = self.bn.get_cpt(evidence_node[0])

        posterior = joint_marginal.copy()
        posterior['p'] = joint_marginal['p'] / float(prior.loc[prior[evidence_node[0]]==e[evidence_node[0]], 'p'])

        return posterior
    
    
    
    def MAP(self, Q, e):
        #TODO: Compute the maximum a-posteriory instantiation + value of query variables Q, given a possibly empty evidence e. (3pts)
        # Each dataframe has columns for each variable and a 'p' column for the probabilities
        self.netPrune(Q, e)
        variables = self.bn.get_all_variables()
        cpts = [self.bn.get_cpt(var) for var in variables]
        # Define the initial probability for each query variable as 1
        probs = {var: 1 for var in Q}

        # Iterate through the CPTS and update the probabilities for each query variable
        # based on the evidence and the probabilities in the CPT
        for cpt in cpts:
            # Filter the CPT based on the evidence

            if not set(e.keys()).intersection(set(cpt.columns)):
                continue

            # Filter the CPT based on the evidence
            isin = cpt.columns.isin(list(e.keys()))
            cpt = cpt.loc[:, isin]

            # Update the probabilities for each query variable
            for var in Q:
                if var not in cpt.columns:
                    continue
                probs[var] = probs[var] * cpt[cpt[var] == True]['p'].prod()

        # The maximum a-posteriori query is the variable with the highest probability
        instantiation = max(probs, key=probs.get)

        # The probability of the MAP query is the highest probability
        value = probs[instantiation]
        # Return the results
        return (instantiation, value)
        
    def MEP(self, Q, e):
        #TODO: Compute the most probable explanation given an evidence e. (1.5pts)
        # Each dataframe has columns for each variable and a 'p' column for the probabilities
        # self.netPrune(Q, e)
        variables =self.bn.get_all_variables()
        cpts = [self.bn.get_cpt(var) for var in variables]
        
        # Define the initial probability for each possible combination of query variables as 1
        probs = {var: 1 for var in Q}

        # Iterate through the CPTS and update the probabilities for each possible combination of query variables
        # based on the evidence and the probabilities in the CPT
        for cpt in cpts:
            # Check if any of the evidence variables are in the CPT
            if not set(e.keys()).intersection(set(cpt.columns)):
                continue

            # Filter the CPT based on the evidence
            isin = cpt.columns.isin(list(e.keys()))
            cpt = cpt.loc[:, isin]

            # Update the probabilities for each possible combination of query variables
            for var in Q:
                # Get the name of the variable in the bayesian network
                # that corresponds to the query variable
                if var not in cpt.columns:
                    continue

                # Update the probabilities using the name of the variable in the bayesian network
                probs[var] = probs[var] * cpt[cpt[var] == True]['p'].prod()

        # The most probable explanation is the combination of query variables with the highest probability
        explanation = max(probs, key=probs.get)

        # The probability of the MEP is the highest probability
        value = probs[explanation]
        # Return the results
        return (explanation, value)

        



if __name__ == '__main__':
    
    BN = BNReasoner("/Users/jonas/Documents/GitHub/KR21_project2FORKED/KR21_forked/testing/asia.BIFXML")


    assert BN.dSeperation(['either'],['asia'],['dysp','bronc','smoke']) == False
    assert BN.dSeperation(['xray'],['smoke'],['lung']) == True

    exit()
