# algebraic-connectivity-directed
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python functions to compute various notions of algebraic connectivity of directed graphs

## Requirements
Requires `python` >= 3.5 and packages `numpy`, `scipy`, `networkx` and `cvxpy`.

## Installation
`pip install algebraic-connectivity-directed`

## Usage
After installation, run `from algebraic_connectivity_directed.algebraic_connectivity_directed import *`

There are 3 main functions:

1. Function `algebraic_connectivity_directed`: algebraic_connectivity_directed(G) returns `a, b, M` where `a` is the algebraic connectivity of the digraph G. The graph G is a `networkx` DiGraph object. The definitions of `a, b, M = Q'*(L+L')*Q/2` can be found in Ref. [2].

2. Function `algebraic_connectivity_directed_variants`: algebraic_connectivity_directed_variants(G,k) returns variations of algebraic connectivity of the digraph G.
The graph `G` is a `networkx` DiGraph object. Setting `k = 1, 2, 3, 4` returns a<sub>1</sub>, a<sub>2</sub>, a<sub>3</sub>, a<sub>4</sub> as defined in Ref. [5]. 

3. Function `compute_mu_directed`:
compute_mu_directed(G)
returns `mu(G)` defined as the supremum of numbers &mu; such that 
U(L-&mu;\*I)+(L'-&mu;\*I)U is positive semidefinite for some symmetric zero row sums
real matrix `U` with nonpositive off-diagonal elements where `L` is the Laplacian matrix
of graph `G` (see Ref. [1]).

`compute_mu_directed` accepts multiple arguments. If the input are multiple graphs G<sub>1</sub>, G<sub>2</sub>, G<sub>3</sub>, ... with L<sub>i</sub> the Laplacian matrix of G<sub>i</sub>, 
and all G<sub>i</sub> have the same number of nodes,
then compute_mu_directed(G<sub>1</sub>, G<sub>2</sub>, G<sub>3</sub>, ...) returns the supremum of &mu; such that there 
exist some symmetric zero row sums real matrix U with nonpositive off-diagonal elements 
where for all i, U(L<sub>i</sub>-&mu;\*I)+(L<sub>i</sub> '-&mu;\*I)U is positive semidefinite. This is useful in analyzing
synchronization of networked systems where systems are coupled via multiple networks. See Ref. [6].
The graph G is a `networkx` DiGraph object.

a<sub>1</sub> is the same as the value returned by algebraic_connectivity_directed(G)[0] (see Ref. [2]).
    
a<sub>2</sub> is the same as $\tilde{a}$ as described in Ref. [3].

a<sub>3</sub> is described in the proof of Theorem 21 in Ref. [3].

a<sub>4</sub> is equal to &eta; as described in Ref. [4].

If the reversal of the graph does not contain a spanning directed tree, then a<sub>2</sub> &le; 0.

If G is strongly connected then a<sub>3</sub> &ge; a<sub>2</sub> > 0.

a<sub>4</sub> > 0 if and only if the reversal of the graph contains a spanning directed tree.

## Examples

Cycle graph
``` 
from algebraic_connectivity_directed.algebraic_connectivity_directed import *
import networkx as nx
import numpy as np
G = nx.cycle_graph(10,create_using=nx.DiGraph)
print(algebraic_connectivity_directed(G)[0:2])

>> (0.19098300562505233, 2.0)
print(algebraic_connectivity_directed_variants(G,2))
>> 0.1909830056250514
```  

Directed graphs of 5 nodes

```
A1 = np.array([[0,0,1,0,0],[0,0,0,1,1],[1,0,0,1,1],[1,1,0,0,1],[0,0,0,1,0]])     
G1 = nx.from_numpy_matrix(A1,create_using=nx.DiGraph)
print(compute_mu_directed(G1))
>>> 0.8521009635833089
print(algebraic_connectivity_directed_variants(G1, 4))
>>> 0.6606088707716056
A2 = np.array([[0,1,0,0,1],[0,0,0,1,0],[0,0,0,1,1],[1,0,0,0,0],[1,0,1,1,0]])  
G2 = nx.from_numpy_matrix(A2,create_using=nx.DiGraph)
A3 = np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,0,0],[0,0,1,0,0],[1,1,1,0,0]]) 
G3 = nx.from_numpy_matrix(A3,create_using=nx.DiGraph)
print(compute_mu_directed(G1,G2,G3))
>>> 0.8381214637786955
```
## References
1. C. W. Wu, "Synchronization in coupled arrays of chaotic oscillators 
with nonreciprocal coupling.", IEEE Transactions on Circuits and Systems–I, vol. 50,
no. 2, pp. 294–297, 2003.

2. C. W. Wu, "Algebraic connecivity of directed graphs", 
    Linear and Multilinear Algebra, vol. 53, no. 3, pp. 203-223, 2005.

3. C. W. Wu, "On Rayleigh-Ritz ratios of a generalized Laplacian matrix of directed graphs", Linear Algebra
    and its applications, vol. 402, pp. 207-227, 2005.
    
4. C. W. Wu, "Synchronization in networks of nonlinear dynamical systems coupled via a directed graph", 
    Nonlinearity, vol. 18, pp. 1057-1064, 2005.

5. C. W. Wu, "Synchronization in Complex Networks of Nonlinear Dynamical Systems", World Scientific, 2007.

6. C. W. Wu, "Synchronization in dynamical systems coupled via multiple directed networks," 
IEEE Transactions on Circuits and Systems-II: Express Briefs, vol. 68, no. 5, pp. 1660-1664, 2021.
