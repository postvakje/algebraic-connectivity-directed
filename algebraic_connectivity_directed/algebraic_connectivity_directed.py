# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:56:28 2022

@author: Chai Wah Wu

functions to compute various notions of algebraic connectivity of directed graphs 
and other metrics useful for synchronization and control of networked dynamical systems.

"""

import math
import networkx as nx
import cvxpy as cp
import numpy as np
import scipy as sp
import scipy.sparse


def algebraic_connectivity_directed(G):
    """
    calculate algebraic connectivity a(G) and quantity b(G) of directed graph G,
    where G is a networkx digraph with Laplacian matrix L

    Reference: C. W. Wu, "Algebraic connecivity of directed graphs",
    Linear and Multilinear Algebra, vol. 53, no. 3, pp. 203-223, 2005.

    input: networkx graph G
    returns: a(G), b(G), M = Q'*(L+L')*Q/2
    """
    L = nx.laplacian_matrix.__wrapped__(G)
    n, m = L.shape
    r = (1, -1) + (0,) * (n - 2)
    c = (1,) + (0,) * (n - 2)
    C = sp.linalg.toeplitz(c, r)
    Q = sp.sparse.csr_matrix(scipy.linalg.orth(C.T))
    M = 0.5 * Q.T @ (L + L.T) @ Q
    if n <= 3:
        a = np.min(np.real(sp.linalg.eig(M.toarray(), right=False)))
        b = np.max(np.real(sp.linalg.eig(M.toarray(), right=False)))
    else:
        a = np.real(
            sp.sparse.linalg.eigs(M, k=1, which="SR", return_eigenvectors=False)[0]
        )
        b = np.real(
            sp.sparse.linalg.eigs(M, k=1, which="LR", return_eigenvectors=False)[0]
        )
    return a, b, M


def compute_left_ev(A):
    """compute left eigenvalue v such that v^TA = lambda where lambda is eigenvalue smallest in magnitude
    such that max_i v_i = 1
    """
    n, m = A.shape
    if n <= 3:
        v, w = sp.linalg.eig(A.astype(np.float64).toarray(), left=True, right=False)
        index = np.argmin(np.abs(v))
        w = np.real(w[:, index])
    else:
        _, w = sp.sparse.linalg.eigs(A.T.astype(np.float64), k=1, which="SM")
        w = np.real(w.flatten())
    w[np.isclose(w, 0)] = 0
    wmax = np.max(w)
    if wmax <= 0:
        w = w / np.min(w)
        w[np.isclose(w, 0)] = 0
    else:
        w = w / wmax
    return w


def algebraic_connectivity_directed_variants(G, k=1):
    """
    calculate other generalizations of algebraic connectivity a(G) of directed graph G,
    where G is a networkx digraph with Laplacian matrix L

    References:
    1. C. W. Wu, "Algebraic connecivity of directed graphs",
    Linear and Multilinear Algebra, vol. 53, no. 3, pp. 203-223, 2005.
    2. C. W. Wu, "On Rayleigh-Ritz ratios of a generalized Laplacian matrix of directed graphs", Linear Algebra
    and its applications, vol. 402, pp. 207-227, 2005.
    3. C. W. Wu, "Synchronization in networks of nonlinear dynamical systems coupled via a directed graph",
    Nonlinearity, vol. 18, pp. 1057-1064, 2005.

    k = 1, 2, 3, 4 return a_1, a_2, a_3, a_4 as described in
    C. W. Wu, "Synchronization in Complex Networks of Nonlinear Dynamical Systems", World Scientific, 2007.

    a_1 is the same as the value returned by algebraic_connectivity_directed(G)[0] (see Ref. [1]).
    a_2 is the same as tilde{a} as described in Ref. [2].
    a_3 is described in the proof of Theorem 21 in Ref. [2].
    a_4 is equal to eta as described in Ref. [3].

    input: networkx graph G
    returns: a_k(G)
    """
    if k == 1:
        return algebraic_connectivity_directed(G)[0]
    L = nx.laplacian_matrix.__wrapped__(G)
    n, m = L.shape
    if k >= 2:
        if not nx.is_strongly_connected(G):
            w = np.zeros(n)
            Gdict = {a: b for b, a in enumerate(G.nodes())}
            for scc in nx.strongly_connected_components(G):
                scct = list(scc)
                nt = len(scct)
                scindex = [Gdict[i] for i in scct]
                Li = L[scindex, :][:, scindex]
                Li = Li - sp.sparse.spdiags(
                    Li.sum(axis=1).flatten(), 0, nt, nt, format="csr"
                )
                wi = compute_left_ev(Li)
                w[scindex] = wi
        else:
            w = compute_left_ev(L)
        W = sp.sparse.diags(w, 0)
        r = (1, -1) + (0,) * (n - 2)
        c = (1,) + (0,) * (n - 2)
        C = sp.linalg.toeplitz(c, r)
        Q = sp.sparse.csr_matrix(scipy.linalg.orth(C.T))
        M = 0.5 * Q.T @ (W @ L + L.T @ W) @ Q
    if k == 2:
        if n <= 3:
            a = np.min(np.real(sp.linalg.eig(M.toarray(), right=False)))
        else:
            a = np.real(
                sp.sparse.linalg.eigs(M, k=1, which="SR", return_eigenvectors=False)[0]
            )
    elif k == 3:
        if not nx.is_strongly_connected(G):
            raise ValueError("Graph is not strongly connected.")
        U = W - np.outer(w, w) / np.sum(w)
        U = 0.5 * Q.T @ (U + U.T) @ Q
        if n <= 3:
            a = np.min(np.real(sp.linalg.eig(M.toarray(), b=U, right=False)))
        else:
            a = np.real(
                sp.sparse.linalg.eigs(
                    M, k=1, M=U, which="SR", return_eigenvectors=False
                )[0]
            )
    elif k == 4:
        w = np.zeros(n)
        Gdict = {a: b for b, a in enumerate(G.nodes())}
        eta = np.Inf
        flag = False
        for scc in nx.strongly_connected_components(G):
            scct = list(scc)
            nt = len(scct)
            r = (1, -1) + (0,) * (nt - 2)
            c = (1,) + (0,) * (nt - 2)
            C = sp.linalg.toeplitz(c, r)
            Q = sp.sparse.csr_matrix(scipy.linalg.orth(C.T))
            I = sp.sparse.eye(nt)
            scindex = [Gdict[i] for i in scct]
            Bi = L[scindex, :][:, scindex]
            Lsum = np.asarray(Bi.sum(axis=1)).flatten()
            alpha = 1 if math.isclose(0, max(np.abs(Lsum))) else 0
            if alpha == 1:
                if flag:  # reversal of graph does not contain a spanning directed tree
                    eta = 0
                    break
                else:
                    flag = True
            if nt == 1:
                if alpha == 0:
                    a = np.asarray(Bi).flatten()[0][0, 0]
                else:
                    a = np.Inf
            else:
                Li = Bi - sp.sparse.spdiags(Lsum, 0, nt, nt, format="csr")
                wi = compute_left_ev(Li)
                Wi = sp.sparse.diags(wi, 0)
                if alpha == 1:
                    Ui = Wi - np.outer(wi, wi) / np.sum(wi)
                    Ui = 0.5 * Q.T @ (Ui + Ui.T) @ Q
                    Mi = 0.5 * Q.T @ (Wi @ Bi + Bi.T @ Wi) @ Q
                else:
                    Ui = Wi
                    Ui = 0.5 * I @ (Ui + Ui.T) @ I
                    Mi = 0.5 * I @ (Wi @ Bi + Bi.T @ Wi) @ I
                if nt <= 3:
                    if sp.sparse.isspmatrix(Mi):
                        Mi = Mi.todense()
                    if sp.sparse.isspmatrix(Ui):
                        Ui = Ui.todense()
                    a = np.min(np.real(sp.linalg.eig(Mi, b=Ui, right=False)))
                else:
                    a = np.real(
                        sp.sparse.linalg.eigs(
                            Mi, k=1, M=Ui, which="SR", return_eigenvectors=False
                        )[0]
                    )
            eta = min(a, eta)
        a = eta
    else:
        raise NotImplementedError()
    return a


def compute_mu_directed(*Graphs):
    """
    returns mu(G) defined as the supremum of numbers mu such that
    U(L-mu*I)+(L'-mu*I)U is positive semidefinite for some symmetric zero row sums
    real matrix U with nonpositive off-diagonal elements where L is the Laplacian matrix
    of graph G.

    References:
    1. C. W. Wu, "Synchronization in coupled arrays of chaotic oscillators
    with nonreciprocal coupling.", IEEE Transactions on Circuits and Systems–I, vol. 50,
    no. 2, pp. 294–297, 2003.
    2. C. W. Wu, "Synchronization in Complex Networks of Nonlinear Dynamical Systems", World Scientific, 2007.
    3. C. W. Wu, "Synchronization in dynamical systems coupled via multiple directed networks,"
    IEEE Transactions on Circuits and Systems-II: Express Briefs, vol. 68, no. 5, pp. 1660-1664, 2021.

    input: networkx graph G
    returns: mu(G)

    Function accepts multiple arguments. If the input are multiple graphs G1, G2, G3, ... with Li the Laplacian matrix of Gi,
    and all Gi have the same number of nodes,
    then compute_mu_directed(G1, G2, G3, ...) returns the supremum of mu such that there
    exist some symmetric zero row sums real matrix U with nonpositive off-diagonal elements
    where for all i, U(Li-mu*I)+(Li'-mu*I)U is positive semidefinite. This is useful in analyzing
    synchronization of networked systems where systems are coupled via multiple networks. See Ref. [3].

    """
    G = Graphs[0]
    L = nx.laplacian_matrix.__wrapped__(G)
    n, m = L.shape
    r = (1, -1) + (0,) * (n - 2)
    c = (1,) + (0,) * (n - 2)
    C = sp.linalg.toeplitz(c, r)
    Q = sp.sparse.csr_matrix(scipy.linalg.orth(C.T))
    U = cp.Variable((n, n), symmetric=True)
    I = sp.sparse.eye(n)
    I2 = sp.sparse.eye(n - 1)
    e = np.ones((n, 1))
    if n <= 3:
        ub = sorted(
            np.real(sp.linalg.eig(L.astype(np.float64).toarray(), right=False))
        )[1]
    else:
        ub = sorted(
            np.real(
                sp.sparse.linalg.eigs(
                    L.astype(np.float64), k=2, which="SR", return_eigenvectors=False
                )
            )
        )[1]
    lb = algebraic_connectivity_directed(G)[0]
    constraints = [U @ e == 0]
    constraints += [U - cp.diag(cp.diag(U)) <= 0]
    constraints += [Q.T @ U @ Q >> I2]
    Llist = []
    for Gi in Graphs[1:]:
        Li = nx.laplacian_matrix.__wrapped__(Gi)
        Llist.append(Li)
        ni, mi = Li.shape
        if ni != n or mi != m:
            raise ValueError("Graphs are not of the same order.")
        if n <= 3:
            ub = min(
                ub,
                sorted(
                    np.real(sp.linalg.eig(Li.astype(np.float64).toarray(), right=False))
                )[1],
            )
        else:
            ub = min(
                ub,
                sorted(
                    np.real(
                        sp.sparse.linalg.eigs(
                            Li.astype(np.float64),
                            k=2,
                            which="SR",
                            return_eigenvectors=False,
                        )
                    )
                )[1],
            )
        lb = min(lb, algebraic_connectivity_directed(Gi)[0])
    mu = ub
    while np.abs(ub - lb) > 1e-9:
        mu = 0.5 * (ub + lb)
        newconstraints = constraints.copy()
        newconstraints += [U @ (L - mu * I) >> 0]
        for Li in Llist:
            newconstraints += [U @ (Li - mu * I) >> 0]
        prob = cp.Problem(cp.Minimize(0), constraints=newconstraints)
        prob.solve()
        if prob.status not in ["infeasible", "unbounded"]:
            lb = mu
        else:
            ub = mu
    return mu


def run_tests():
    # directed cycle graphs
    for n in range(2, 16):
        G = nx.cycle_graph(n, create_using=nx.DiGraph)
        a, b, M = algebraic_connectivity_directed(G)
        assert math.isclose(
            2 * np.sin(np.pi / n) ** 2, algebraic_connectivity_directed_variants(G, 2)
        )
        assert math.isclose(
            2 * np.sin(np.pi / n) ** 2, algebraic_connectivity_directed_variants(G, 3)
        )
        assert math.isclose(
            2 * np.sin(np.pi / n) ** 2, algebraic_connectivity_directed_variants(G, 4)
        )
        assert math.isclose(2 * np.sin(np.pi / n) ** 2, a)
        assert math.isclose(2 * np.sin(np.pi * (n - 1) / 2 / n) ** 2 if n % 2 else 2, b)

    # directed path graphs
    for n in range(2, 16):
        G = nx.path_graph(n, create_using=nx.DiGraph)
        assert math.isclose(1, algebraic_connectivity_directed_variants(G, 4))

    # exploding star graphs
    for n in range(3, 16):
        G = nx.DiGraph()
        for n in range(2, n + 1):
            G.add_edge(1, n)
        a, b, M = algebraic_connectivity_directed(G)
        assert math.isclose(0, a, abs_tol=1e-9)
        assert math.isclose(n - 1, b)

    # imploding star graphs
    for n in range(2, 16):
        G = nx.DiGraph()
        for n in range(2, n + 1):
            G.add_edge(n, 1)
        a, b, M = algebraic_connectivity_directed(G)
        assert math.isclose(1, a)
        assert math.isclose(1, b)

    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 2)
    assert math.isclose(1, compute_mu_directed(G))
    assert math.isclose(0.3819660107037309, compute_mu_directed(G.reverse()))
    assert math.isclose(
        0.381966011250105, algebraic_connectivity_directed_variants(G.reverse(), 4)
    )
    assert math.isclose(2.0773502691896, algebraic_connectivity_directed(G)[1])
    assert math.isclose(
        2.6547005383793, algebraic_connectivity_directed(G.reverse())[1]
    )

    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 1)
    G.add_edge(1, 3)
    G.add_edge(3, 1)
    G.add_edge(2, 3)
    G.add_edge(2, 4)
    G.add_edge(4, 5)
    G.add_edge(4, 6)
    G.add_edge(5, 6)
    G.add_edge(6, 5)
    assert math.isclose(0.1206147578199189, compute_mu_directed(G))
    assert math.isclose(-0.0426650385846, algebraic_connectivity_directed(G)[0])
    assert math.isclose(
        0.119049451961957, algebraic_connectivity_directed_variants(G, 4)
    )
    assert math.isclose(1, algebraic_connectivity_directed_variants(G.reverse(), 4))

    A1 = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ]
    )

    A2 = np.array(
        [
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0],
        ]
    )

    A3 = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ]
    )
    G1 = nx.from_numpy_matrix(A1, create_using=nx.DiGraph)
    G2 = nx.from_numpy_matrix(A2, create_using=nx.DiGraph)
    G3 = nx.from_numpy_matrix(A3, create_using=nx.DiGraph)
    assert math.isclose(1.225195885988909, compute_mu_directed(G2))
    assert math.isclose(
        0.660608870771608, algebraic_connectivity_directed_variants(G1, 4)
    )
    assert math.isclose(0.83812, compute_mu_directed(G1, G2, G3), rel_tol=1e-4)

    # disconnected graph
    G = nx.from_numpy_matrix(
        np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), create_using=nx.DiGraph
    )
    assert math.isclose(
        -0.07735026918962568, algebraic_connectivity_directed_variants(G, 1)
    )
    assert math.isclose(0, compute_mu_directed(G), abs_tol=1e-9)
    assert math.isclose(0, algebraic_connectivity_directed_variants(G, 4), abs_tol=1e-9)


def main():
    run_tests()


if __name__ == "__main__":
    main()
