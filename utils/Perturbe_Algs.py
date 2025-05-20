import torch
import numpy as np
import networkx as nx
from scipy.optimize import root_scalar



class Attacks:
  def __init__(self, A, K, m, alpha = 1, max_iter = 100, filter = 'adj'):
        '''
        A: adjacency matrix
        K: autocorrelation matrix
        m: number of edges to be perturbed
        alpha: step size
        max_iter: maximum number of iterations
        filter: type of filter to be used: 
            'adj': adjacency matrix
            'lap': Laplacian matrix
            'adj_norm': normalized adjacency matrix
            if other coustimized filter is used, please provide the function
        '''
        self.A = A
        self.K = K
        self.n = A.shape[0]
        self.J = (torch.ones((self.n, self.n)) - torch.eye(self.n))
        self.m = m
        self.alpha = alpha
        self.max_iter = max_iter
        self.S = torch.ones(self.n, self.n)
        self.mute = True
        self.filter = filter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def to(self, device):
    self.A = self.A.to(device)
    self.K = self.K.to(device)
    self.J = self.J.to(device)
    self.S = self.S.to(device)
    return self

  def perturb(self, S):
    """
    Apply perturbation to the adjacency matrix A based on the selected edges S.
    The perturbation is done by flipping the edges in S.
    Parameters:
        S (torch.Tensor): A 2D tensor of shape (m, 2) where m is the number of edges to be perturbed.
    Return: 
        The perturbed adjacency matrix.
    """

    # m = S.shape[0]
    M = self.A.clone()
    for i in range(self.m):
        M[S[i, 0], S[i, 1]] = 1 - M[S[i, 0], S[i, 1]]
        M[S[i, 1], S[i, 0]] = 1 - M[S[i, 1], S[i, 0]]
    return M

  def bisection_method(self, func, a, b, precision):
    """
    Bisection method for finding a root of a continuous function.
    Parameters:
        func (callable): The function for which to find the root.
        a (float): The start of the interval.
        b (float): The end of the interval.
        precision (float): The desired precision for the root.
    Returns:
        float: The approximated root.
    """
    if func(a) * func(b) > 0:
        raise ValueError("The function must have opposite signs at a and b (root not guaranteed).")

    midpoint = (a + b) / 2.0
    while abs(func(b) - func(a)) > precision:
        if func(midpoint) == 0:
            return midpoint  # Exact root found
        elif func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        midpoint = (a + b) / 2.0

    return midpoint
  

  def obj_avg(self, S):
    '''
    Compute the objective function for the average perturbation.
    Parameters:
        S (torch.Tensor): The perturbation matrix.
    Returns:
        The average emebdding perturbation: trace( K @ (g(A) - g(Ap)) @ (g(A) - g(Ap).T ),
        where g(A) is the graph filter;
        g(Ap) is the perturbed graph filter;
        K is the autocorrelation matrix.
    '''
    # A - Ap = (J-2A)*S
    diff = (self.J - 2 * self.A) * S
    # Compute the objective: trace( K @ (gA - gAp) @ (gA - gAp).T )
    if self.filter == 'adj':
      avg_pertb = torch.trace(self.K @ diff @ diff.T)
    if self.filter == 'lap':
      Ap = self.A + diff
      dA = torch.sum(self.A, dim=1)
      dAp = torch.sum(Ap, dim=1)
      gA = torch.diag(dA) - self.A
      gAp = torch.diag(dAp) - Ap

      avg_pertb = torch.trace(self.K @ (gA - gAp) @ (gA - gAp).T)
    if self.filter == 'adj_norm':
      eps=1e-6
      Ap = self.A + diff
      dA = torch.sum(self.A, dim=1)
      dAp = torch.sum(Ap, dim=1)
      # Compute gA = D_A^(-1/2) * A * D_A^(-1/2), where D_A = diag(sum(A, axis=1))
      inv_sqrt_dA = torch.diag(torch.pow(dA + eps, -0.5))
      inv_sqrt_dA = torch.nan_to_num(inv_sqrt_dA, posinf=0, neginf=0) 
      gA = inv_sqrt_dA @ self.A @ inv_sqrt_dA
      # Compute gAp = D_Ap^(-1/2) * Ap * D_Ap^(-1/2), where D_Ap = diag(sum(Ap, axis=1))
      inv_sqrt_dAp = torch.diag(torch.pow(dAp + eps, -0.5))
      inv_sqrt_dAp = torch.nan_to_num(inv_sqrt_dAp, posinf=0, neginf=0)
      gAp = inv_sqrt_dAp @ Ap @ inv_sqrt_dAp

      avg_pertb = torch.trace(self.K @ (gA - gAp) @ (gA - gAp).T)
       
    return avg_pertb

  def obj_wst(self, S):
    '''
    Compute the objective function for the worst-case perturbation.
    Parameters:
        S (torch.Tensor): The perturbation matrix.
    Returns:
        The worst-case emebdding perturbation: |g(A) - g(Ap)|_{sp},
        where g(A) is the graph filter;
        g(Ap) is the perturbed graph filter.
    '''
    # A - Ap = (J-2A)*S
    diff = (self.J - 2 * self.A) * S
    # Compute the objective: |gA - gAp|_{sp}
    if self.filter == 'adj':
      wst_pertb = torch.linalg.norm(diff, ord=2)
    if self.filter == 'lap':
      Ap = self.A + diff
      dA = torch.sum(self.A, dim=1)
      dAp = torch.sum(Ap, dim=1)
      gA = torch.diag(dA) - self.A
      gAp = torch.diag(dAp) - Ap
      wst_pertb = torch.linalg.norm(gA-gAp, ord=2)
    if self.filter == 'adj_norm':
      Ap = self.A + diff
      dA = torch.sum(self.A, dim=1)
      dAp = torch.sum(Ap, dim=1)
      # Compute gA = D_A^(-1/2) * A * D_A^(-1/2), where D_A = diag(sum(A, axis=1))
      inv_sqrt_dA = torch.diag(torch.pow(dA, -0.5))
      inv_sqrt_dA = torch.nan_to_num(inv_sqrt_dA, posinf=0, neginf=0) 
      gA = inv_sqrt_dA @ self.A @ inv_sqrt_dA
      # Compute gAp = D_Ap^(-1/2) * Ap * D_Ap^(-1/2), where D_Ap = diag(sum(Ap, axis=1))
      inv_sqrt_dAp = torch.diag(torch.pow(dAp, -0.5))
      inv_sqrt_dAp = torch.nan_to_num(inv_sqrt_dAp, posinf=0, neginf=0)
      gAp = inv_sqrt_dAp @ Ap @ inv_sqrt_dAp
      wst_pertb = torch.linalg.norm(gA-gAp, ord=2)

    return wst_pertb


  def pgd_avg(self):
    """
    Projected Gradient Descent (PGD) for average perturbation.
    Each iteration updates the perturbation indicator matrix S using gradient descent,
    followed by a projection step to its feasible region (a convex relaxation of indicator matrix).
    The process continues until the maximum number of iterations is reached or
    the number of edges in S is less than or equal to 2 * (m + 2).

    Returns:
        torch.Tensor: The perturbed adjacency matrix.
    """
    # Initialize S as a symmetric matrix and enable gradient tracking
    S = torch.ones(self.n, self.n, device = self.device)
    a = self.alpha

    for i in range(self.max_iter):
      S_iter = S.clone().detach().requires_grad_(True)
      loss = self.obj_avg(S_iter)
      # Step 1. Compute the gradient of loss with respect to S
      loss.backward()

      # Step 2: Update by gradient descent
      S_iter = S_iter + (a/torch.linalg.norm(S_iter.grad, ord=2)) * S_iter.grad
      S_iter = (S_iter + S_iter.t()) / 2
      # print(f'Iteration {i:d} gradent updated' )

      # Step 3: Project S to probability matrix
      def f_tolP(s):
          delta = S_iter - s
          # If delta >= 1, set value to 1.
          # Else if 0 < delta < 1, keep the original S_iter value.
          # Else (delta <= 0), set value to 0.
          projected = torch.where(delta >= 1,
                          torch.ones_like(delta),
                          torch.where(delta > 0, delta, torch.zeros_like(delta)))

          return torch.sum(projected) - 2 * self.m

      if f_tolP(0) <= 0:
        S = torch.where(S_iter >= 1,
                          torch.ones_like(S_iter),
                          torch.where((S_iter > 0) & (S_iter < 1),
                                      S_iter,
                                      torch.zeros_like(S_iter)))
      else:
          s = self.bisection_method(f_tolP, 0, torch.max(S_iter),5)
          delta = S_iter - s
          S = torch.where(delta >= 1,
                          torch.ones_like(delta),
                          torch.where(delta > 0, delta, torch.zeros_like(delta)))
          # print(f'Iteration {i:d} projected.' )

      if torch.sum(S > 0) <= 2 * (self.m+2):
        break

    # Create an upper-triangular mask (excluding the diagonal)
    UpS = torch.triu(S, diagonal=1)
    # Get the values from the upper triangle
    flat_upS = UpS.flatten()
    # Get the top k values and their flat indices
    topk_vals, topk_indices = torch.topk(flat_upS, self.m, sorted = True)

    # Convert the flat indices to 2D indices
    num_cols = S.shape[1]
    rows = topk_indices // num_cols
    cols = topk_indices % num_cols
    # Stack the row and column indices together; each row is a pair (row, col)
    indices = torch.stack((rows, cols), dim=1)

    return self.perturb(indices)


  def pgd_wst(self):
    """
    Projected Gradient Descent (PGD) for worst-case perturbation.
    Each iteration updates the perturbation indicator matrix S using gradient descent,
    followed by a projection step to its feasible region (a convex relaxation of indicator matrix).
    The process continues until the maximum number of iterations is reached or
    the number of edges in S is less than or equal to 2 * (m + 2).
    Returns:
        torch.Tensor: The perturbed adjacency matrix.
    """
    # Initialize S as a symmetric matrix and enable gradient tracking
    S = torch.ones(self.n, self.n, device = self.device)
    a = self.alpha
    for i in range(self.max_iter):
      S_iter = S.clone().detach().requires_grad_(True)
      loss = self.obj_wst(S_iter)
      # Step 1. Compute the gradient of loss with respect to S
      loss.backward()
      # Step 2: Update by gradient descent
      S_iter = S_iter + (a/torch.linalg.norm(S_iter.grad, ord=2)) * S_iter.grad
      S_iter = (S_iter + S_iter.t()) / 2
      # print(f'Iteration {i:d} gradent updated' )

      # Step 3: Project S to random sample matrix
      def f_tolP(s):
          delta = S_iter - s
          # If delta >= 1, set value to 1.
          # Else if 0 < delta < 1, keep the original S_iter value.
          # Else (delta <= 0), set value to 0.
          projected = torch.where(delta >= 1,
                          torch.ones_like(delta),
                          torch.where(delta > 0, delta, torch.zeros_like(delta)))

          return torch.sum(projected) - 2 * self.m

      if f_tolP(0) <= 0:
        S = torch.where(S_iter >= 1,
                          torch.ones_like(S_iter),
                          torch.where((S_iter > 0) & (S_iter < 1),
                                      S_iter,
                                      torch.zeros_like(S_iter)))
      else:
          s = self.bisection_method(f_tolP, 0, torch.max(S_iter),5)
          delta = S_iter - s
          S = torch.where(delta >= 1,
                          torch.ones_like(delta),
                          torch.where(delta > 0, delta, torch.zeros_like(delta)))
          # print(f'Iteration {i:d} projected.' )

      # clear_output(wait=True)

      if torch.sum(S > 0) <= 2 * (self.m+2):
        # print("S(prob) is too sparse. Exit the loop." )
        break

    # Create an upper-triangular mask (excluding the diagonal)
    UpS = torch.triu(S, diagonal=1)
    # Get the values from the upper triangle
    flat_upS = UpS.flatten()

    # Get the top k values and their flat indices
    topk_vals, topk_indices = torch.topk(flat_upS, self.m, sorted = True)

    # Convert the flat indices to 2D indices
    num_cols = S.shape[1]
    rows = topk_indices // num_cols
    cols = topk_indices % num_cols
    # Stack the row and column indices together; each row is a pair (row, col)
    indices = torch.stack((rows, cols), dim=1)
    return self.perturb(indices)

  def randomAttack(self):
    """
    Randomly select m edges and then perturbed (flipped) in the adjacency matrix.
    Returns:
        torch.Tensor: The perturbed adjacency matrix.
    """
    B = []
    count = 0
    for i in range(self.n):
        for j in range(i + 1, self.n):
            B.append([i, j])
            count += 1

    B = np.array(B)
    c = np.random.choice(len(B), self.m, replace=False)
    S = B[c, :]
    # print(np.shape(S))

    return self.perturb(S)

 

  def greedy_avg(self):
    """
    Greedy algorithm for average perturbation.
    Each iteration selects the edge that maximizes the reward (based on the objective function)
    and adds it to the perturbation set S. The process continues until m edges are selected.
    Returns:
        torch.Tensor: The perturbed adjacency matrix.
    """
    S = torch.zeros(self.n, self.n).to(self.device)
    for _ in range(self.m):
      reward = torch.zeros(self.n, self.n)
      S_current = S.clone()
      S_next = S.clone()
      base_score = self.obj_avg(S_current)
      # Get reward for each additional edge perturbation
      for i in range(self.n):
        for j in range(i+1, self.n):
          if S_current[i, j] == 1:
            reward[i, j] = -1
          else:
            S_next[i, j] = 1
            S_next[j, i] = 1
            reward[i, j] = self.obj_avg(S_next) - base_score
            reward[j, i] = reward[i, j]
            S_next[i, j] = 0
            S_next[j, i] = 0
      index = reward.argmax()
      u, v = index // self.n, index % self.n
      S[u,v] = 1
      S[v,u] = 1
    return (self.A + (self.J - 2 * self.A) * S)

  def greedy_wst(self):
    """
    Greedy algorithm for worst-case perturbation.
    Each iteration selects the edge that maximizes the reward (based on the objective function)
    and adds it to the perturbation set S. The process continues until m edges are selected.
    Returns:
        torch.Tensor: The perturbed adjacency matrix.
    """
    S = torch.zeros(self.n, self.n).to(self.device)
    for _ in range(self.m):
      reward = torch.zeros(self.n, self.n)
      S_current = S.clone()
      S_next = S.clone()
      base_score = self.obj_wst(S_current)
      # Get reward for each additional edge perturbation
      for i in range(self.n):
        for j in range(i+1, self.n):
          if S_current[i, j] == 1:
            reward[i, j] = -1
          else:
            S_next[i, j] = 1
            S_next[j, i] = 1
            reward[i, j] = self.obj_wst(S_next) - base_score
            reward[j, i] = reward[i, j]
            S_next[i, j] = 0
            S_next[j, i] = 0
      index = reward.argmax()
      u, v = index // self.n, index % self.n
      S[u,v] = 1
      S[v,u] = 1
    return (self.A + (self.J - 2 * self.A) * S)
