import numpy as np
import torch
import torch.nn as nn
from  torch.distributions import Dirichlet
from sklearn.base import BaseEstimator
from scipy.stats import rankdata
from sklearn.neighbors import kneighbors_graph

class TinyAdd(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return x + self.eps


class Generic_DRAM(BaseEstimator):

    def __init__(self, n_models=3, lam=1e-5, max_iter=10, n_samples=20, 
                max_iter_lbfgs=500, lr_lbfgs=1, verbose=False, random_state=123):
        self.n_models = n_models
        self.lam = lam
        self.max_iter = max_iter
        self.n_samples = n_samples
        self.lr_lbfgs = lr_lbfgs
        self.max_iter_lbfgs = max_iter_lbfgs
        self.verbose = verbose
        self.random_state = random_state

    def phiNet(self, X, R, Y):
        def _func(Dsams):
            assert (Dsams.shape[0], Dsams.shape[2]) == R.shape
            return torch.ones(Dsams.shape[:2])
        return _func

    def predict(self, X):
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            posterior = self.post_fn(X)  # shape=(N, K)
            cct = self.loc_fn(X).view(X.shape[0], self.n_models, -1) # shape=(N, K, M)
            Zhat = cct / cct.sum(2, keepdims=True)
            Zhat = (Zhat * posterior.unsqueeze(2)).sum(1)
        return Zhat.numpy()

    def fit(self, X, R, Y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        K, L, (_, M) = self.n_models, self.n_samples, R.shape
        phiNet = self.phiNet(X, R, Y)
        X = torch.FloatTensor(X)
        self.loc_fn = nn.Sequential(nn.Linear(X.shape[1], M*K), nn.Softplus(), TinyAdd(1e-9))
        self.post_fn = nn.Sequential(nn.Linear(X.shape[1], K), nn.Softmax(dim=1), TinyAdd(1e-9))
        params = list(self.loc_fn.parameters()) + list(self.post_fn.parameters())
        for p in params:
            nn.init.normal_(p, mean=0.0, std=0.1)
        gammas = torch.softmax(torch.rand((X.shape[0], L, K)), dim=-1)   # shape=(N, L, K)
        
        # Expectation Maximization
        for em in range(self.max_iter):    
            # Generate label distribution samples
            Dsams = torch.FloatTensor(self.zsample_generator(R, Y, L)) # shape=(N, L, M)
            Dsams /= Dsams.sum(-1, keepdims=True)
            phi = phiNet(Dsams)  # shape=(N, L)
            
            # Maximization step    
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                cct = self.loc_fn(X).view(-1, 1, K, M)    # shape=(N, 1, K, M)
                llh = Dirichlet(cct).log_prob(Dsams.unsqueeze(-2)) # shape=(N, L, K)
                elbo = ((self.post_fn(X).unsqueeze(1).log() + llh) * gammas).sum(2) # shape=(N, L)
                reg = 0
                for p in self.loc_fn.parameters():
                    reg += self.lam * p.square().sum()
                loss = (reg - (elbo * phi).sum(1)).sum()
                if loss.requires_grad:
                    loss.backward()
                return loss
            for _lr in [self.lr_lbfgs, 1e-3, 1e-4]:
                try:
                    optimizer = torch.optim.LBFGS(params, max_eval=None,
                        lr=_lr, max_iter=self.max_iter_lbfgs, 
                        tolerance_change=1e-5, tolerance_grad=1e-6, 
                        history_size=5, line_search_fn='strong_wolfe')
                    loss = optimizer.step(closure)
                    self.lr_lbfgs = _lr
                    break
                except: pass
            if K == 1:
                break

            # Expectation step
            with torch.no_grad():
                posterior = self.post_fn(X) # shape=(N, K)
                cct = self.loc_fn(X).view(-1, 1, K, M) # shape=(N, 1, M, K)
                llh = Dirichlet(cct).log_prob(Dsams.unsqueeze(-2)) # shape=(N, L, K)
                gammas = (posterior.log().unsqueeze(1) + llh)
                gammas = torch.softmax(gammas, dim=2)
            if self.verbose:
                print("iter: %2d, Rho: %.3f" % (em, Rho(self.predict(X), Dsams[:, 0, :].numpy())))
        return self
        
    def sequential_sampling(self, n_samples, n_labels, delta=None):
        '''Output n_samples n_labels-dimensional increasing vectors.'''
        if delta is None:
            delta = n_labels / (n_labels + 1)**2
        U = np.random.rand(n_labels, n_samples)
        samples = np.concatenate([U, delta + np.ones((1, n_samples))])
        
        for k in range(n_labels, 0, -1):
            temp = k * delta + np.power(U[k-1], 1/k) * (samples[k] - (k+1) * delta)
            samples[k-1] = temp
        samples = samples[:-1]
        return samples.T

    def zsample_generator(self, R, Y, n_samples):
        assert (R[Y == 0] == 1).all() == True
        R = R.copy()
        res = np.zeros((R.shape[0], n_samples, R.shape[1]))
        for i in range(R.shape[0]):
            rank, y = R[i], Y[i]
            temp = rankdata(rank, method='dense')
            _m = max(temp)
            opt_delta = _m / (_m+1) * sum([i*(_m+1-i) for i in temp]) / sum([i*(_m**2-i*_m+2*i) for i in temp])
            if (y == 0).any():  # rank includes zero
                samples = self.sequential_sampling(n_samples, np.unique(rank).size-1, delta=opt_delta)
                rank -= 2
                rank[rank == -1] = 0
                samples = samples[:, rank]
                samples[:, y == 0] = 1e-9
            else:   # rank doesn't include zero
                samples = self.sequential_sampling(n_samples, np.unique(rank).size, delta=opt_delta)
                rank -= 1
                samples = samples[:, rank]
            res[i] = samples
        return res


class DRAM_LN(Generic_DRAM):
    def __init__(self, n_models=3, lam=1e-5, max_iter=10, n_samples=20, 
                verbose=False, max_iter_lbfgs=500, lr_lbfgs=1, random_state=123):
        super().__init__(n_models=n_models, lam=lam, max_iter=max_iter,
            n_samples=n_samples, verbose=verbose, max_iter_lbfgs=max_iter_lbfgs, 
            lr_lbfgs=lr_lbfgs, random_state=random_state)

    def fit(self, X, D):
        R, Y = reduce_label_distributions(D)
        return super().fit(X, R, Y)


class DRAM_LP(Generic_DRAM):
    def __init__(self, lam=1e-5, sigma_LP=1, sigma_a=1, K_LP=7, n_samples=20, 
                verbose=False, max_iter_lbfgs=500, lr_lbfgs=1, random_state=123):
        self.sigma_LP = sigma_LP
        self.K_LP = K_LP
        self.sigma_a = sigma_a
        super().__init__(n_models=1, lam=lam, n_samples=n_samples, 
            verbose=verbose, max_iter_lbfgs=max_iter_lbfgs, 
            lr_lbfgs=lr_lbfgs, random_state=random_state)
    
    def label_enhancement(self, X, R, Y):
        Dsams = torch.FloatTensor(self.zsample_generator(R, Y, self.n_samples))
        Dsams /= Dsams.sum(-1, keepdims=True)
        phiNet = self.phiNet(X, R, Y)
        phi = phiNet(Dsams)  # shape=(N, L)
        phi /= phi.sum(1, keepdims=True)
        return ((phi[:,:,None] * Dsams).sum(1)).numpy()

    def phiNet(self, X, R, Y):
        Lprime = 30
        A = kneighbors_graph(X, n_neighbors=self.K_LP, mode='distance').toarray()
        A = np.square(A)
        A[A == 0] = np.inf
        A[np.arange(A.shape[0]), np.arange(A.shape[0])] = 0
        A = np.exp(-A * 0.5 / self.sigma_a**2)
        Dv = np.power(A.sum(1), -.5)
        A = (A * Dv.reshape(-1, 1) * Dv.reshape(1, -1)).astype(np.float32)
        _max, _min = np.max(R-1, axis=1, keepdims=True), np.min(R-1, axis=1, keepdims=True)
        Init = (((R-1) - _min) / (_max - _min)).astype(np.float32)    # shape=(N, M)
        alpha = (np.linspace(0.001, 0.999, Lprime)).reshape(-1, 1, 1).astype(np.float32)  # shape=(L',1,1)
        
        # compute label propagation results
        Z = Init
        for _ in range(5):
            Z = alpha * A @ Z + (1-alpha) * Init[None,:]
        Z = Z.astype(np.float32)
        Z = torch.from_numpy(Z)
        Z = Z.transpose(1, 0)
        
        # filter
        ZR = rankdata(Z, axis=-1)
        _R = rankdata(R, method='min', axis=1)[:,None,:]
        p = np.abs(ZR - _R).sum(2)
        mask = p == np.min(p, axis=1, keepdims=True)    # shape=(N, L')
        def _func(Dsams):
            # Dsams: (N, L, M)
            temp = (-.5*(Dsams.unsqueeze(2) - Z.unsqueeze(1)).square()/self.sigma_LP**2).exp()   # shape=(N,L,L',M)
            # temp = temp * torch.BoolTensor(mask).unsqueeze(1).unsqueeze(-1)
            temp = temp.sum(-1).mean(-1)
            return torch.FloatTensor(temp)
        return _func