import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from scipy.optimize import minimize

"""
Helper functions for the GP and BO notebook.

author: Chenran Xu  (chenran.xu@kit.edu)
"""

# plot helper functions
def plot_gpr_samples(gpr_model: GaussianProcessRegressor, ax, x=np.linspace(0,5,100), n_samples=5, random_state=0):
    """Plot samples drawn from the Gaussian process model.
    modified from sklearn example: https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    random_state: int, RandomState instance or None, defualt=0
        Determines random number generation to randomly draw samples.
        Pass an int for reproducible results across multiple function calls.
    """
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples, random_state=random_state)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sample #{idx + 1}",
        )
    ax.plot(x, y_mean, color="black", label=r"GP mean $\mu(x)$")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm 1 \sigma$",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

def plot_gp(gpr, x, y, x_samples, y_samples, ax=None):
    """Helper function to plot GP posterior

    Input:
        gpr: GaussianProcessRegression
        x, y: fine array representing the target function
        x_samples, y_samples: noisy samples used for building GP
        ax: matplotlib.pyplot axes.axes
    """
    if ax is None:
        ax = plt.gcf.add_subplot()
    ax.plot(x, y, label="True f")
    y_mean, y_std = gpr.predict(x.reshape(-1, 1), return_std=True)
    ax.plot(x, y_mean, label=r"GP mean $\mu(x)$", color='black')
    ax.fill_between(
        x,
        np.array(y_mean - y_std),
        np.array(y_mean + y_std),
        alpha=0.3,
        color="grey",
        label=r"$\pm 1 \sigma$",
    )
    ax.plot(x_samples, y_samples, "*", label="Noisy Samples")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

def plot_gp_with_acq(gpr, x, y, x_samples, y_samples, y_acq, axes, fig, legend=True):

    ax1, ax2 = axes
    x_acq_argmax = np.argmax(y_acq)
    
    # plotting
    plot_gp(gpr, x, y, x_samples, y_samples, ax=ax1)
    ax1.set_xticks([])
    
    ax2.set_xlabel('x')
    ax2.plot(x, y_acq, color='g', label=r'Acquisition $\alpha$')
    ax2.plot(x[x_acq_argmax], y_acq[x_acq_argmax], '*', color='r', label=r"argmax($\alpha$)")
    if legend:
        fig.subplots_adjust(0,0,0.8,0.85,hspace=0.1)
        fig.legend(bbox_to_anchor = (0.95,0.3,0.2,0.5))

def plot_bo_result(yhist, ax, n_tries = None, nsteps=None, label=None):
    if n_tries is None or nsteps is None:
        ybest = np.asarray(yhist)
        n_tries, nsteps = ybest.shape
    else:
        ybest = np.zeros((n_tries, nsteps))
    for i in range(n_tries):
        for n in range(nsteps):
            ybest[i,n] = np.max(yhist[i][:n+1])
    
    ybest_mean = np.mean(ybest, axis=0)
    ybest_std = np.std(ybest, axis=0)

    ax.plot(ybest_mean, label=label)
    ax.fill_between(np.arange(nsteps), ybest_mean-ybest_std, ybest_mean+ybest_std, alpha=0.3)



# Acquisition function classes
class Acquisition:
    """Acquisition function base class"""

    def __init__(self):
        pass

    def get_acq(self, x, gp: GaussianProcessRegressor):
        return NotImplementedError

    def suggest_next_sample(self, gp: GaussianProcessRegressor, bounds):
        """Return the next point to sample by maximizing the acquisition function
        
        Input:
            gp: GaussianProcessRegressor object
            bounds: Optimization ranges with a shape of (n, 2), e.g. [[x1_min, x1_max],... [xi_min, xi_max]]
        """
        # initial guesses
        xdim = bounds.shape[0]
        x_tries = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(5000, xdim))
        ys = self.get_acq(x_tries,gp)
        max_acq = ys.max()
        x_max = x_tries[ys.argmax()]
        # simply use scipy.optimize.minimize for now
        res = minimize(
            lambda x: -1*self.get_acq(x.reshape(-1, xdim), gp), 
            x_max.reshape(xdim,),
            bounds=bounds,
            )
        
        if res.success and -res.fun >= max_acq:
            x_max = res.x

        # ensure the returned point is within bounds
        return np.clip(x_max.reshape(-1,xdim), bounds[:, 0], bounds[:, 1])

class AcqEI(Acquisition):
    """
    Expected Improvement (EI) acquisition
    a(x) = E[ f(x) - f(best)]

    Parameter:
        xi : hyperparamter for exploitation-exploration tradeoff
    """
    def __init__(self, xi=0.):

        super().__init__()
        self.xi = xi

    def get_acq(self, x, gp):
        """Calculate EI at point x"""
        if len(np.shape(x)) == 1:
            x = np.array(x).reshape(-1,1)
        y_mean, y_std = gp.predict(x, return_std=True)
        y_best = np.max(gp.y_train_)
        imp = y_mean - y_best - self.xi
        z = imp / y_std
        return imp * norm.cdf(z) + y_std * norm.pdf(z)


class AcqUCB(Acquisition):
    """
    Upper confidence bound (UCB) acquisition
    a(x) = mu(x) + k * sigma(x)
    
    Parameter:
        k : hyperparamter for exploitation-exploration tradeoff
    """
    def __init__(self, k = 2.):

        super().__init__()
        self.k = k

    def get_acq(self, x, gp: GaussianProcessRegressor):
        """Calculate UCB at point x"""
        if len(np.shape(x)) == 1:
            x = np.array(x).reshape(-1,1)
        mu, sigma = gp.predict(x, return_std=True)

        return mu + sigma * self.k
