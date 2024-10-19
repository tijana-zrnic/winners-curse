import numpy as np
from scipy.stats import norm
from utils import max_z_width

"""
    Conditional inference
"""
def conditional_inference(y, Sigma, A, b, eta, alpha = 0.1, grid_radius = 10, num_gridpoints = 10000):
    
    m = len(y)

    c = Sigma @ eta / (eta.T @ Sigma @ eta)
    z = (np.eye(m) - np.outer(c,eta)) @ y
    Az = A @ z
    Ac = A @ c    
    V_frac = np.divide(b - Az, Ac)
    if (Ac < 0).any():
        V_minus = np.max(V_frac[Ac < 0])
    else:
        V_minus = - np.inf
    if (Ac > 0).any():
        V_plus = np.min(V_frac[Ac > 0])
    else:
        V_plus = np.inf
    eta_dot_y = eta.dot(y)
    sigma = np.sqrt(eta.T @ Sigma @ eta)
    
    grid = np.linspace(eta_dot_y - grid_radius, eta_dot_y + grid_radius, num_gridpoints)
    
    ci_l = grid[0]
    ci_u = grid[-1]
    found_l = False
    found_u = False
    
    for i in range(len(grid)):
        mu = grid[i]
        num = norm.cdf((eta_dot_y - mu)/sigma) - norm.cdf((V_minus - mu)/sigma)
        denom = norm.cdf((V_plus - mu)/sigma) - norm.cdf((V_minus - mu)/sigma)
        
        if not found_l:
            if num/denom < 1-alpha/2:
                ci_l = mu
                found_l = True
            
        if not found_u and mu >= eta_dot_y:
            if num/denom < alpha/2:
                ci_u = mu
                found_u = True
        
        if found_u and found_l:
            break
            

    return [ci_l, ci_u]

"""
    Hybrid inference
"""
def hybrid_inference(y, Sigma, A, b, eta, alpha = 0.1, beta=0.01, num_gridpoints = 10000, SI_halfwidth = None):
    
    m = len(y)
    
    c = Sigma @ eta / (eta.T @ Sigma @ eta)
    z = (np.eye(m) - np.outer(c,eta)) @ y
    Az = A @ z
    Ac = A @ c
    V_frac = np.divide(b - Az, Ac)
    if (Ac < 0).any():
        V_minus = np.max(V_frac[Ac < 0])
    else:
        V_minus = - np.inf
    if (Ac > 0).any():
        V_plus = np.min(V_frac[Ac > 0])
    else:
        V_plus = np.inf
    
    eta_dot_y = eta.dot(y)
    sigma = np.sqrt(eta.T @ Sigma @ eta)

    if SI_halfwidth == None:
        SI_halfwidth = eta.dot(max_z_width(Sigma, beta)*np.sqrt(np.diag(Sigma)))
    
    grid = np.linspace(eta_dot_y - SI_halfwidth, eta_dot_y + SI_halfwidth, num_gridpoints)
    
    ci_l = grid[0]
    ci_u = grid[-1]
    found_l = False
    found_u = False
    
    for i in range(len(grid)):
        mu = grid[i]
        V_minus_hybrid = np.maximum(V_minus, mu - SI_halfwidth)
        V_plus_hybrid = np.minimum(V_plus, mu + SI_halfwidth) 
        
        num = norm.cdf((eta_dot_y - mu)/sigma) - norm.cdf((V_minus_hybrid - mu)/sigma)
        denom = norm.cdf((V_plus_hybrid - mu)/sigma) - norm.cdf((V_minus_hybrid - mu)/sigma)
        if not found_l:
            if num/denom < 1-(alpha-beta)/(2*(1-beta)):
                ci_l = grid[i-1]
                found_l = True
            
        if not found_u and mu >= eta_dot_y:
            if num/denom < (alpha-beta)/(2*(1-beta)):
                ci_u = mu
                found_u = True
        
        if found_u and found_l:
            break
            
    return [ci_l, ci_u]

"""
    Max-z simultaneous inference
"""
def max_z_inference(point_estimate, Sigma, alpha = 0.1):
    halfwidth = max_z_width(Sigma, alpha)*np.sqrt(np.diag(Sigma))
    return [point_estimate - halfwidth, point_estimate + halfwidth]

"""
    Locally simultaneous inference
"""
def locally_simultaneous_inference(X, Sigma, plausible_gap, alpha = 0.1, nu = 0.01):
    ihat = np.argmax(X)
    point_estimate = X[ihat]
    plausible_inds = plausible_winners(X, plausible_gap)
    Sigma_plausible = Sigma[np.ix_(plausible_inds, plausible_inds)]
    var_ihat = Sigma[ihat, ihat]
    local_halfwidth = max_z_width(Sigma_plausible, alpha-nu)
    SI_halfwidth = max_z_width(Sigma, alpha)
    halfwidth = np.minimum(local_halfwidth, SI_halfwidth)*np.sqrt(var_ihat)
    return [point_estimate - halfwidth, point_estimate + halfwidth]

def plausible_winners(X, plausible_gap):
    return np.where(X >= np.max(X) - plausible_gap)[0]

def LSI_union_bound(X, sigmas, plausible_gap, alpha = 0.1, nu = 0.01):
    ihat = np.argmax(X)
    point_estimate = X[ihat]
    plausible_inds = plausible_winners(X, plausible_gap)
    halfwidth = norm.isf((alpha-nu)/(2*len(plausible_inds)))*sigmas[ihat]
    return [point_estimate - halfwidth, point_estimate + halfwidth]

"""
    Zoom correction (grid search)
"""
def zoom_grid(X, Sigma, alpha = 0.1, simulation_draws = 10000, grid_points = 1000):
    max_radius = max_z_width(Sigma, alpha)*np.sqrt(np.max(np.diag(Sigma)))
    ihat = np.argmax(X)
    l = X[ihat] - max_radius
    u = X[ihat] + max_radius
    noise_mat = np.abs(np.random.multivariate_normal(np.zeros(len(X)), Sigma, simulation_draws))
    # lower bound
    for t in np.linspace(X[ihat] - max_radius, X[ihat], grid_points):
        theta_t = np.minimum(2/3*X + 1/3*t, t)
        theta_t[ihat] = t
        Deltas_t = t - theta_t
        noise_mat_max = np.max(noise_mat * np.greater(noise_mat,Deltas_t.T/2), axis=1)
        radius = np.quantile(noise_mat_max, 1-alpha) # active radius
        if radius > X[ihat]-t:
            l = t
            break
    # upper bound
    for t in np.linspace(X[ihat], X[ihat] + max_radius, grid_points):
        theta_t = np.minimum(2/3*X + 1/3*t, t)
        theta_t[ihat] = t
        Deltas_t = t - theta_t
        noise_mat_max = np.max(noise_mat * np.greater(noise_mat,Deltas_t.T/2), axis=1)
        radius = np.quantile(noise_mat_max, 1-alpha) # active radius
        if radius < t - X[ihat]:
            u = t
            break
    return [l, u]


def zoom_union_bound(X, sigmas, alpha = 0.1, grid_points = 1000):
    m = len(X)
    max_radius = norm.isf(alpha / (2*m))*np.max(sigmas)
    ihat = np.argmax(X)
    radius_grid = np.linspace(0,max_radius, grid_points)
    # lower bound
    for t in np.linspace(X[ihat] - max_radius, X[ihat], grid_points):
        theta_t = np.minimum(2/3*X + 1/3*t, t)
        theta_t[ihat] = t
        Deltas_t = t - theta_t
        tail_vals = np.array([tail_bound(radius_grid[j], Deltas_t, sigmas) for j in range(len(radius_grid))])
        radius = radius_grid[np.where(tail_vals <= alpha)[0][0]] # active radius
        if radius > X[ihat]-t:
            l = t
            break
    # upper bound
    for t in np.linspace(X[ihat], X[ihat] + max_radius, grid_points):
        theta_t = np.minimum(2/3*X + 1/3*t, t)
        theta_t[ihat] = t
        Deltas_t = t - theta_t
        tail_vals = np.array([tail_bound(radius_grid[j], Deltas_t, sigmas) for j in range(len(radius_grid))])
        radius = radius_grid[np.where(tail_vals <= alpha)[0][0]] # active radius
        if radius < t - X[ihat]:
            u = t
            break
    return [l, u]

def tail_bound(r, Deltas, sigmas):
    return np.sum([2*norm.sf(np.maximum(r, Deltas[j]/2), scale = sigmas[j]) for j in range(len(Deltas))])

"""
    Zoom correction (step-down)
"""
def zoom_stepdown(X, sigma, alpha = 0.1):
    Deltas = -np.sort(X - np.max(X))
    m = len(Deltas)
    # lower bound
    alpha_hat = alpha
    for k in range(m):
        r_hat_k = norm.isf(alpha_hat / (2*(m - k)), scale = sigma)
        if Deltas[k] <= 4 * r_hat_k:
            r_l = r_hat_k
            break
        else:
            alpha_hat -= 2*norm.sf((Deltas[k] - r_hat_k) / 3, scale = sigma)
    # upper bound
    alpha_hat = alpha
    for k in range(m):
        r_hat_k = norm.isf(alpha_hat / (2*(m - k)), scale = sigma)
        if Deltas[k] <= 2 * r_hat_k:
            r_u = r_hat_k
            break
        else:
            alpha_hat -= 2*norm.sf((Deltas[k] + norm.isf(alpha/2, scale=sigma)) / 3, scale = sigma)
    return [np.max(X) - r_l, np.max(X) + r_u]