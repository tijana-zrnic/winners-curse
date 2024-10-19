import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

"""
    Max-z quantile
"""
def max_z_width(Sigma, err_level = 0.1, num_draws = 100000):
    m = Sigma.shape[0]
    bstrap_noise = np.random.multivariate_normal(np.zeros(m), Sigma, num_draws)
    bstrap_noise_normed = np.divide(bstrap_noise, np.sqrt(np.diag(Sigma)))
    max_noise = np.amax(np.abs(bstrap_noise_normed), axis = 1)
    return np.quantile(max_noise, 1-err_level)

"""
    Plot and save results
"""
def plot_and_save_results(xaxis, zoom_grid_widths, zoom_stepdown_widths, LSI_widths, cond_widths, hybrid_widths, plot_title, ylabel, xlabel, filename, SoS_width=None, alpha=0.1, fill_between=True, baseline_val = None, ylim = None, legend=True):
    zoom_grid_median = np.quantile(zoom_grid_widths, 0.5, axis=1)
    zoom_grid_95 = np.quantile(zoom_grid_widths, 0.95, axis=1)
    zoom_grid_5 = np.quantile(zoom_grid_widths, 0.05, axis=1)
    
    zoom_stepdown_median = np.quantile(zoom_stepdown_widths, 0.5, axis=1)
    zoom_stepdown_95 = np.quantile(zoom_stepdown_widths, 0.95, axis=1)
    zoom_stepdown_5 = np.quantile(zoom_stepdown_widths, 0.05, axis=1)

    LSI_median = np.quantile(LSI_widths, 0.5, axis=1)
    LSI_95 = np.quantile(LSI_widths, 0.95, axis=1)
    LSI_5 = np.quantile(LSI_widths, 0.05, axis=1)

    cond_median = np.quantile(cond_widths, 0.5, axis=1)
    cond_95 = np.quantile(cond_widths, 0.95, axis=1)
    cond_5 = np.quantile(cond_widths, 0.05, axis=1)
        
    hybrid_median = np.quantile(hybrid_widths, 0.5, axis=1)
    hybrid_95 = np.quantile(hybrid_widths, 0.95, axis=1)
    hybrid_5 = np.quantile(hybrid_widths, 0.05, axis=1)

    plt.clf()
    
    plt.plot(xaxis, cond_median, 'orange', label='conditional', linewidth=3)
    if fill_between:
        plt.fill_between(xaxis, cond_5, cond_95, color='orange', alpha = 0.08)
        
    plt.plot(xaxis, hybrid_median, 'peru', label='hybrid', linewidth=3)
    if fill_between:
        plt.fill_between(xaxis, hybrid_5, hybrid_95, color='peru', alpha = 0.15)
            
    if SoS_width != None:
        plt.plot(xaxis, [SoS_width] * len(xaxis), 'mediumseagreen', label='SoS', linewidth=3)
    
    plt.plot(xaxis, LSI_median, 'tomato', label='locally simultaneous', linewidth=3)
    if fill_between:
        plt.fill_between(xaxis, LSI_5, LSI_95, color='tomato', alpha = 0.15)

    plt.plot(xaxis, zoom_grid_median, 'skyblue', label='zoom (grid)', linewidth=3)
    if fill_between:
        plt.fill_between(xaxis, zoom_grid_5, zoom_grid_95, color='skyblue', alpha = 0.15)
        
    plt.plot(xaxis, zoom_stepdown_median, 'steelblue', label='zoom (step-down)', linewidth=3)
    if fill_between:
        plt.fill_between(xaxis, zoom_stepdown_5, zoom_stepdown_95, color='steelblue', alpha = 0.15)
    
    if baseline_val != None:
        plt.plot(xaxis, [baseline_val] * len(xaxis), 'gray', linewidth=3, linestyle = 'dashed')
    else:
        plt.plot(xaxis, [2*scipy.stats.norm.isf(alpha/2)] * len(xaxis), 'gray', linewidth=3, linestyle = 'dashed')

    if legend:
        plt.legend(bbox_to_anchor = (1.05,1.035), borderpad=1, labelspacing = 1, fontsize=14)
        
    plt.title(plot_title, fontsize = 22)
    pivot_ax = plt.gca()
    pivot_ax.set_ylabel(ylabel, fontsize = 22)
    pivot_ax.set_xlabel(xlabel, fontsize = 22)
    if ylim != None:
        plt.ylim(ylim)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    path = filename
    plt.savefig(path, bbox_inches='tight')


"""
    Polyhedron describing selecting the winner
"""
def inference_on_winner_polyhedron(m, selected_ind):
    b = np.zeros(m-1)
    A = np.eye(m)
    A = np.delete(A, selected_ind, 0)
    A[:, selected_ind] = -1
    return A, b

"""
    For plotting
"""
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def make_intervals(cis, labels, plot_title, xlabel, legend=True):
    k = len(cis)
    colors = ['orange','peru','mediumseagreen','tomato','skyblue'][5-k:]
    
    for j in range(k):
        plt.plot([cis[j][0], cis[j][1]],[0.7 - j*0.1, 0.7 - j*0.1], linewidth=14, color=lighten_color(colors[j]), path_effects=[pe.Stroke(linewidth=16, offset=(-1,0), foreground=colors[j]), pe.Stroke(linewidth=16, offset=(1,0), foreground=colors[j]), pe.Normal()], label=labels[j], solid_capstyle='butt')
    plt.ylabel("")
    plt.title(plot_title, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.yticks([])
    plt.ylim([0.7-k*0.1 + 0.05,0.75])
    plt.xlim([None, None])
    if legend:
        plt.legend(bbox_to_anchor = (1.75,1.035), borderpad=1, labelspacing = 1, fontsize=16)

"""
    Pearson correlation coefficient utils
"""
def pearson_confidence_interval(r, n, alpha=0.1):
    z = fisher_transform(r)
    se = 1 / np.sqrt(n - 3)  
    z_crit = norm.ppf(1 - alpha / 2)
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)
    return r_lower, r_upper

def fisher_transform(r):
     return np.arctanh(r)

"""
    Logistic regression utils
"""
def logistic(X, Y):
    regression = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-15,
        fit_intercept=False,
    ).fit(X, Y)
    return regression.coef_.squeeze()

def logistic_stats(X, Y, alpha=0.1, alternative="two-sided"):
    n = Y.shape[0]
    d = X.shape[1]
    pointest = logistic(X, Y)
    mu = expit(X @ pointest)
    V = np.zeros((d, d))
    grads = np.zeros((n, d))
    for i in range(n):
        V += 1 / n * mu[i] * (1 - mu[i]) * X[i : i + 1, :].T @ X[i : i + 1, :]
        grads[i] = (mu[i] - Y[i]) * X[i]
    V_inv = np.linalg.inv(V)
    cov_mat = V_inv @ np.cov(grads.T) @ V_inv
    return pointest, cov_mat/n