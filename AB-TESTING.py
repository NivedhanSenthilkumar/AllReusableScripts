
"""LIBRARIES"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import binom
from scipy.stats import chi2_contingency, mannwhitneyu, norm, t, ttest_ind
%matplotlib inline


                                   """Discrete Metrics"""
#Let's consider first discrete metrics, e.g. click-though rate. We randomly show visitors one of two possible designs of an advertisement, and based on how many of them click on it we need to determine whether our data significantly contradict the hypothesis that the two designs are equivalently efficient.
np.random.seed(42)
x = np.random.binomial(n=1, p=0.6, size=15)
y = np.random.binomial(n=1, p=0.4, size=19)
_, (a, c) = np.unique(x, return_counts=True)
_, (b, d) = np.unique(y, return_counts=True)

df = pd.DataFrame(data=[[a, b], [c, d]],
                 index=["click", "no click"],
                 columns=["A", "B"])
m = df.values
print("- Observations:")
print(f"  - Version A: = {x}")
print(f"  - Version B: = {y}")
print("")
print("- Contingency table:")
display(df)


                            1.1 """Fisher's exact test"""
#Since we have a 2x2 contingency table we can use Fisher's exact test to compute an exact p-value and test our hypothesis.
def hypergeom(k, K, n, N):
    """Probability mass funciton of the hypergeometric distribution."""
    return binom(K, k) * binom(N-K, n-k) / binom(N, n)


def fisher_prob(m):
    """Probability of a given observed contingency table according to Fisher's exact test."""
    ((a, b), (c ,d)) = m
    k = a
    K = a+b
    n = a+c
    N = a+b+c+d
    return hypergeom(k, K, n, N)

def fisher_probs_histogram(m):
    """Computes prob mass function histogram accroding to Fisher's exact test."""
    neg_val = -min(m[0,0], m[1,1])
    pos_val = min(m[1,0], m[1,0])
    probs = []
    for k in range(neg_val, pos_val+1):
        m1 = m + np.array([[1, -1], [-1, 1]]) * k
        probs.append(fisher_prob(m1))
    return probs

bars_h = np.array(fisher_probs_histogram(m))

f, ax = plt.subplots(figsize=(6, 3))
ii = np.arange(len(bars_h))
ax.bar(ii, bars_h)
idxs = bars_h <= fisher_prob(m)
ax.bar(ii[idxs], bars_h[idxs], color='r')
ax.set_ylabel("prob density")
p_val = bars_h[idxs].sum()
neg_val = -min(m[0,0], m[1,1])
pos_val = min(m[1,0], m[1,0])
ax.bar(ii[-neg_val], bars_h[-neg_val], color='orange')

ax.set_xticks(ii)
ax.set_xticklabels(np.arange(neg_val, pos_val+1))
f.tight_layout()
print(f"- Fisher's exact test: p-val = {100*p_val:.1f}%")


                         """ 1.2 Pearson's chi-squared test """
#Fisher's exact test has the important advantage of computing exact p-values. But if we have a large sample size, it may be computationally inefficient. In this case, we can use Pearson's chi-squared test to compute an approximate p-value.
chi2_val, p_val = chi2_contingency(m, correction=False)[:2]
print("- Pearson's chi-squared t-test:")
print(f"   - χ2 value: {chi2_val:.3f}")
print(f"   - p-value: {p_val*100:.1f}%")


                              """2. Continuous metrics"""
#Let's now consider the case of a continuous metrics, e.g. average revenue per user. We randomly show visitors of our website one of two possible layouts of products for sale, and based on how much revenue each user generated in a month we want to determine whether our data significantly contradict the hypothesis that the two website layouts are equivalently efficient.
np.random.seed(42)
n_x, n_y = 17, 14
d1 = norm(loc=200, scale=100)
d2 = norm(loc=280, scale=90)
disc = 50
x = (d1.rvs(size=n_x) / disc).astype(int) * disc
y = (d2.rvs(size=n_y) / disc).astype(int) * disc
print("- Observations:")
print(f"  - Version A: = {x}")
print(f"  - Version B: = {y}")
print("")
print(f"- Distribution plot:")

f, ax = plt.subplots(figsize=(6, 3))
for i, (x_, l_, c_) in enumerate(zip([x, y], ["A", "B"], ["tab:blue", "tab:olive"])):
    v, c = np.unique(x_, return_counts=True)
    ax.bar(v-5+10*i, c, width=10, label=l_, color=c_)

ax.set_xlabel("purchase in $")
ax.set_ylabel("count")
ax.legend();

def plot_pval(distribution, t_val, xlims=(-5, 5), ylims=(0, 0.5)):
    xxx = np.linspace(*xlims, 1000)
    f, ax = plt.subplots(figsize=(4,3))
    ax.plot(xxx, distribution.pdf(xxx))
    ax.set_ylim(ylims)
    ax.vlines(t_val, 0, stat_distrib.pdf(t_val), color='orange')
    ax.plot(t_val, stat_distrib.pdf(t_val), 'o', color='orange')
    xp = xxx <= t_val
    ax.fill_between(xxx[xp], xxx[xp] * 0, stat_distrib.pdf(xxx[xp]), color='r')
    xp = xxx >= -t_val
    ax.fill_between(xxx[xp], xxx[xp] * 0, stat_distrib.pdf(xxx[xp]), color='r')
    ax.set_ylabel("prob denisty")
    f.tight_layout()
    return f, ax


                                         """2.1-Z-test """"
#The Z-test can be applied under the following assumptions.The observations are normally distributed (or the sample size is large).The sampling distributions have known variance σ_X and σ_Y.Under the above assumptions, the Z-test relies on the observation that the following Z statistic has a standard normal distribution.

# Known standard deviations
s_x = 100
s_y = 90

# Z value
z_val = (x.mean() - y.mean()) / np.sqrt(s_x**2/n_x + s_y**2/n_y)

# Test statistic distribution under null hypothesis H0
stat_distrib = norm(loc=0, scale=1)

# p-value
p_val = stat_distrib.cdf(z_val) * 2

print("- Z-test:")
print(f"   - z value: {z_val:.3f}")
print(f"   - p-value: {p_val*100:.1f}%")
plot_pval(stat_distrib, z_val);