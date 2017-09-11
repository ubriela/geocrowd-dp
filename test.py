import numpy as np
from scipy.stats import rice
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

# Calculate a few first moments.
# composed of letters ['mvsk'] specifying which moments to compute where
# 'm' = mean, 'v' = variance, 's' = (Fisher's) skew and 'k' = (Fisher's) kurtosis.
# (default='mv')
b = 1.0
mean, var, skew, kurt = rice.stats(b, moments='mvsk')
# stats(b, loc=0, scale=1, moments='mv')
# b is a shape parameter = (v^2/2*delta^2)
# scale parameter = v^2 + 2*delta^2

# Display the probability density function
# ppf = Percent point function (inverse of cdf - percentiles).
x = np.linspace(rice.ppf(0.01, b), rice.ppf(0.99, b), 100)
ax.plot(x, rice.pdf(x, b), 'r-', lw=5, alpha=0.6, label='rice pdf')

# rv = rice(b)
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

# vals = rice.ppf([0.001, 0.5, 0.999], b)
# np.allclose([0.001, 0.5, 0.999], rice.cdf(vals, b))
# r = rice.rvs(b, size=1000)
# ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
# ax.legend(loc='best', frameon=False)
plt.show()