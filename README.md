### RDKMATH
#### by Robert Knight


## Release Notes
The intent is to use only included Python libraries in the default
installation and not require any installation of third party additions
via PIP. RDKMATH will be copy-paste ready for then-antique servers a
decade from now, whereas SCIPY and the others may have moved on, and PIP
installation may not be possible.  RDKMATH is designed for longevity.

Only one function uses SCIPY and that is p_value_t_dist_scipy, which
calculates the p-value, give a t-statistic and degrees of freedom. All
other functions operate without library imports.

Z-Score P-values are obtained using the error function added in Python's
included math library in 2.7.

Includes Significance tests, P-Values for Z and T scores, Hypothesis
testing (via confidence interval of slope), confidence intervals,
regression, and other statistical measures such as variance, standard
deviation, etc...

This has been created and testing using text books on the subject.  The
code is original.  This is an ongoing project.

Email bobby@rdknight.net to discuss.



