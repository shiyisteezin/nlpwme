# This file was generated, do not modify it. # hide
import Pkg
Pkg.add(["CausalityTools", "Statistics", "Plots"])
using CausalityTools, Statistics, Plots

# Span a range of x-y coupling strengths
c = 0.0:0.1:1.0

# Number of observations in each time series
npts = 500

# Number of unique realizations of each system
n_realizations = 1000

# Get MI for multiple realizations of two systems,
# saving three quantiles for each c value
mi = zeros(length(c), 3, 2)

# Define an estimator for MI
b = RectangularBinning(4)
mi_estimator = VisitationFrequency(b)

for i in 1 : length(c)

    tmp = zeros(n_realizations, 2)

    for k in 1 : n_realizations

        # Obtain time series realizations of the two 2D systems
        # for a given coupling strength and random initial conditions
        lmap = trajectory(logistic2_unidir(u₀ = rand(2), c_xy = c[i]), npts - 1, Ttr = 1000)
        ar1 = trajectory(ar1_unidir(u₀ = rand(2), c_xy = c[i]), npts - 1)

        # Compute the MI between the two coupled components of each system
        tmp[k, 1] = mutualinfo(lmap[:, 1], lmap[:, 2], mi_estimator)
        tmp[k, 2] = mutualinfo(ar1[:, 1], ar1[:, 2], mi_estimator)
    end

    # Compute lower, middle, and upper quantiles of MI for each coupling strength
    mi[i, :, 1] = quantile(tmp[:, 1], [0.025, 0.5, 0.975])
    mi[i, :, 2] = quantile(tmp[:, 2], [0.025, 0.5, 0.975])
end

# Plot distribution of MI values as a function of coupling strength for both systems
plot(c, mi[:, 2, 1], label = "2D chaotic logistic maps", lc = "black",
    ribbon = (mi[:, 2, 1] - mi[:, 1, 1], mi[:, 3, 1] - mi[:, 2, 1]), c = "black", fillalpha = 0.3,
    legend = :topleft)
plot!(c, mi[:, 2, 2], label = "2D order-1 autoregressive", lc = "red",
    ribbon = (mi[:, 2, 2] - mi[:, 1, 2], mi[:, 3, 2] - mi[:, 2, 2]), c = "red", fillalpha = 0.3)
xlabel!("Coupling strength")
ylabel!("Mutual information")