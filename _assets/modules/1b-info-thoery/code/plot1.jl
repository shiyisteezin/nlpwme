# This file was generated, do not modify it. # hide
using PyPlot
figure(figsize=(8, 6))

x = range(0, 1, length=1000)
f(x) = .- x .* log2.(x) .- (1 .- x) .* log2.(1 .- x)

xticks(range(0,1,length=5))
yticks(range(0,1,length=5))

plot(x, f(x))
xlabel("Pr(x)=1")
ylabel("H(x)")
scatter([0.5],[1])
savefig(joinpath(@OUTPUT, "ent1.svg"))