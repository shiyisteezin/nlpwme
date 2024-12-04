# This file was generated, do not modify it. # hide
using PyPlot
figure(figsize=(8, 6))

x = range(0,0.5,length=1000)
y = range(0.5,1,length=1000)

f(x) =  (x .* log2.(x) .+ (1 .- x) .* log2.(1 .- x)) .* 0.66
g(y) = (.- f(y)) 

plot(x, f(x) , y, g(y))

yticks([])
ylabel("H(x)")
savefig(joinpath(@OUTPUT, "ent3.svg"))