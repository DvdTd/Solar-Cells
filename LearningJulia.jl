# using Pkg
# Pkg.add("cuDNN")
# Pkg.add("HighDimPDE")
using HighDimPDE, Plots, Random

function forty2()
    # x = 0
    # while x <3
    #     msg = "Hello World"
    #     println(msg) # print(str) does not add a new line after
    #     x += 1
    # end
    x[1] =5
    return y
end

x=[3]
y=2
# println(x)
# println(forty2())
# println(x)
a = [1,2,3]
# println(a[1])
# println(1f-2)

x = range(0, 10, length=100)
y1 = sin.(x)
y2 = cos.(x)
# display(plot(x, [y1 y2]))

m(x) = sum(x.^2, dims=3)
g(x) = exp( -sum(x.^2))
# println(m([0,1,2,3]))
d = 2
x0 = fill(2.,d)
println(sum(x0.*x0))
pn = UniformSampling(0,1)
Sampler
println(pn)
