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
# println(sum(x0.*x0))
# pn = UniformSampling(0,1)
# println(pn)

# Pick 1st element: `map(x -> x[1],a)` or `a[:,1]`

v = [1.1, 2.2, 3.3, 4.3]
s = 4

f(t, x) = exp(-x^2/(2t))/√(2π*t)
u(t, x) = f(t+0.1, x+2) + 2f(t+0.2, x-2)
t = range(0, 3; length=3)
x = range(-5, 5; length=3)
T = u.(t', x)
# println(T)
# display(wireframe(t, x, T; colorbar=false, xlabel="t", ylabel="x", zlabel="T", camera=(30, 45)))
# start at 0,0, for y: (for x)


println([(i-1)*3 + j for i in range(1, 5)  for j in range(1, 3)])
println("____")
println([i for i in range(1, 5*3)])
