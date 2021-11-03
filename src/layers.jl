using Images
using Knet
using AutoGrad

mutable struct Linear
  w
  b
  function Linear(inputsize::Int, outputsize::Int,
                  atype=Array{Float32}, scale::Float32=0.1)
      w = convert(atype, scale .*  randn(outputsize, inputsize))
      b = convert(atype, zeros(outputsize,1))
      new(param(w), param(b))
  end
end
(l::Linear)(x) = l.w *x .+ l.b

mutable struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)


mutable struct Conv; w; b; p; s; f; end
(c::Conv)(x) = c.f.(conv4(c.w, x, padding=c.p, stride=c.s) .+ c.b)
Conv(c_in::Int, c_out::Int, kernel_size::Int, p=1, s=1, f=relu) = Conv(param(kernel_size,kernel_size,c_in,c_out), param0(1,1,c_out,1), p,s,f)


mutable struct DoubleConv; conv_1; conv_2; end
(c::DoubleConv)(x) = c.conv_2(c.conv_1(x)) 
DoubleConv(c_in::Int, c_out::Int, kernel_size=3) = DoubleConv( Conv(c_in, c_out, kernel_size), Conv(c_out,c_out, kernel_size))


mutable struct InConv; dc; end
(c::InConv)(x) = c.dc(x)
InConv(c_in::Int, c_out::Int) = InConv( DoubleConv(c_in, c_out))

mutable struct OutConv; conv; end
(c::OutConv)(x) = c.conv(x)
OutConv(c_in::Int, c_out::Int) = OutConv( Conv(c_in, c_out, 3, 1, 1))


mutable struct Down; dc; end
(c::Down)(x) = c.dc( pool(x; window=2, mode=0) )
Down(c_in::Int, c_out::Int) = Down(DoubleConv(c_in, c_out))


# found out that imresize uses bilinear interpolation underneath
# https://discourse.julialang.org/t/how-to-upsample-an-array-or-image/12415 
mutable struct Up; scale; dc; 
  function Up(c_in::Int, c_out::Int, scale=2)
    scale = scale
    dc = DoubleConv(c_in, c_out)
    new(scale,dc)
  end
end
#forward
(c::Up)(x1,x2) = begin 

  # do x1 = c.up(x1)
  # upsample first 2 dims, keep others same
  # (assuming H,W are first 2 dims, C,N are last 2)  
  output_size = size(x1)[1:end-2].*c.scale..., size(x1)[end-1:end]...
  x1 = imresize(x1, output_size) #bilinear interpolation

  #concat on C dim
  x = c.dc( cat(x2,x1, dims=3) )
  return x
end
