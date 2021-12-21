using Knet
using AutoGrad


mutable struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

function identity(x)
    return x
end
  

mutable struct Conv; w; b; p; s; f; end
(c::Conv)(x) = c.f.(conv4(c.w, x, padding=c.p, stride=c.s) .+ c.b)
Conv(c_in::Int, c_out::Int, kernel_size::Int, p=1, s=1, f=relu) = Conv( Param(Knet.atype()(xavier_normal(kernel_size,kernel_size,c_in,c_out; gain=0.02))), param0(1,1,c_out,1), p,s,f)


mutable struct DoubleConv; conv_1; conv_2; end
(c::DoubleConv)(x) = c.conv_2(c.conv_1(x)) 
DoubleConv(c_in::Int, c_out::Int, kernel_size=3) = DoubleConv( Conv(c_in,c_out,kernel_size,1), Conv(c_out,c_out, kernel_size,1))


mutable struct InConv; dc; end
(c::InConv)(x) = c.dc(x)
InConv(c_in::Int, c_out::Int) = InConv( DoubleConv(c_in, c_out))


mutable struct OutConv; conv; end
(c::OutConv)(x) = c.conv(x)
OutConv(c_in::Int, c_out::Int) = OutConv( Conv(c_in, c_out, 1, 0, 1, identity))


mutable struct Down; dc; end
(c::Down)(x) = c.dc(pool(x; window=2, mode=0) )
Down(c_in::Int, c_out::Int) = Down(DoubleConv(c_in, c_out))


mutable struct DeConv; w;p;s; end
(c::DeConv)(x) = deconv4(c.w, x, padding=c.p, stride=c.s)
DeConv(kernel_size::Int, p::Int, s::Int) = DeConv(Param(Knet.atype()(xavier_normal(kernel_size, kernel_size, 1, 1; gain=0.02))), p,s)


mutable struct Up; doublec; dec;
  function Up(c_in::Int, c_out::Int, dtype)
    new(DoubleConv(c_in, c_out), DeConv(4,1,2))
  end
end
#forward
(c::Up)(x1,x2) = begin 

  H, W, C, B = size(x1)
  x1 = reshape(x1, (H, W, 1, C*B))
  x1 = c.dec(x1)
  x1 = reshape(x1, size(x1)[1:2]..., C, B)
        
  #concat on C dim
  x = c.doublec(cat(x2, x1, dims=3))
  return x
end
