include("layers.jl")
include("loss.jl")

using Knet
using AutoGrad
using Distributions: Binomial


mutable struct Denoise_net 
    removal
    approximator
    dtype


    function Denoise_net(feature_num::Int, dtype)
        approximator = Chain(
            Conv(3, 3, 1, 0, 1) #c_in, c_out, kernel_size, padding, stride
        )
                
        removal =  Chain(
            InConv(3,feature_num),
        
            Down(feature_num,feature_num*2),
            Down(feature_num*2,feature_num*4),
            Down(feature_num*4,feature_num*4),
        
            Up(feature_num*8,feature_num*2,dtype),
            Up(feature_num*4,feature_num,dtype),
            Up(feature_num*2,feature_num,dtype),
        
            OutConv(feature_num, 3)
        )    

        new(removal, approximator, dtype)
    end


    #forward
    function (c::Denoise_net)(x) 

        o1 = c.removal.layers[1](x)  #inconv
        #@show size(x), size(o1)
        
        o2 = c.removal.layers[2](o1) #down1
        #@show size(o1), size(o2)
   
        o3 = c.removal.layers[3](o2) #down2
        #@show size(o2), size(o3)
   
        o4 = c.removal.layers[4](o3) #down3
        #@show size(o3), size(o4)

        o5 = c.removal.layers[5](o4,o3) #up1
        #@show size(o3), size(o4), size(o5)
    
        o6 = c.removal.layers[6](o5,o2) #up2
        #@show size(o5), size(o2), size(o6)
    
        o7 = c.removal.layers[7](o6,o1) #up3
        #@show size(o6), size(o1), size(o7)
    
        o8 = c.removal.layers[8](o7) #outconv
        #@show size(o7), size(o8)

        n_hat = x - o8
        n_tilde = c.approximator(n_hat)
        o9 = x - n_tilde

        return n_hat, n_tilde, o8, o9    
    end
end


mutable struct N2G 
    net
    iteration
    dtrn
    dtype

    function N2G(feature_num::Int, dtrn::TrainDataset, dtype)
        net = Denoise_net(feature_num, dtype)
        new(net,0,dtrn,dtype)
    end

    # indexing is adapted from https://github.com/HuangxingLin123/Noise2Grad_Pytorch_code/blob/main/models/denoise_model.py 
    function gradient(img)
        h_dev = (img[1:end-1,2:end,:,:] - img[1:end-1,1:end-1,:,:])
        v_dev = (img[2:end,1:end-1,:,:] - img[1:end-1,1:end-1,:,:])
        grad = (h_dev+v_dev) .* 0.5
        return grad
    end

    
    #train forward
    function (m::N2G)(item_id::Int)
    
        x,y = get_batch(m.dtrn,item_id)
        m.iteration += size(x)[end]
    
        n_hat, n_tilde, X_denoise1, X_denoise2 = m.net(x)
        n_grad = gradient(n_tilde)   

        noise3 = value(n_tilde)
        mask = convert(m.dtype ,rand(Binomial(1,0.5), size(n_tilde))) .* 2 .- 1
        
        x_s = noise3 .* mask .+ y
        
        x_s_1 = convert(m.dtype, max.(x_s, 0.0))
        x_s_2 = convert(m.dtype, min.(x_s_1, 1.0))

        #x_s = clamp.(x_s, 0.0, 1.0)
    
        _, _, x_s_denoise,_ = m.net(value(x_s_2)) 
        #_, _, x_s_denoise,_ = m.net(value(x_s)) 

        tau = (m.iteration ÷ 500) + 1
        if m.iteration % tau == 0
            loss_grad = mse(n_grad, value(gradient(x)))
        else 
            loss_grad = 0
        end

        loss_denoise = mse(x_s_denoise, y)

        loss = loss_grad + loss_denoise
        
        return loss
    end

    #test forward
    function (m::N2G)(x)
        _, _, x_denoise,_ = m.net(value(x)) 
        return value(x_denoise)
    end

end

