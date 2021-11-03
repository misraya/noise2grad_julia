include("layers.jl")
include("loss.jl")

using AutoGrad
using Distributions


struct Denoise_net 
    removal
    approximator


    function Denoise_net(feature_num::Int)
        approximator = Chain(
            Conv(3, 3, 1, 0, 1) #c_in, c_out, kernel_size, padding, stride
        )
                
        removal =  Chain(
            InConv(3,feature_num),
        
            Down(feature_num,feature_num*2),
            Down(feature_num*2,feature_num*4),
            Down(feature_num*4,feature_num*4),
        
            Up(feature_num*8,feature_num*2),
            Up(feature_num*4,feature_num),
            Up(feature_num*2,feature_num),
        
            OutConv(feature_num, 3)
        )    

        new(removal, approximator)
    end


    #forward
    function (c::Denoise_net)(x) 


        o1 = c.removal.layers[1](x)  #inconv
        o2 = c.removal.layers[2](o1) #down1
        o3 = c.removal.layers[3](o2) #down2
        o4 = c.removal.layers[4](o3) #down3

        o5 = c.removal.layers[5](o4,o3) #up1
        o6 = c.removal.layers[6](o5,o2) #up2
        o7 = c.removal.layers[7](o6,o1) #up3

        o8 = c.removal.layers[8](o7) #outconv

        n_hat = x - o8
        n_tilde = c.approximator(n_hat)
        o9 = x - n_tilde

        return n_hat, n_tilde, o8, o9    
    end
end


struct N2G 
    net
    optimizer

    function N2G(feature_num::Int)
        optimizer = Adam(;lr=0.0002, gclip=0, beta1=0.5, beta2=0.999, eps=1e-8)
        net = Denoise_net(feature_num)
        new(net, optimizer)
    end

    # indexing is adapted from https://github.com/HuangxingLin123/Noise2Grad_Pytorch_code/blob/main/models/denoise_model.py 
    function gradient(img)
        h_dev = img[1:end-1,2:end,:,:] - img[1:end-1,1:end-1,:,:] 
        v_dev = img[2:end,1:end-1,:,:] - img[1:end-1,1:end-1,:,:] 
        grad =(h_dev+v_dev) * 0.5
        return grad
    end

    
    #forward
    function (m::N2G)(x,y)
        n_hat, n_tilde, X_denoise1, X_denoise2 = m.net(x)
        n_grad = gradient(n_tilde)
        
        noise3 = n_tilde #detach?
        mask = rand(Binomial(1,0.5), size(n_tilde)) .* 2 .- 1
        x_s = noise3 .* mask .+ y
        x_s[x_s .> 1.0] .= 1.0
        x_s[x_s .< 0] .= 0
        _, _, x_s_denoise,_ = m.net(x_s) #detach?


        loss_grad = mse(n_grad, gradient(x))
        loss_denoise = mse(x_s_denoise, y)

        return loss_grad, loss_denoise, loss_grad+loss_denoise
    end


end

