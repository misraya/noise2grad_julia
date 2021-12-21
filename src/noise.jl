using Noise 

function additive_gaussian_noise(img; fixed=true, std=25)
    if fixed 
        return add_gauss(img, std^2/255^2, 0.0)
    else        
        noise_level = rand(1:50)
        return add_gauss(img, noise_level^2/255^2, 0.0)
    end     
end

function poisson_noise(img)
    lambda = rand(1:50)
    return poisson(img, lambda)
end

function speckle_noise(img)
    # rand generates between [0,1)
    # 1-rand is between (0,1]
    # (1-rand)*0.02 is between (0,0.02]
    v = (1 - rand()) * 0.02
    s = size(img)

    # result = x + x * n
    # where n is a uniform noise with mean 0 and var v
    return img + img .* ((rand(size(img)...) .- 0.5 ) * sqrt(v))

end