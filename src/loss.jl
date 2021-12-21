using Images
using Statistics

function mean_squared_error(image0, image1)    
    @assert size(image0) == size(image1)
    return Statistics.mean((image0 - image1).^2) 
end

# 235-16 range for YCbCr. 
# source: https://en.wikipedia.org/wiki/YCbCr#:~:text=called%20chroma%20subsampling.-,YCbCr,-%5Bedit%5D
function calc_psnr(image_true, image_test)
    err = mean_squared_error(image_true, image_test)
    return 10 * log10(((235.00-16.00)^2) / err)
end


function ssim(batch1, batch2)

    custom_IQI = SSIM(
        # can change default parameters
        )
    
    batch_ssim = []
    for i in 1:size(batch1)[end]
        push!(batch_ssim, assess(custom_IQI, batch1[:,:,:,i], batch2[:,:,:,i])) 
    end
    return batch_ssim
end


function psnr(batch1, batch2)

    custom_IQI = PSNR(
        # can change default parameters
        )
    
    batch_psnr = []
    for i in 1:size(batch1)[end]
        #push!(batch_psnr, assess(custom_IQI, batch1[:,:,:,i], batch2[:,:,:,i])) #old version
        push!(batch_psnr, calc_psnr(batch1[:,:,:,i], batch2[:,:,:,i])) 
    end
    return batch_psnr
end


function mse(batch1, batch2)
    n = size(batch1)[end]
    return sum(abs2.(batch1.-batch2)) / (2*n)
end


function l1_loss(batch1, batch2)
    return sum(abs.(batch1 .- batch2))
end

