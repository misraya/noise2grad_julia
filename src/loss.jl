using Images

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
        push!(batch_psnr, assess(custom_IQI, batch1[:,:,:,i], batch2[:,:,:,i])) 
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

