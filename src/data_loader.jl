include("helper.jl")
using NaturalSort

struct TrainDataset
    paths 
    length
    bs
    dtype

    function TrainDataset(x_path, y_path, batch_size, dtype; limit=false, max_samples=100)

        if limit #load only max_sample instances, used for fast debugging 
            x_paths = sort(readdir(x_path; join=true,sort=false,), lt=natural)[1:max_samples]
            y_paths = sort(readdir(y_path; join=true,sort=false,), lt=natural)[1:max_samples]
        else #load all
            x_paths = sort(readdir(x_path; join=true,sort=false,), lt=natural)
            y_paths = sort(readdir(y_path; join=true,sort=false,), lt=natural)
        end
        
        #n = size(y_imgs)[1]
        #indices = randperm(n)
        #y_img_names = y_img_names[indices]
    
        paths =  (x_paths, y_paths)
        new(paths, length(x_paths), batch_size, dtype)
    
    end

end


struct TestDataset
    data 
    length
    bs

    function TestDataset(x_path, batch_size, dtype) 
        x_paths = sort(readdir(x_path; join=true,sort=false,), lt=natural)
        x_imgs = convert.(Array{Float32}, load_imgs(x_paths)) 
        
        for (i,x) in enumerate(x_imgs)
            x_imgs[i] = random_crop(x,128)
            #random flip / other preprocessing?
        end
                
        x_imgs = reshape(hcat(x_imgs...), (size(x_imgs[1])..., size(x_imgs)...))
        x_batches = minibatch(x_imgs, batch_size, partial=true, xtype=dtype)
        new(x_batches, length(x_paths), batch_size)
    end

end

function get_item(dtrn::TrainDataset, id::Int)
    
    x = convert_to_3d(load(dtrn.paths[1][id]))
    rand_id = trunc.(Int,rand() .* 5000)
    y = convert_to_3d(load(dtrn.paths[2][rand_id]))

    x = random_crop(x,128)
    y = random_crop(y,128)
    
    x = reshape(x, (size(x)..., 1))
    y = reshape(y, (size(y)..., 1))
    
    return convert(dtrn.dtype,x),convert(dtrn.dtype,y)
end
    

function get_batch(dtrn::TrainDataset, id::Int)
    
    ids = collect(id: min(dtrn.length, id+dtrn.bs-1))
    
    x_batch = []
    y_batch = []
    
    for id in ids
        x = convert_to_3d(load(dtrn.paths[1][id]))
        rand_id = trunc.(Int,rand() .* 5000)
        y = convert_to_3d(load(dtrn.paths[2][rand_id]))

        x = random_crop(x,128)
        y = random_crop(y,128)
        
        #random flip / other preprocessing?

        push!(x_batch, x)
        push!(y_batch, y)
    end
        
    x_batch = reshape(hcat(x_batch...), (size(x_batch[1])..., size(x_batch)...))
    y_batch = reshape(hcat(y_batch...), (size(y_batch[1])..., size(y_batch)...))    
    
    return convert(dtrn.dtype,x_batch),convert(dtrn.dtype,y_batch)
end
    


