
function minibatch_x(x, bs=2)
    data = Any[]

    println("num of instances ", length(x))
    num_of_instances = length(x)
    d, r = divrem(num_of_instances,bs)
    n = r > 0 ? d+1 : d

    img_size = size(x[1])
    println("img size ", img_size)

    #reshape (n, ) to (img_size..., n) 4 dims
    x = permutedims(reshape(vcat(x...), (length(x), length(x[1]))))
    
    for i in 1:n
        j = min(i*bs, num_of_instances)
        l = j - (i-1)*bs #num of instances in batch
        batch_x = reshape(x[:,(i-1)*bs+1:j], img_size[1:end]..., l)
        push!(data, batch_x)
    end
    
    return data
end

function minibatch_x_y(x, y, bs=2)
    data = Any[]

    println("num of instances ", length(x))
    num_of_instances = length(x)
    d, r = divrem(num_of_instances,bs)
    n = r > 0 ? d+1 : d

    img_size = size(x[1])
    println("img size ", img_size)

    #reshape (n, ) to (img_size..., n) 4 dims
    x = permutedims(reshape(vcat(x...), (length(x), length(x[1]))))
    y = permutedims(reshape(vcat(y...), (length(y), length(y[1]))))
    println(size(x), " ", size(y))
    
    for i in 1:n
        j = min(i*bs, num_of_instances)
        l = j - (i-1)*bs #num of instances in batch

        batch_x = reshape(x[:,(i-1)*bs+1:j], img_size[1:end]..., l)
        batch_y = reshape(y[:,(i-1)*bs+1:j], img_size[1:end]..., l)
        #println(size(batch_x), " ", size(batch_y))
        push!(data, (batch_x,batch_y))
    end
    
    return data
end

function read_img_names_in_dir(dir)

    files = String[]
    for f in readdir(dir)
        if f[end-3:end] == ".png"
            push!(files, f)
        end
    end

    return files    
end

function convert_to_3d(img)
    h,w = size(img)
    output = zeros((h,w,3))

    #RGB
    output[:,:,1] = Float32.(broadcast(red,img))
    output[:,:,2] = Float32.(broadcast(green,img))
    output[:,:,3] = Float32.(broadcast(blue,img))

    return output
end


function load_imgs(dir, files)
    return collect(convert_to_3d(load(string(dir ,"/",f))) for f in files)
end


function random_crop(img, patch_length)
    Random.seed!(0)
    img_size = size(img)
    r = rand(1:img_size[1]- (patch_length-1) ,1)[1]
    c = rand(1:img_size[2]- (patch_length-1) ,1)[1]
    return img[r:r+(patch_length-1), c:c+(patch_length-1),:]
end