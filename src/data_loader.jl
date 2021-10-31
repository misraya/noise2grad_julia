include("helper.jl")

struct TrainDataset
    data 

    function TrainDataset(x_path, y_path, batch_size) 
        x_imgs = load_imgs(x_path, read_img_names_in_dir(x_path)[1:100])
        y_imgs = load_imgs(y_path, read_img_names_in_dir(y_path)[1:100])
        
        n = size(y_imgs)[1]
        indices = randperm(n)

        for (i,x) in enumerate(x_imgs)

            x_imgs[i] = random_crop(x,128)
            y_imgs[indices[i]] = random_crop(y_imgs[indices[i]],128)

            #random flip / other preprocessing?

        end

        data = _minibatch(x_imgs,y_imgs,batch_size) 
        new(data)
    end

end