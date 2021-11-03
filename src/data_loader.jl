include("helper.jl")

struct TrainDataset
    data 
    length

    function TrainDataset(x_path, y_path, batch_size; limit=false, max_samples=100)

        if limit #load only 100 instances, used for fast debugging 
            x_imgs = load_imgs(x_path, read_img_names_in_dir(x_path)[1:max_samples])
            y_imgs = load_imgs(y_path, read_img_names_in_dir(y_path)[1:max_samples])
        else #load all
            x_imgs = load_imgs(x_path, read_img_names_in_dir(x_path))
            y_imgs = load_imgs(y_path, read_img_names_in_dir(y_path))
        end
        
        n = size(y_imgs)[1]
        indices = randperm(n)

        for (i,x) in enumerate(x_imgs)

            x_imgs[i] = random_crop(x,128)
            y_imgs[indices[i]] = random_crop(y_imgs[indices[i]],128)

            #random flip / other preprocessing?

        end

        data = minibatch_x_y(x_imgs,y_imgs,batch_size) 
        new(data, size(x_imgs)[1])
    end

end


struct TestDataset
    data 
    length

    function TestDataset(x_path, batch_size) 
        x_imgs = load_imgs(x_path, read_img_names_in_dir(x_path))
        

        for (i,x) in enumerate(x_imgs)

            x_imgs[i] = random_crop(x,128)

            #random flip / other preprocessing?

        end

        data = minibatch_x(x_imgs,batch_size) 
        new(data, size(x_imgs)[1])
    end

end