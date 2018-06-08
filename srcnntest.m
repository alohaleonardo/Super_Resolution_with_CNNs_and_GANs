file_path = './';
img_path_list = dir(strcat(file_path,'*.jpg'));
img_num = length(img_path_list);

up_scale = 2;
if img_num > 0
    for j = 1:img_num
        image_name = img_path_list(j).name;
        image_name1 = image_name;
        
        image = imread(strcat(file_path, image_name));

        im = imread(strcat(file_path, image_name));
        im = imcrop(im,[17 37 143 143]);
        
        im_gnd = modcrop(im, up_scale);
        im_gnd = single(im_gnd)/255;
        
        im_b = imresize(im_gnd, 1/up_scale, 'bicubic');
        im_b = imresize(im_b, up_scale, 'bicubic'); % blur
        
        im_h = SRCNN(model, im_b);  % high
        % save img
         save_dir = './output2/';
         save_dir_blur = './blur_test/';
         
         save_name = fullfile(save_dir, image_name);
         save_name1 = fullfile(save_dir_blur, image_name1);
         
         imwrite(im_gnd, save_name);
         imwrite(im_h, save_name1);

    end
end