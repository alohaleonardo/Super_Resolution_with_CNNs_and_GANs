file_path = './img_align_celeba/';
img_path_list = dir(strcat(file_path,'*.jpg'));
img_num = length(img_path_list);

up_scale = 4;
if img_num > 0
    for j = 1:50000
        image_name = img_path_list(j).name;
        
        im = imread(strcat(file_path, image_name));
%         im = rgb2ycbcr(im);
%         im = im(:, :, 1);

        im = imcrop(im,[17 37 143 143]);
        
        im_gnd = modcrop(im, up_scale);                  % label
        im_1 = imresize(im_gnd, 1/up_scale, 'bicubic');  % half the size        
        im_2 = imresize(im_1, up_scale, 'bicubic');      % after bicubic
%         im_gnd = single(im_gnd)/255;
%         save img
        save_dir_label = './4_3/ground_truth_4_3/';
        save_dir_smallsize = './4_3/small_size_4_3/';
        save_dir_input = './4_3/input_4_3/';
         
%         save_name = fullfile(save_dir_label, image_name);
%         save_name1 = fullfile(save_dir_smallsize, image_name);
        save_name2 = fullfile(save_dir_input, image_name);
         
%         imwrite(im_gnd, save_name);
%         imwrite(im_1, save_name1);
        imwrite(im_2, save_name2)
    end
end
