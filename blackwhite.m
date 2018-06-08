file_path = './train_faces/';
img_path_list = dir(strcat(file_path,'*.jpg'));
img_num = length(img_path_list);
if img_num > 0
    for j = 1:img_num
        image_name = img_path_list(j).name;
        im = imread(strcat(file_path, image_name));
%         % resize
%         resize_img = imresize(image,[128 128]);
%         
        im = rgb2ycbcr(im);
        im = im(:, :, 1);
        
        % save img
         save_dir = './output/';
         save_name = fullfile(save_dir, img_path_list(j).name);
         imwrite(im, save_name);
    end
end