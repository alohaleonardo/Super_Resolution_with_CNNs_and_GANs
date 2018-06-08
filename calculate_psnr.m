file_path_blur = './testImageOutput (2)/';  % generate
file_path_label = './4_3_2/ground_truth_4_3/';  % ground_truth
img_path_list_label = dir(strcat(file_path_label,'*.jpg'));  
img_path_list_blur = dir(strcat(file_path_blur,'*.jpg'));
img_num = length(img_path_list_blur);
up_scale = 4;

psnr_total = 0;
psnr_total_blur = 0;

if img_num > 0
    for j = 1:img_num
        image_name = img_path_list_blur(j).name;
        
        im1 = imread(strcat(file_path_label, image_name));  % label
        im2 = imread(strcat(file_path_blur, image_name));   % generate
        
        % here we get the blur
        im = imresize(im1, 1/up_scale, 'bicubic');   % 1/4 the size        
        im = imresize(im, up_scale, 'bicubic');      % after bicubic
        
        psnr = compute_psnr(im1,im2);     % label-generate
        psnr_between_bicubic_and_label = compute_psnr(im1, im);  % bicubic-label
        
        psnr_total = psnr_total + psnr;    % label-generate
        psnr_total_blur = psnr_total_blur + psnr_between_bicubic_and_label; % bicubic-label
        
        if j == 1
            imwrite(im,image_name);
        end
    end
end

display(psnr_total / img_num);  % label-generate
display(psnr_total_blur / img_num); % bicubic-label

% a = [1 2 3 4 5];
% b = [1 3 3 4 5];
% 
% compute_psnr(a,b)