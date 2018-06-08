function psnr=compute_psnr(im1,im2)

imdff = double(im1) - double(im2);
imdff = imdff(:);

im3 = im2(:);
max_element = max(im3);
max_element = double(max_element);
% max_element = 255;

rmse = sqrt(mean(imdff.^2));
psnr = 20*log10(max_element/rmse);

end
