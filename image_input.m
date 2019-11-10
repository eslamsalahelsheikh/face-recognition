function y = image_input(image_name, N)
%This function takes the image name, converts it to gray and resize it to NXN. 
y = imread(image_name);
y = rgb2gray(y);
y = imresize(y,[N N]);
end

