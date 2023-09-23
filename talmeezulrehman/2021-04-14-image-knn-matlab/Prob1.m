format compact; clear; close; clc;

%% load and visualize pepper image
I = imread('peppers.png');
imwrite(I,"peppers-original.png");
figure(1)
subplot(2,2,1);
imshow(I);
title('Original','Fontsize',16)

%% visualize compressed images for k = 5,20,75
i=1;

for k = [5, 20, 75] % for loop for different values of k
    i = i+1;
    X = compressImg(I,k); % complete this part by calling compressImg to get compressed image for each k
    imwrite(X,sprintf("peppers-compressed-%d.png",k))
    subplot(2,2,i);
    imshow(X);
    title(['Compressed, k = ', num2str(k)],'Fontsize',16)
end
