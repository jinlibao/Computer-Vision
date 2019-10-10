% Template to help you calibrate a camera which is purely rotating and is not changing its
% focal length using a full search for K that minimizes reprojection error
% J. McInroy

clear; close all;
load '../data/pureRotPrexyCorrespondencePoints'; fn = '../data/prexy'; % load in the set of correspondence points

% display images
for i = 1:6
    imgstr1 = strcat(fn,int2str(i),'.jpg');
    I1 = imread(imgstr1,'jpg');
    % info  =  imfinfo(imgstr1); % get Exif info about image
    I2 = imread(strcat(fn,int2str(i+1),'.jpg'),'jpg');
    figure(1); imshow(I1); % display the images
    figure(2); imshow(I2)
    xtemp = x1pMat(:,:,i); % load in correspondence points in first image (pixel coordinates)
    % find first column of xtemp that is all zeros.  It signifies the end of
    % corresondence points collected for that pair of images.
    n = 1;
    while norm(xtemp(:,n))>0 && n<100 % there are at most 100 correspondence points per image pair
        n = n+1;
    end
    n = n-1;
    x1pmat = x1pMat(:,1:n,i); % extract pixel correspondence points in first image
    x2pmat = x2pMat(:,1:n,i); % extract pixel correspondence points in second image
    pause
end

kinit = randn(5,1)*2000; % initial K randomly chosen.  Put the entries of K into a single vector
% now add 5D search for K, using previous K as the starting point
% you'll have to develop your own algorithm to do this.
%     kvec  =  fminsearch(@(k) ReproCostPureRot(k,x1pMat,x2pMat ),kinit);
kvec = kinit; % use random choice for calibration matrix
Kbest = [kvec(1), kvec(2), kvec(3); 0, kvec(4), kvec(5); 0, 0, 1];

% Here is an example of how you can display the image and correspondence points
% create some fake reprojections so you can see how to do plots
% you'll need to implement the calculations for the actual reprojections
x1pReproMat = x1pmat; % perfect reprojections (fake)
x2pReproMat = x2pmat;
K = Kbest;
invK = inv(K);

jj = 1; % this determines which pair of images will be displayed
I1 = imread(strcat(fn,int2str(jj),'.jpg'),'jpg');
I2 = imread(strcat(fn,int2str(jj+1),'.jpg'),'jpg');

figure(21); hold on
image(I1);
plot(x1pmat(1,:)+j*x1pmat(2,:),'g*')
plot(x1pReproMat(1,:)+j*x1pReproMat(2,:),'bo')
legend('Image Points','Reprojections')
title('Image 1')

figure(22); hold on
image(I2);
plot(x2pmat(1,:)+j*x2pmat(2,:),'g*')
plot(x2pReproMat(1,:)+j*x2pReproMat(2,:),'bo')
legend('Image Points','Reprojections')
title('Image 2')
