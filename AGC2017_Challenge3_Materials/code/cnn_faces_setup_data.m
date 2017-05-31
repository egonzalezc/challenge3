function [imdb] = cnn_faces_setup_data()

datadir = fullfile('..','data', 'faces_data' );
totalSamples = 303; 
image_size = [224 224]; 

imdb.images.data   = zeros(image_size(1), image_size(2), 3, totalSamples, 'single');
imdb.images.labels = zeros(1, totalSamples, 'single');
imdb.images.set = zeros(1, totalSamples, 'single');
imdb.meta.sets = {'train', 'test'};
imdb.meta.classes = {};
for c=1:80
    imdb.meta.classes{c}=num2str(c);
end

% imdb.meta.classes = {'1', '2', '5'};


images = zeros(224, 224, 3, totalSamples, 'single') ;
labels = zeros(totalSamples, 1) ;
set = ones(totalSamples, 1) ;
imDouble = zeros(224, 224, 3, totalSamples) ;

sample = 1 ;
for s = 1:length(imdb.meta.sets)
    for label = 1:length(imdb.meta.classes)
    %     path = fullfile( datadir, num2str(label), '/');
        path = fullfile( datadir, imdb.meta.sets{s}, imdb.meta.classes{label} , '/');

        folderName = strcat(path,'*.jp*');
        imagefiles = dir(folderName);
        nimages = length(imagefiles);
%         train = round(nimages*0.7);
%         saved = 0;
        for i=1:nimages
            filename = strcat(path, imagefiles(i).name);
            im = imread(filename);

            %Delete
            im2 = imresize(im, image_size) ;
            im2 = im2double(im2);
            %

            im = single(im);
            im = imresize(im, image_size) ;
            if size(im,3)==1
                im = cat(3, im, im, im);
                im2 = cat(3, im2, im2, im2);
            end

            images(:,:,:,sample) = im ;
    %         labels(sample) = label ;

            %Change
            imDouble(:,:,:,sample) = im2 ;
            labels(sample) = label ;
            %
%             if saved < train
%                 set(sample) = 1;
%             else
%                 set(sample) = 2;
%             end
            set(sample) = s;
%             saved = saved + 1;
            sample = sample + 1 ;
        end
    end
end

% Show some random example images
figure(2) ;
montage(imDouble(:,:,:,randperm(totalSamples, 100))) ;
title('Example images') ;

% Remove mean over whole dataset
images = bsxfun(@minus, images, mean(images, 4)) ;

% Store results in the imdb struct
imdb.images.data = images ;
imdb.images.labels = labels ;
imdb.images.set = set ;

% Mean to be set in the net
imdb.meta.avarage = mean(images, 4);
