src =  '../data/faces_data/all_no_cropped';
dest = '../data/faces_data/';
originalSrc = '../data/only_original_cropped_faces';
% Initialize timer accumulator
total_time = 0;

folder_indexes = 1:80;
nimages_subject = zeros(size(folder_indexes,2),3);
not_found = 0;
found = 0;

% sources = {fullfile(dest, 'test_no_cropped'), fullfile(dest, 'train_no_cropped')};
% joinFolders(sources, src, folder_indexes)

for i_f = 1 : size(folder_indexes, 2)
    j = folder_indexes(i_f);
    folderName = fullfile(src, num2str(j),'*.j*');
    imagefiles = dir(folderName); 
    numImgs = length(imagefiles);

    training = round(numImgs*0.8);
    nimages_subject(i_f,:)= [j, training, numImgs- training];
    
    saved = copyTrainingImgs(fullfile(originalSrc, num2str(j)),  dest, j);
    
    for i=1:numImgs
        tic;
        
        if saved<training
            output1 = fullfile(dest, 'train' ,num2str(j));
        else
            output1 = fullfile(dest, 'test' ,num2str(j));
        end
        
        if exist(fullfile(output1, imagefiles(i).name), 'file') ~= 2
            A = imread(fullfile(imagefiles(i).folder, '/', imagefiles(i).name));
            det_faces = MyFaceDetectionFunction( A); 
            tt = toc;
            total_time = total_time + tt;

            if exist(output1 , 'dir') ==0 
                mkdir(output1);
            end

            for k=1 : size(det_faces, 1)
                coord = det_faces(k, :);
                croped_image = A(coord(2):coord(4), coord(1):coord(3), :);
                imwrite(croped_image, fullfile(output1, imagefiles(i).name));
                found = found+1;
            end
            if size(det_faces,1) == 0
                if exist(fullfile(output1 ,'not_found'), 'dir') ==0 
                    mkdir(fullfile(output1 ,'not_found'));
                end
                imwrite(A, fullfile(output1, 'not_found', imagefiles(i).name));
                not_found = not_found + 1;
            else
                saved = saved + 1;
            end    
            
        end
    end
    nimages_subject(i_f,:)= [j, training, saved- training];
end

disp(sprintf('Found = %d Not found = %d Time = %d ', found, not_found, total_time));


function [coordinates] =  MyFaceDetectionFunction(image)
    coordinates = [];
    actual_faces = 0;
    models = {'FrontalFaceCART', 'FrontalFaceLBP', 'ProfileFace'};
    m=1;
    while actual_faces==0 && m<=size(models,2)
        faceDetector = vision.CascadeObjectDetector(models{m});
        faceBbox = step(faceDetector, image);
        for i=1:size(faceBbox,1)
            noseDetector = vision.CascadeObjectDetector('nose');                 
            noseDetector.UseROI = true;
            noseBox = step(noseDetector, image, faceBbox(i,:));
            
            x = faceBbox(i,1);
            y = faceBbox(i,2);
            width = x + faceBbox(i,3);
            height = y + faceBbox(i,4);

            if size(noseBox, 1) > 0
                actual_faces = actual_faces + 1;
                coordinates(actual_faces,:) = [x, y, width, height];
            else
                eyeDetector = vision.CascadeObjectDetector('RightEye');
                eyeDetector.UseROI = true;
                eyeBbox = step(eyeDetector, image,  faceBbox(i,:));
                if size(eyeBbox, 1) > 0
                    actual_faces = actual_faces + 1;
                    coordinates(actual_faces,:) = [x, y, width, height];
                end
            end
        end
        m=m+1;
    end
end

function count = copyTrainingImgs(src, dest, subj)
    folderName = fullfile(src,'*.j*');
    imagefiles = dir(folderName); 
    nimages = length(imagefiles);
    count=0;
    for i=1:nimages
        A = imread(fullfile(imagefiles(i).folder, '/', imagefiles(i).name));
        output1 = fullfile(dest, 'train' ,num2str(subj));
        if exist(fullfile(output1), 'dir') ==0 
            mkdir(fullfile(output1 ));
        end
        imwrite(A, fullfile(output1, imagefiles(i).name));
        count = count + 1;
    end

end

function joinFolders(sources, dest, folder_indexes)
    for subj = 1 : size(folder_indexes, 2)
        for src=1:  length(sources)
            folderName = fullfile(sources{src}, num2str(subj),'*.j*');
            imagefiles = dir(folderName); 
            nimages = length(imagefiles);
            for i=1:nimages
                A = imread(fullfile(imagefiles(i).folder, imagefiles(i).name));
                if exist(fullfile(dest ,num2str(subj)), 'dir') ==0 
                    mkdir(fullfile(dest ,num2str(subj)));
                end
                imwrite(A, fullfile(dest, num2str(subj), imagefiles(i).name))
            end
        end
    end
    
end 