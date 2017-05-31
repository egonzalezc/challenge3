% Basic script for Face Recognition Challenge
% --------------------------------------------------------------------
% AGC Challenge 2017 
% Universitat Pompeu Fabra
%

% Load challenge Training data
load AGC17_Challenge3_Training.mat

% Provide the path to the input images, for example 
% 'C:\AGC_Challenge_2017\images\'
imgPath = '../data/all_data_set/';

% Initialize results structure
AutoRecognSTR = struct();

% Initialize timer accumulator
total_time = 0;

% Load Face Recognition model from myFaceRecognitionModel.mat
% This file must contain a single variable called 'myFRModel'
% with any information or parameter needed by the
% function 'MyFaceRecognFunction' (see below)
% load myFaceRecognitionModel
run(fullfile('../../', 'matconvnet-1.0-beta24', 'matlab', 'vl_setupnn.m')) ;

my_FRmodel = load(fullfile('..', 'output', 'models', 'net-30-faces-May.mat'));
my_FRmodel.layers{end}.type = 'softmax';
my_FRmodel = vl_simplenn_tidy(my_FRmodel) ;

% Process all images in the Training set
for j = 1 : length( AGC17_Challenge3_TRAINING )
    A = imread( sprintf('%s%s',...
        imgPath, AGC17_Challenge3_TRAINING(j).imageName ));    
    
    %try
        % Timer on
        tic;
                
        % ###############################################################
        % Your face recognition function goes here. It must accept 2 input
        % parameters:
        %
        % 1. the input image A
        % 2. the recognition model
        %
        % and it must return a single integer number as output, which can
        % be:
        % a) A number between 1 and 80 (representing one of the identities
        % in the trainig set)
        % b) A "-1" indicating that none of the 80 users is present in the
        % input image
        %
        faces = face_detection(A);
        autom_id=-1;
        for f=1:size(faces, 1)
            faceImg = A(faces(f,2):faces(f,4), faces(f,1):faces(f,3),:);
            autom_id = my_face_recognition_function(faceImg, my_FRmodel);        
        end
        % ###############################################################
        
        % Update total time
        tt = toc;
        total_time = total_time + tt;
        
    %catch
        % % If the face recognition function fails, it will be assumed that no
        % % user was detected for this input image
        % autom_id = -1;
    %end

    % Store the detection(s) in the resulst structure
    AutoRecognSTR(j).id = autom_id;
end
   
% Compute detection score
FR_score = CHALL_AGC17_ComputeRecognScores(...
    AutoRecognSTR, AGC17_Challenge3_TRAINING);

% Display summary of results
fprintf(1, '\nF1-score: %.2f%% \t Total time: %dm %ds\n', ...
    100 * FR_score, int16( total_time/60),...
    int16(mod( total_time, 60)) );




