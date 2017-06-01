function best = test_model()
net = load(fullfile('..', 'output','models', 'net-1-June-4.mat'));
net.layers{end}.type = 'softmax';
net = vl_simplenn_tidy(net) ;% run(fullfile(HOME_PATH, 'matconvnet-1.0-beta24' ,'matlab', 'vl_setupnn.m')) ;
% for c=1:80
%     classes{c}=num2str(c);
% end

classes={'9', '25', '28', '30'};

good = 0;
bad = 0;
total = 0;
for class=1:length(classes)
   folderName = fullfile('..','data', 'only_original_cropped_faces', classes{class}, '/');
   imagefiles = dir(strcat(folderName,'*.jp*'));
   nimages = length(imagefiles);
   for i=1:nimages
        filename = strcat(folderName, imagefiles(i).name);
        im = imread(filename);
        im_ = single(im) ;
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
        res = vl_simplenn(net, im_) ;

        scores = squeeze(gather(res(end).x)) ;
        [bestScore, best] = max(scores) ;
        if best == class && bestScore >0.5
            good = good + 1;
        else
            bad = bad + 1;
            disp(sprintf('Predicted %d,  True %d', best, class));
        end
        total = total + 1;
   end
end
disp(sprintf('Total %d, Good %d, Bad %d', total, good, bad));
% im_ = single(im) ; % note: 255 range
% im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
% im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
% res = vl_simplenn(net, im_) ;
% 
% % Show the classification result.
% scores = squeeze(gather(res(end).x)) ;
% [bestScore, best] = max(scores) ;
% if bestScore < 0.5
%     best =-1;
% else
%     figure(1) ; clf ; imagesc(im) ; axis equal off ;
%     title(sprintf('%s (%d), score %.3f',...
%               classes{best}, best, bestScore), ...
%       'Interpreter', 'none') ;
% 
% end