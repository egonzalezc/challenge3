function [net, info] = cnn_faces_main()

run(fullfile('../../', 'matconvnet-1.0-beta24', 'matlab', 'vl_setupnn.m')) ;

pretrainedmodelName = 'imagenet-vgg-s';
opts.expDir = fullfile('..', 'output') ;
imdbPath = fullfile(opts.expDir, 'imdb', 'imdb_faces.mat');
newModelPath = fullfile(opts.expDir, 'models', 'net-31-faces-May.mat');
opts.batchSize = 50 ;
opts.learningRate = 0.0001 ;
opts.numEpochs = 2 ;
opts.continue = false ;

%%  Preparing Data

if exist(imdbPath, 'file')
  imdb = load(imdbPath) ;
else
  imdb = cnn_faces_setup_data();
  mkdir(opts.expDir) ;
  save(imdbPath, '-struct', 'imdb') ;
end

%% Load pre trained model
net = cnn_faces_init(pretrainedmodelName);
net.meta.normalization.averageImage = imdb.meta.avarage;


%% Training
vl_simplenn_display(net, 'inputSize', [224 224 3 50])

[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts, 'train', find(imdb.images.set == 1),'val', find(imdb.images.set == 2)) ;

save(newModelPath, '-struct', 'net') ;
end

function [images, labels] = getBatch(imdb, batch)
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(batch) ;
end