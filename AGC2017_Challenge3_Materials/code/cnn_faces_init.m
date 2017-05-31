function net = cnn_faces_init(pretrainedmodelName)
net = load(fullfile('..' , 'pre_trained', pretrainedmodelName));

f=1/100;

net.layers{end}={};
net.layers{end-1}={};

net.layers{end-1} = net.layers{end-2}; %%lower the layer of fc7 to insert a dropout layer before it
net.layers{end-2} = net.layers{end-3}; %%lower the layer of fc7 to insert a dropout layer before it

net.layers{end-3} = struct('type','dropout','rate',0.3);

net.layers{end} = struct('type','dropout','rate',0.3); %% the dropout between fc7 & fc8

net.layers = net.layers(1:end-2);
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(3,3,4096,80, 'single'), zeros(1, 80, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 1, ...
                           'name', 'fc8', ...
                            'dilate', 1) ;  %% new fc8
              
net.layers{end+1} = struct('type', 'softmaxloss') ; %% new softmaxloss

vl_simplenn_display(net, 'inputSize', [224 224 3 50])


