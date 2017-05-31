function best = my_face_recognition_function(im, net)
im_ = single(im) ;
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
res = vl_simplenn(net, im_) ;

scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
if bestScore < 0.5
    best =-1;
end