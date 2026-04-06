function net = InpaintUnet3(imageSize, numOutputChannels)
% Hardcoded equivalent of:
%   net = skip(input_depth, img_np.shape[0],
%              num_channels_down = [16, 32, 64, 128, 128],
%              num_channels_up   = [16, 32, 64, 128, 128],
%              num_channels_skip = [0, 0, 0, 0, 0],
%              filter_size_down=3, filter_size_up=3, filter_skip_size=1,
%              upsample_mode='bilinear', downsample_mode='avg',
%              need_sigmoid=True, pad='zero')
%
%  imageSize         = [H W input_depth]  e.g. [256 256 32]
%  numOutputChannels = img_np.shape[0]    e.g. 1 or 3

nCd = [16, 32, 64, 128, 128];   % num_channels_down
nCu = [16, 32, 64, 128, 128];   % num_channels_up
% num_channels_skip = [0,0,0,0,0] -> no skip branches, no Concat layers

lgraph = layerGraph();
lgraph = addLayers(lgraph, ...
    imageInputLayer(imageSize, 'Name','input', 'Normalization','none'));

% ── Encoder ──────────────────────────────────────────────────────────────────
% Each scale: conv3(stride1)->bn->lrelu -> conv3->bn->lrelu -> avgpool(2)
% (downsample_mode='avg' => stride-1 conv + separate avgpool)

% Scale 1  (input_depth -> 16)
lgraph = addLayers(lgraph, [
    convolution2dLayer(3,16,'Padding','same','Stride',1,'Name','e1_c1')
    batchNormalizationLayer('Name','e1_b1')
    leakyReluLayer(0.2,'Name','e1_a1')
    convolution2dLayer(3,16,'Padding','same','Stride',1,'Name','e1_c2')
    batchNormalizationLayer('Name','e1_b2')
    leakyReluLayer(0.2,'Name','e1_a2')
    averagePooling2dLayer(2,'Stride',2,'Name','e1_pool')
]);
lgraph = connectLayers(lgraph,'input','e1_c1');

% Scale 2  (16 -> 32)
lgraph = addLayers(lgraph, [
    convolution2dLayer(3,32,'Padding','same','Stride',1,'Name','e2_c1')
    batchNormalizationLayer('Name','e2_b1')
    leakyReluLayer(0.2,'Name','e2_a1')
    convolution2dLayer(3,32,'Padding','same','Stride',1,'Name','e2_c2')
    batchNormalizationLayer('Name','e2_b2')
    leakyReluLayer(0.2,'Name','e2_a2')
    averagePooling2dLayer(2,'Stride',2,'Name','e2_pool')
]);
lgraph = connectLayers(lgraph,'e1_pool','e2_c1');

% Scale 3  (32 -> 64)
lgraph = addLayers(lgraph, [
    convolution2dLayer(3,64,'Padding','same','Stride',1,'Name','e3_c1')
    batchNormalizationLayer('Name','e3_b1')
    leakyReluLayer(0.2,'Name','e3_a1')
    convolution2dLayer(3,64,'Padding','same','Stride',1,'Name','e3_c2')
    batchNormalizationLayer('Name','e3_b2')
    leakyReluLayer(0.2,'Name','e3_a2')
    averagePooling2dLayer(2,'Stride',2,'Name','e3_pool')
]);
lgraph = connectLayers(lgraph,'e2_pool','e3_c1');

% Scale 4  (64 -> 128)
lgraph = addLayers(lgraph, [
    convolution2dLayer(3,128,'Padding','same','Stride',1,'Name','e4_c1')
    batchNormalizationLayer('Name','e4_b1')
    leakyReluLayer(0.2,'Name','e4_a1')
    convolution2dLayer(3,128,'Padding','same','Stride',1,'Name','e4_c2')
    batchNormalizationLayer('Name','e4_b2')
    leakyReluLayer(0.2,'Name','e4_a2')
    averagePooling2dLayer(2,'Stride',2,'Name','e4_pool')
]);
lgraph = connectLayers(lgraph,'e3_pool','e4_c1');

% Scale 5 — bottleneck  (128 -> 128)
lgraph = addLayers(lgraph, [
    convolution2dLayer(3,128,'Padding','same','Stride',1,'Name','e5_c1')
    batchNormalizationLayer('Name','e5_b1')
    leakyReluLayer(0.2,'Name','e5_a1')
    convolution2dLayer(3,128,'Padding','same','Stride',1,'Name','e5_c2')
    batchNormalizationLayer('Name','e5_b2')
    leakyReluLayer(0.2,'Name','e5_a2')
    averagePooling2dLayer(2,'Stride',2,'Name','e5_pool')
]);
lgraph = connectLayers(lgraph,'e4_pool','e5_c1');

% ── Decoder ──────────────────────────────────────────────────────────────────
% skip=0 everywhere so: upsample -> bn(k) -> conv3(k,nCu) -> bn -> lrelu
%                                         -> conv1(nCu,nCu) -> bn -> lrelu
% k at each scale = nCu of the scale below (or nCd at bottleneck)

% Scale 5 -> 4  (bottleneck up, k=128 -> nCu[4]=128)
lgraph = addLayers(lgraph, [
    resize2dLayer('Scale',2,'Method','bilinear','Name','d5_up')
    batchNormalizationLayer('Name','d5_bn0')          % bn(k=128)
    convolution2dLayer(3,128,'Padding','same','Name','d5_c1')
    batchNormalizationLayer('Name','d5_b1')
    leakyReluLayer(0.2,'Name','d5_a1')
    convolution2dLayer(1,128,'Padding','same','Name','d5_c2')  % 1x1
    batchNormalizationLayer('Name','d5_b2')
    leakyReluLayer(0.2,'Name','d5_a2')
]);
lgraph = connectLayers(lgraph,'e5_pool','d5_up');

% Scale 4 -> 3  (k=nCu[5]=128 -> nCu[3]=128)
lgraph = addLayers(lgraph, [
    resize2dLayer('Scale',2,'Method','bilinear','Name','d4_up')
    batchNormalizationLayer('Name','d4_bn0')          % bn(k=128)
    convolution2dLayer(3,128,'Padding','same','Name','d4_c1')
    batchNormalizationLayer('Name','d4_b1')
    leakyReluLayer(0.2,'Name','d4_a1')
    convolution2dLayer(1,128,'Padding','same','Name','d4_c2')  % 1x1
    batchNormalizationLayer('Name','d4_b2')
    leakyReluLayer(0.2,'Name','d4_a2')
]);
lgraph = connectLayers(lgraph,'d5_a2','d4_up');

% Scale 3 -> 2  (k=nCu[4]=128 -> nCu[2]=64)
lgraph = addLayers(lgraph, [
    resize2dLayer('Scale',2,'Method','bilinear','Name','d3_up')
    batchNormalizationLayer('Name','d3_bn0')          % bn(k=128)
    convolution2dLayer(3,64,'Padding','same','Name','d3_c1')
    batchNormalizationLayer('Name','d3_b1')
    leakyReluLayer(0.2,'Name','d3_a1')
    convolution2dLayer(1,64,'Padding','same','Name','d3_c2')   % 1x1
    batchNormalizationLayer('Name','d3_b2')
    leakyReluLayer(0.2,'Name','d3_a2')
]);
lgraph = connectLayers(lgraph,'d4_a2','d3_up');

% Scale 2 -> 1  (k=nCu[3]=64 -> nCu[1]=32)
lgraph = addLayers(lgraph, [
    resize2dLayer('Scale',2,'Method','bilinear','Name','d2_up')
    batchNormalizationLayer('Name','d2_bn0')          % bn(k=64)
    convolution2dLayer(3,32,'Padding','same','Name','d2_c1')
    batchNormalizationLayer('Name','d2_b1')
    leakyReluLayer(0.2,'Name','d2_a1')
    convolution2dLayer(1,32,'Padding','same','Name','d2_c2')   % 1x1
    batchNormalizationLayer('Name','d2_b2')
    leakyReluLayer(0.2,'Name','d2_a2')
]);
lgraph = connectLayers(lgraph,'d3_a2','d2_up');

% Scale 1 -> out  (k=nCu[2]=32 -> nCu[0]=16)
lgraph = addLayers(lgraph, [
    resize2dLayer('Scale',2,'Method','bilinear','Name','d1_up')
    batchNormalizationLayer('Name','d1_bn0')          % bn(k=32)
    convolution2dLayer(3,16,'Padding','same','Name','d1_c1')
    batchNormalizationLayer('Name','d1_b1')
    leakyReluLayer(0.2,'Name','d1_a1')
    convolution2dLayer(1,16,'Padding','same','Name','d1_c2')   % 1x1
    batchNormalizationLayer('Name','d1_b2')
    leakyReluLayer(0.2,'Name','d1_a2')
]);
lgraph = connectLayers(lgraph,'d2_a2','d1_up');

% ── Output head ──────────────────────────────────────────────────────────────
% conv1x1(16 -> numOutputChannels) -> sigmoid
lgraph = addLayers(lgraph, [
    convolution2dLayer(1,numOutputChannels,'Padding','same','Name','final_conv')
    sigmoidLayer('Name','final_sigmoid')
]);
lgraph = connectLayers(lgraph,'d1_a2','final_conv');

net = dlnetwork(lgraph);
end