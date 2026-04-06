function net = InpaintUnet3(imageSize, numOutputChannels)


nCd = [16, 32, 64, 128, 128];   
nCu = [16, 32, 64, 128, 128];   


lgraph = layerGraph();
lgraph = addLayers(lgraph, ...
    imageInputLayer(imageSize, 'Name','input', 'Normalization','none'));

% --- Encoder ---
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

% --- Decoder ---
lgraph = addLayers(lgraph, [
    resize2dLayer('Scale',2,'Method','bilinear','Name','d5_up')
    batchNormalizationLayer('Name','d5_bn0')          
    convolution2dLayer(3,128,'Padding','same','Name','d5_c1')
    batchNormalizationLayer('Name','d5_b1')
    leakyReluLayer(0.2,'Name','d5_a1')
    convolution2dLayer(1,128,'Padding','same','Name','d5_c2')  
    batchNormalizationLayer('Name','d5_b2')
    leakyReluLayer(0.2,'Name','d5_a2')
]);
lgraph = connectLayers(lgraph,'e5_pool','d5_up');


lgraph = addLayers(lgraph, [
    resize2dLayer('Scale',2,'Method','bilinear','Name','d4_up')
    batchNormalizationLayer('Name','d4_bn0')          
    convolution2dLayer(3,128,'Padding','same','Name','d4_c1')
    batchNormalizationLayer('Name','d4_b1')
    leakyReluLayer(0.2,'Name','d4_a1')
    convolution2dLayer(1,128,'Padding','same','Name','d4_c2')  
    batchNormalizationLayer('Name','d4_b2')
    leakyReluLayer(0.2,'Name','d4_a2')
]);
lgraph = connectLayers(lgraph,'d5_a2','d4_up');


lgraph = addLayers(lgraph, [
    resize2dLayer('Scale',2,'Method','bilinear','Name','d3_up')
    batchNormalizationLayer('Name','d3_bn0')          
    convolution2dLayer(3,64,'Padding','same','Name','d3_c1')
    batchNormalizationLayer('Name','d3_b1')
    leakyReluLayer(0.2,'Name','d3_a1')
    convolution2dLayer(1,64,'Padding','same','Name','d3_c2')   
    batchNormalizationLayer('Name','d3_b2')
    leakyReluLayer(0.2,'Name','d3_a2')
]);
lgraph = connectLayers(lgraph,'d4_a2','d3_up');


lgraph = addLayers(lgraph, [
    resize2dLayer('Scale',2,'Method','bilinear','Name','d2_up')
    batchNormalizationLayer('Name','d2_bn0')          
    convolution2dLayer(3,32,'Padding','same','Name','d2_c1')
    batchNormalizationLayer('Name','d2_b1')
    leakyReluLayer(0.2,'Name','d2_a1')
    convolution2dLayer(1,32,'Padding','same','Name','d2_c2')   
    batchNormalizationLayer('Name','d2_b2')
    leakyReluLayer(0.2,'Name','d2_a2')
]);
lgraph = connectLayers(lgraph,'d3_a2','d2_up');


lgraph = addLayers(lgraph, [
    resize2dLayer('Scale',2,'Method','bilinear','Name','d1_up')
    batchNormalizationLayer('Name','d1_bn0')          
    convolution2dLayer(3,16,'Padding','same','Name','d1_c1')
    batchNormalizationLayer('Name','d1_b1')
    leakyReluLayer(0.2,'Name','d1_a1')
    convolution2dLayer(1,16,'Padding','same','Name','d1_c2')   
    batchNormalizationLayer('Name','d1_b2')
    leakyReluLayer(0.2,'Name','d1_a2')
]);
lgraph = connectLayers(lgraph,'d2_a2','d1_up');

% --- Final Layer ---
lgraph = addLayers(lgraph, [
    convolution2dLayer(1,numOutputChannels,'Padding','same','Name','final_conv')
    sigmoidLayer('Name','final_sigmoid')
]);
lgraph = connectLayers(lgraph,'d1_a2','final_conv');

net = dlnetwork(lgraph);
end