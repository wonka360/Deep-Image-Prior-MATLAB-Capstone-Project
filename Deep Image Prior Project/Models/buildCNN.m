function net = buildCNN(n_lay,n_chan,ksize,imageSize,input_depth)

pd = floor(ksize/2);

layers = [
    imageInputLayer([imageSize imageSize input_depth], ...
        'Normalization','none', ...
        'Name','input')

    convolution2dLayer(ksize,n_chan, ...
        'Padding',pd, ...
        'WeightsInitializer','he', ...
        'Name','conv1')

    batchNormalizationLayer('Name','bn1')

    preluLayer('Name','prelu1')
];

% Hidden layers
for i = 1:n_lay

    layers = [
        layers

        convolution2dLayer(ksize,n_chan, ...
            'Padding',pd, ...
            'WeightsInitializer','he', ...
            'Name',['conv' num2str(i+1)])

        batchNormalizationLayer('Name',['bn' num2str(i+1)])

        preluLayer('Name',['prelu' num2str(i+1)])
    ];

end

% Reconstruction head
layers = [
    layers

    convolution2dLayer(1,3, ...          % 1x1 output conv (more stable)
        'Padding','same', ...
        'WeightsInitializer','he', ...
        'Name','conv_final')

    sigmoidLayer('Name','sigmoid')       % constrain output to [0,1]
];

net = dlnetwork(layers);

