function net = DipUnet3(imageSize, numClasses)

down_channels = [8, 16, 32, 64, 128];
up_channels   = [8, 16, 32, 64, 128];
numScales     = 5;

deepNet = dlnetwork;

% --- Input ---
inputL = imageInputLayer(imageSize, 'Name', 'input', 'Normalization', 'none');
deepNet = addLayers(deepNet, inputL);
currentInput = 'input';

% --- Encoder Path ---

for i = 1:numScales
    prefix = sprintf('enc%d_', i);

    encoder_block = [
        convolution2dLayer(3, down_channels(i), 'Padding', 'same', 'Name', [prefix 'conv1'])
        batchNormalizationLayer('Name', [prefix 'bn1'])
        leakyReluLayer(0.2, 'Name', [prefix 'lrelu1'])
        convolution2dLayer(3, down_channels(i), 'Padding', 'same', 'Name', [prefix 'conv2'])
        batchNormalizationLayer('Name', [prefix 'bn2'])
        leakyReluLayer(0.2, 'Name', [prefix 'lrelu2'])
        maxPooling2dLayer(2, 'Stride', 2, 'Name', [prefix 'pool'])   
    ];

    deepNet = addLayers(deepNet, encoder_block);
    deepNet = connectLayers(deepNet, currentInput, [prefix 'conv1']);
    currentInput = [prefix 'pool'];
end

% --- Decoder Path ---
for i = 0:numScales-1
    j          = numScales - i;
    up_prefix  = sprintf('dec%d_', j);

    if j == 5 || j == 4
        skip_prefix = sprintf('skip%d_', j);

        dec_upsample = resize2dLayer('Scale', 2, 'Method', 'bilinear', ...
                                     'Name', [up_prefix 'upsample']);

        
        skip_block = [
            convolution2dLayer(1, 4, 'Padding', 'same', 'Name', [skip_prefix 'conv1'])
            batchNormalizationLayer('Name', [skip_prefix 'bn1'])
            leakyReluLayer(0.2, 'Name', [skip_prefix 'lrelu1'])     
        ];

        batchNorm = batchNormalizationLayer('Name', [up_prefix 'bnorm']);

        decoder_block = [
            concatenationLayer(3, 2, 'Name', [up_prefix 'concat'])
            convolution2dLayer(3, down_channels(j), 'Padding', 'same', 'Name', [up_prefix 'conv1'])
            batchNormalizationLayer('Name', [up_prefix 'bn1'])
            leakyReluLayer(0.2, 'Name', [up_prefix 'lrelu1'])
            convolution2dLayer(3, down_channels(j), 'Padding', 'same', 'Name', [up_prefix 'conv2'])
            batchNormalizationLayer('Name', [up_prefix 'bn2'])
            leakyReluLayer(0.2, 'Name', [up_prefix 'lrelu2'])
        ];

        deepNet = addLayers(deepNet, dec_upsample);
        deepNet = addLayers(deepNet, skip_block);
        deepNet = addLayers(deepNet, decoder_block);
        deepNet = addLayers(deepNet, batchNorm);

        deepNet = connectLayers(deepNet, currentInput,          [up_prefix 'upsample']);
        deepNet = connectLayers(deepNet, [up_prefix 'upsample'],[up_prefix 'bnorm']);

        skipSource = sprintf('enc%d_lrelu2', j);
        deepNet = connectLayers(deepNet, skipSource,               [skip_prefix 'conv1']);
        
        deepNet = connectLayers(deepNet, [skip_prefix 'lrelu1'],   [up_prefix 'concat/in1']);
        deepNet = connectLayers(deepNet, [up_prefix 'bnorm'],      [up_prefix 'concat/in2']);

        currentInput = [up_prefix 'lrelu2'];

    else
        
        decoder_block = [
            resize2dLayer('Scale', 2, 'Method', 'bilinear', 'Name', [up_prefix 'upsample'])
            batchNormalizationLayer('Name', [up_prefix 'bn1'])
            convolution2dLayer(3, up_channels(j), 'Padding', 'same', 'Name', [up_prefix 'conv1'])
            batchNormalizationLayer('Name', [up_prefix 'bn2'])
            leakyReluLayer(0.2, 'Name', [up_prefix 'lrelu1'])
            convolution2dLayer(1, up_channels(j), 'Padding', 'same', 'Name', [up_prefix 'conv2'])  % FIX BUG 4
            batchNormalizationLayer('Name', [up_prefix 'bn3'])
            leakyReluLayer(0.2, 'Name', [up_prefix 'lrelu2'])       
        ];

        deepNet = addLayers(deepNet, decoder_block);
        deepNet = connectLayers(deepNet, currentInput, [up_prefix 'upsample']);
        currentInput = [up_prefix 'lrelu2'];                         
    end
end


final_layer = [
    convolution2dLayer(1, numClasses, 'Name', 'final_conv')
    sigmoidLayer('Name', 'final_sigmoid')
];
deepNet = addLayers(deepNet, final_layer);
deepNet = connectLayers(deepNet, currentInput, 'final_conv');

%net = dlnetwork(lgraph);
net = initialize(deepNet);
end