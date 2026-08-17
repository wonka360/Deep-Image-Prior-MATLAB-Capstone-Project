function net = InpaintUnet(imageSize, numClasses)

down_ch  = [128, 128, 128, 128, 128];
up_ch    = [128, 128, 128, 128, 128];
skip_ch  = [128, 128, 128, 128, 128];
numScales = 5;

lgraph = layerGraph();

inputL = imageInputLayer(imageSize, 'Name', 'input', 'Normalization', 'none');
lgraph = addLayers(lgraph, inputL);
currentInput = 'input';


% ------ ENCODER PATH ------


skipOutputs = cell(1, numScales);

for i = 1:numScales
    enc_prefix  = sprintf('enc%d_', i);
    skip_prefix = sprintf('skip%d_', i);

    % --- Encoder (deeper) branch ---
    
    enc_block = [
        convolution2dLayer(3, down_ch(i), 'Padding', 'same', 'Stride', 2, ...
                           'Name', [enc_prefix 'conv1'])
        batchNormalizationLayer('Name', [enc_prefix 'bn1'])
        leakyReluLayer(0.2, 'Name', [enc_prefix 'lrelu1'])
        convolution2dLayer(3, down_ch(i), 'Padding', 'same', ...
                           'Name', [enc_prefix 'conv2'])
        batchNormalizationLayer('Name', [enc_prefix 'bn2'])
        leakyReluLayer(0.2, 'Name', [enc_prefix 'lrelu2'])
    ];

    % --- Skip branch (1x1 conv, from same input as encoder) ---
    skip_block = [
        convolution2dLayer(1, skip_ch(i), 'Padding', 'same', ...
                           'Name', [skip_prefix 'conv1'])
        batchNormalizationLayer('Name', [skip_prefix 'bn1'])
        leakyReluLayer(0.2, 'Name', [skip_prefix 'lrelu1'])
    ];

    lgraph = addLayers(lgraph, enc_block);
    lgraph = addLayers(lgraph, skip_block);

    
    lgraph = connectLayers(lgraph, currentInput, [enc_prefix 'conv1']);
    lgraph = connectLayers(lgraph, currentInput, [skip_prefix 'conv1']);

    
    skipOutputs{i} = [skip_prefix 'lrelu1'];

    
    currentInput = [enc_prefix 'lrelu2'];
end

% ---- DECODER PATH ----

for i = numScales:-1:1
    dec_prefix = sprintf('dec%d_', i);

   
    

    
    upsample_layer = resize2dLayer('Scale', 2, 'Method', 'nearest', ...
                                   'Name', [dec_prefix 'upsample']);

    bn_pre = batchNormalizationLayer('Name', [dec_prefix 'bnorm']);

   
    concat_layer = concatenationLayer(3, 2, 'Name', [dec_prefix 'concat']);

    
    dec_block = [
        convolution2dLayer(3, up_ch(i), 'Padding', 'same', ...
                           'Name', [dec_prefix 'conv1'])
        batchNormalizationLayer('Name', [dec_prefix 'bn1'])
        leakyReluLayer(0.2, 'Name', [dec_prefix 'lrelu1'])
    ];

   
    dec_1x1 = [
        convolution2dLayer(1, up_ch(i), 'Padding', 'same', ...
                           'Name', [dec_prefix 'conv2'])
        batchNormalizationLayer('Name', [dec_prefix 'bn2'])
        leakyReluLayer(0.2, 'Name', [dec_prefix 'lrelu2'])
    ];

    % Add all layers
    lgraph = addLayers(lgraph, upsample_layer);
    lgraph = addLayers(lgraph, bn_pre);
    lgraph = addLayers(lgraph, concat_layer);
    lgraph = addLayers(lgraph, dec_block);
    lgraph = addLayers(lgraph, dec_1x1);

    
    lgraph = connectLayers(lgraph, currentInput,            [dec_prefix 'upsample']);
    lgraph = connectLayers(lgraph, [dec_prefix 'upsample'], [dec_prefix 'bnorm']);

   
    lgraph = connectLayers(lgraph, skipOutputs{i},      [dec_prefix 'concat/in1']);
    lgraph = connectLayers(lgraph, [dec_prefix 'bnorm'],[dec_prefix 'concat/in2']);

    
    lgraph = connectLayers(lgraph, [dec_prefix 'concat'], [dec_prefix 'conv1']);
    lgraph = connectLayers(lgraph, [dec_prefix 'lrelu1'], [dec_prefix 'conv2']);

    currentInput = [dec_prefix 'lrelu2'];
end

% --- OUTPUT ---
final_layers = [
    convolution2dLayer(1, numClasses, 'Name', 'final_conv')
    sigmoidLayer('Name', 'final_sigmoid')
];
lgraph = addLayers(lgraph, final_layers);
lgraph = connectLayers(lgraph, currentInput, 'final_conv');

net = dlnetwork(lgraph);
end