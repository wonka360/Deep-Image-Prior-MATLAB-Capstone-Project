function net = InpaintUnet2(imageSize, numClasses)


down_ch   = [128, 128, 128, 128, 128];
up_ch     = [128, 128, 128, 128, 128];
numScales = 5;

lgraph = layerGraph();

% --- Input ---
inputL = imageInputLayer(imageSize, 'Name', 'input', 'Normalization', 'none');
lgraph = addLayers(lgraph, inputL);
currentInput = 'input';


% ---- ENCODER PATH ---
for i = 1:numScales
    prefix = sprintf('enc%d_', i);

    enc_block = [
        convolution2dLayer(3, down_ch(i), 'Padding', 'same', 'Stride', 2, ...
                           'Name', [prefix 'conv1'])
        batchNormalizationLayer('Name', [prefix 'bn1'])
        leakyReluLayer(0.2, 'Name', [prefix 'lrelu1'])
        convolution2dLayer(3, down_ch(i), 'Padding', 'same', ...
                           'Name', [prefix 'conv2'])
        batchNormalizationLayer('Name', [prefix 'bn2'])
        leakyReluLayer(0.2, 'Name', [prefix 'lrelu2'])
    ];

    lgraph = addLayers(lgraph, enc_block);
    lgraph = connectLayers(lgraph, currentInput, [prefix 'conv1']);
    currentInput = [prefix 'lrelu2'];
end

% ---- DECODER PATH ----
for i = numScales:-1:1
    dec_prefix = sprintf('dec%d_', i);

    % Channel count at this decoder level
    
    dec_block = [
       
        resize2dLayer('Scale', 2, 'Method', 'nearest', ...
                      'Name', [dec_prefix 'upsample'])

        
        batchNormalizationLayer('Name', [dec_prefix 'bnorm'])

        
        convolution2dLayer(3, up_ch(i), 'Padding', 'same', ...
                           'Name', [dec_prefix 'conv1'])
        batchNormalizationLayer('Name', [dec_prefix 'bn1'])
        leakyReluLayer(0.2, 'Name', [dec_prefix 'lrelu1'])

        
        convolution2dLayer(1, up_ch(i), 'Padding', 'same', ...
                           'Name', [dec_prefix 'conv2'])
        batchNormalizationLayer('Name', [dec_prefix 'bn2'])
        leakyReluLayer(0.2, 'Name', [dec_prefix 'lrelu2'])
    ];

    lgraph = addLayers(lgraph, dec_block);
    lgraph = connectLayers(lgraph, currentInput, [dec_prefix 'upsample']);
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