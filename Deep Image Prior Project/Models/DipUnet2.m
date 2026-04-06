function net = DipUnet2(imageSize, numClasses)

% Initial Parameters
down_channels = [8, 16, 32, 64, 128];
up_channels = [8, 16, 32, 64, 128];
numScales = 5;
lgraph = layerGraph();

% Input Layer
inputL = imageInputLayer(imageSize,'Name','input','Normalization','none');
lgraph = addLayers(lgraph,inputL);
currentInput = 'input';


% --- Encoder Path ---
for i = 1:numScales
    
    prefix = sprintf('enc%d_', i);

    encoder_block = [
        
        convolution2dLayer(3,down_channels(i),'Padding','same','Name',[prefix 'conv1'])
        batchNormalizationLayer('Name',[prefix 'bn1'])
        leakyReluLayer(0.2,'Name',[prefix 'lrelu1'])

        convolution2dLayer(3,down_channels(i),'Padding','same','Name',[prefix 'conv2'])
        batchNormalizationLayer('Name',[prefix 'bn2'])
        leakyReluLayer(0.2,'Name',[prefix 'lrelu2'])
        ];
    lgraph = addLayers(lgraph,encoder_block);

    % connect previous block
    lgraph = connectLayers(lgraph,currentInput,[prefix 'conv1']);

    currentInput = [prefix 'lrelu2'];
end

% --- Decoder Path ---
for i = 0:numScales-1
    j = numScales-i;

    up_prefix = sprintf('dec%d_',j);

    if j == 5 || j == 4
        skip_prefix = sprintf('skip%d_',j);

        dec_upsample = resize2dLayer('Scale', 2, 'Method', 'bilinear', 'Name', [up_prefix 'upsample']);
        skip_block = [
            convolution2dLayer(1,4,'Padding','same','Name',[skip_prefix 'conv1'])
            batchNormalizationLayer('Name',[skip_prefix 'bn1'])
            leakyReluLayer(0.2,'Name',[up_prefix 'lrelu1'])
            ];

        
        batchNorm = batchNormalizationLayer('Name',[up_prefix 'bnorm']);
        

        decoder_block = [
            concatenationLayer(3,2,'Name',[up_prefix 'concat'])

            convolution2dLayer(3,down_channels(j),'Padding','same','Name',[up_prefix 'conv1'])
            batchNormalizationLayer('Name',[up_prefix 'bn1'])
            leakyReluLayer(0.2,'Name',[up_prefix 'lrelu1'])

            convolution2dLayer(3,down_channels(j),'Padding','same','Name',[up_prefix 'conv2'])
            batchNormalizationLayer('Name',[up_prefix 'bn2'])
            leakyReluLayer(0.2,'Name',[up_prefix 'lrelu2'])
            ];
        % Add Layers to graph
        lgraph = addLayers(lgraph, dec_upsample);
        lgraph = addLayers(lgraph, skip_block);
        lgraph = addLayers(lgraph, decoder_block);
        lgraph = addLayers(lgraph, batchNorm);

        % Connect layers in the decoder path
        lgraph = connectLayers(lgraph, currentInput, [up_prefix 'upsample']);
        lgraph = connectLayers(lgraph, [up_prefix 'upsample'], [up_prefix 'bnorm']);
        
        % Skip source
        skipSource = sprintf('enc%d_lrelu2',j);
        lgraph = connectLayers(lgraph, skipSource, [skip_prefix 'conv1']);

        lgraph = connectLayers(lgraph, [skip_prefix 'lrelu1'], [up_prefix 'concat/in1']);
        lgraph = connectLayers(lgraph, [up_prefix 'bnorm'], [up_prefix 'concat/in2']);
        currentInput = [up_prefix 'lrelu2'];
    else
        decoder_block = [
            resize2dLayer('Scale', 2, 'Method', 'bilinear', 'Name', [up_prefix 'upsample'])
            batchNormalizationLayer('Name',[up_prefix 'bn1'])
            convolution2dLayer(3,up_channels(j),'Padding','same','Name',[up_prefix 'conv1'])

            batchNormalizationLayer('Name',[up_prefix 'bn2'])
            leakyReluLayer(0.2,'Name',[up_prefix 'lrelu1'])
            convolution2dLayer(1,up_channels(j),'Padding','same','Name',[up_prefix 'conv1'])
            ];
        lgraph = addLayers(lgraph, decoder_block);
        % Connect layers
        lgraph = connectLayers(lgraph, currentInput, [up_prefix 'upsample']);
        currentInput = [up_prefix 'conv1'];
    end

end


final_layer = [
    convolution2dLayer(3,numClasses,'Name','final_conv')
    sigmoidLayer('Name','final_sigmoid')
    ];
lgraph = addLayers(lgraph, final_layer);
lgraph = connectLayers(lgraph, currentInput, 'final_conv');
net = dlnetwork(lgraph);
