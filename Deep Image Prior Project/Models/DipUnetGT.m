function net = DipUnetGT(imageSize, numClasses, numScales)

% Initial parameters
numFilters = 64; 
lgraph = layerGraph();

% Input Layer
inputL = imageInputLayer(imageSize,'Name','input','Normalization','none');
lgraph = addLayers(lgraph,inputL);

currentInput = 'input';

% --- Encoder Path ---
for i = 1:numScales

    prefix = sprintf('enc%d_', i);

    conv_block = [

        convolution2dLayer(3,numFilters,'Padding','same','Name',[prefix 'conv1'])
        batchNormalizationLayer('Name',[prefix 'bn1'])      
        leakyReluLayer(0.2,'Name',[prefix 'lrelu1'])

        convolution2dLayer(3,numFilters,'Padding','same','Name',[prefix 'conv2'])
        batchNormalizationLayer('Name',[prefix 'bn2'])      
        leakyReluLayer(0.2,'Name',[prefix 'lrelu2'])
        ];

    lgraph = addLayers(lgraph,conv_block);

    % connect previous block
    lgraph = connectLayers(lgraph,currentInput,[prefix 'conv1']);

    currentInput = [prefix 'lrelu2'];

    % Pooling except last scale
    if i < numScales

        pool = maxPooling2dLayer(2,'Stride',2,'Padding','same','Name',[prefix 'pool']);
        lgraph = addLayers(lgraph,pool);

        lgraph = connectLayers(lgraph,[prefix 'lrelu2'],[prefix 'pool']);
        currentInput = [prefix 'pool'];

    end
end

% --- Decoder Path ---
for i = 1:numScales-1

    up_prefix = sprintf('dec%d_',i);

    
    dec_block_pre = transposedConv2dLayer(2,numFilters,...
        'Stride',2,...
        'Cropping','same',...
        'Name',[up_prefix 'upconv']);

    dec_block_post = [

        concatenationLayer(3,2,'Name',[up_prefix 'concat'])

        convolution2dLayer(3,numFilters,'Padding','same','Name',[up_prefix 'conv1'])
        batchNormalizationLayer('Name',[up_prefix 'bn1'])   
        leakyReluLayer(0.2,'Name',[up_prefix 'lrelu1'])

        convolution2dLayer(3,numFilters,'Padding','same','Name',[up_prefix 'conv2'])
        batchNormalizationLayer('Name',[up_prefix 'bn2'])   
        leakyReluLayer(0.2,'Name',[up_prefix 'lrelu2'])
        ];

    lgraph = addLayers(lgraph,dec_block_pre);
    lgraph = addLayers(lgraph,dec_block_post);

    % Upsampling path
    lgraph = connectLayers(lgraph,currentInput,[up_prefix 'upconv']);
    lgraph = connectLayers(lgraph,[up_prefix 'upconv'],[up_prefix 'concat/in1']);

    % Skip connection
    skipSource = sprintf('enc%d_lrelu2',numScales-i);
    lgraph = connectLayers(lgraph,skipSource,[up_prefix 'concat/in2']);

    currentInput = [up_prefix 'lrelu2'];

end

% --- Final Layer ---
final_layer = [
    convolution2dLayer(1,numClasses,'Name','final_conv')
    sigmoidLayer('Name','final_sigmoid')
    ];

lgraph = addLayers(lgraph,final_layer);
lgraph = connectLayers(lgraph,currentInput,'final_conv');

net = dlnetwork(lgraph);






end





