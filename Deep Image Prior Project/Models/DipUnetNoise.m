function net = DipUnetNoise(imageSize, numScales)

    assert(numScales == 5, 'numScales must be 5 for this configuration');

    num_channels_down = 8 * (2.^(0:numScales-1));  % [8 16 32 64 128]
    num_channels_up   = 8 * (2.^(0:numScales-1));  % [8 16 32 64 128]
    num_channels_skip = [0, 0, 0, 4, 4];
    input_depth = imageSize(3);

    lgraph = layerGraph();

    % Input Layer
    inputLayer = imageInputLayer(imageSize, 'Name', 'input', 'Normalization', 'none');
    lgraph = addLayers(lgraph, inputLayer);
    currentInput = 'input';

    % --- Encoder ---
    for i = 1:numScales
        prefix = sprintf('enc_%d_', i);
        convBlock = [
            reflectionPadding2dLayer(1, 'Name', [prefix 'pad1'])
            convolution2dLayer(3, num_channels_down(i), 'Stride', 1, 'Padding', 0, 'Name', [prefix 'conv1'])
            leakyReluLayer(0.01, 'Name', [prefix 'relu1'])
            reflectionPadding2dLayer(1, 'Name', [prefix 'pad2'])
            convolution2dLayer(3, num_channels_down(i), 'Stride', 1, 'Padding', 0, 'Name', [prefix 'conv2'])
            leakyReluLayer(0.01, 'Name', [prefix 'relu2'])
        ];
        lgraph = addLayers(lgraph, convBlock);
        lgraph = connectLayers(lgraph, currentInput, [prefix 'pad1']);
        currentInput = [prefix 'relu2'];

        % Downsample with maxpool after every level except the last
        if i < numScales
            pool = maxPooling2dLayer(2, 'Stride', 2, 'Name', [prefix 'pool']);
            lgraph = addLayers(lgraph, pool);
            lgraph = connectLayers(lgraph, [prefix 'relu2'], [prefix 'pool']);
            currentInput = [prefix 'pool'];
        end
    end

    % --- Decoder ---
    for i = 1:numScales-1
        up_prefix  = sprintf('dec_%d_', i);
        enc_level  = numScales - i;  % 4, 3, 2, 1

        % Upsample
        lgraph = addLayers(lgraph, resize2dLayer('Scale', 2, 'Method', 'bilinear', ...
            'Name', [up_prefix 'upsample']));
        lgraph = connectLayers(lgraph, currentInput, [up_prefix 'upsample']);

        % Skip connection
        skip_out_name = sprintf('enc_%d_relu2', enc_level);

        if num_channels_skip(enc_level) > 0
            skip_prefix = sprintf('skip_%d_', enc_level);
            sLayers = [
                convolution2dLayer(1, num_channels_skip(enc_level), 'Name', [skip_prefix 'conv'])
                leakyReluLayer(0.01, 'Name', [skip_prefix 'relu'])
            ];
            lgraph = addLayers(lgraph, sLayers);
            lgraph = connectLayers(lgraph, skip_out_name, [skip_prefix 'conv']);
            skip_final = [skip_prefix 'relu'];
        else
            skip_final = skip_out_name;
        end

        % Concatenation
        lgraph = addLayers(lgraph, concatenationLayer(3, 2, 'Name', [up_prefix 'concat']));
        lgraph = connectLayers(lgraph, [up_prefix 'upsample'], [up_prefix 'concat/in1']);
        lgraph = connectLayers(lgraph, skip_final,             [up_prefix 'concat/in2']);

        % Post-concat conv block
        postLayers = [
            reflectionPadding2dLayer(1, 'Name', [up_prefix 'pad1'])
            convolution2dLayer(3, num_channels_up(enc_level), 'Stride', 1, 'Padding', 0, 'Name', [up_prefix 'conv1'])
            leakyReluLayer(0.01, 'Name', [up_prefix 'relu1'])
        ];
        lgraph = addLayers(lgraph, postLayers);
        lgraph = connectLayers(lgraph, [up_prefix 'concat'], [up_prefix 'pad1']);
        currentInput = [up_prefix 'relu1'];
    end

    % --- Output ---
    finalLayers = [
        convolution2dLayer(1, input_depth, 'Name', 'final_conv')
        sigmoidLayer('Name', 'output_sigmoid')
    ];
    lgraph = addLayers(lgraph, finalLayers);
    lgraph = connectLayers(lgraph, currentInput, 'final_conv');

    net = dlnetwork(lgraph);

         
end