% DL n/w for solving inpainting problem
function net = InpaintUnet4(imageSize, numOutputChannels, numFilters, stackdepth)
    
        %------- Encoder ------
        % Define a custom ResNet Encoder
        
        
        encoder_backbone = resnetNetwork(imageSize, ...
            numOutputChannels,...
            StackDepth = stackdepth, ...
            NumFilters = numFilters);
        % Remove final classification layer
        layersToRemove = {};
        for i = 1:numel(encoder_backbone.Layers)
            layer = encoder_backbone.Layers(i);
            % Check if the layer is a global pool, FC, softmax, or output layer
            if isa(layer, 'nnet.cnn.layer.GlobalAveragePooling2DLayer') || ...
                    isa(layer, 'nnet.cnn.layer.FullyConnectedLayer') || ...
                    isa(layer, 'nnet.cnn.layer.SoftmaxLayer') || ...
                    isa(layer, 'nnet.cnn.layer.ClassificationOutputLayer')

                layersToRemove{end+1} = layer.Name; %#ok<AGROW>
            end
        end

        % Remove them all to expose the spatial feature map to the decoder
        encoder_backbone = removeLayers(encoder_backbone, layersToRemove);
        
        % Initialize decoder branch
        decoder_branch =  dlnetwork;
        %---- Bottleneck layer ----
        bottl_neck_layer = [
            convolution2dLayer(3,128,'Padding','same','Stride',1,'Name','e5_c1')
            batchNormalizationLayer('Name','e5_b1')
            leakyReluLayer(0.2,'Name','e5_a1')
            convolution2dLayer(3,128,'Padding','same','Stride',1,'Name','e5_c2')
            batchNormalizationLayer('Name','e5_b2')
            leakyReluLayer(0.2,'Name','e5_a2')
            
            ];
        decoder_branch = addLayers(decoder_branch, bottl_neck_layer);

        %---- Decoder -----
        % Define Decoder blocks manually
        decoder_branch = addLayers(decoder_branch, [
            resize2dLayer('Scale',2,'Method','bilinear','Name','d5_up')
            batchNormalizationLayer('Name','d5_bn0')          % bn(k=128)
            convolution2dLayer(3,128,'Padding','same','Name','d5_c1')
            batchNormalizationLayer('Name','d5_b1')
            leakyReluLayer(0.2,'Name','d5_a1')
            convolution2dLayer(1,128,'Padding','same','Name','d5_c2')  % 1x1
            batchNormalizationLayer('Name','d5_b2')
            leakyReluLayer(0.2,'Name','d5_a2')
            ]);
        decoder_branch = connectLayers(decoder_branch,'e5_a2','d5_up');

        % Scale 4 -> 3  (k=nCu[5]=128 -> nCu[3]=128)
        decoder_branch = addLayers(decoder_branch, [
            resize2dLayer('Scale',2,'Method','bilinear','Name','d4_up')
            batchNormalizationLayer('Name','d4_bn0')          % bn(k=128)
            convolution2dLayer(3,128,'Padding','same','Name','d4_c1')
            batchNormalizationLayer('Name','d4_b1')
            leakyReluLayer(0.2,'Name','d4_a1')
            convolution2dLayer(1,128,'Padding','same','Name','d4_c2')  % 1x1
            batchNormalizationLayer('Name','d4_b2')
            leakyReluLayer(0.2,'Name','d4_a2')
            ]);
        decoder_branch = connectLayers(decoder_branch,'d5_a2','d4_up');

        % Scale 3 -> 2  (k=nCu[4]=128 -> nCu[2]=64)
        decoder_branch = addLayers(decoder_branch, [
            resize2dLayer('Scale',2,'Method','bilinear','Name','d3_up')
            batchNormalizationLayer('Name','d3_bn0')          % bn(k=128)
            convolution2dLayer(3,64,'Padding','same','Name','d3_c1')
            batchNormalizationLayer('Name','d3_b1')
            leakyReluLayer(0.2,'Name','d3_a1')
            convolution2dLayer(1,64,'Padding','same','Name','d3_c2')   % 1x1
            batchNormalizationLayer('Name','d3_b2')
            leakyReluLayer(0.2,'Name','d3_a2')
            ]);
        decoder_branch = connectLayers(decoder_branch,'d4_a2','d3_up');

        % Scale 2 -> 1  (k=nCu[3]=64 -> nCu[1]=32)
        decoder_branch = addLayers(decoder_branch, [
            resize2dLayer('Scale',2,'Method','bilinear','Name','d2_up')
            batchNormalizationLayer('Name','d2_bn0')          % bn(k=64)
            convolution2dLayer(3,32,'Padding','same','Name','d2_c1')
            batchNormalizationLayer('Name','d2_b1')
            leakyReluLayer(0.2,'Name','d2_a1')
            convolution2dLayer(1,32,'Padding','same','Name','d2_c2')   % 1x1
            batchNormalizationLayer('Name','d2_b2')
            leakyReluLayer(0.2,'Name','d2_a2')
            ]);
        decoder_branch = connectLayers(decoder_branch,'d3_a2','d2_up');
        
        % Scale 1 -> out  (k=nCu[2]=32 -> nCu[0]=16)
        decoder_branch = addLayers(decoder_branch, [
            resize2dLayer('Scale',2,'Method','bilinear','Name','d1_up')
            batchNormalizationLayer('Name','d1_bn0')          % bn(k=32)
            convolution2dLayer(3,16,'Padding','same','Name','d1_c1')
            batchNormalizationLayer('Name','d1_b1')
            leakyReluLayer(0.2,'Name','d1_a1')
            convolution2dLayer(1,16,'Padding','same','Name','d1_c2')   % 1x1
            batchNormalizationLayer('Name','d1_b2')
            leakyReluLayer(0.2,'Name','d1_a2')
            ]);
        decoder_branch = connectLayers(decoder_branch,'d2_a2','d1_up');

        % ── Output head ──────────────────────────────────────────────────────────────
        % conv1x1(16 -> numOutputChannels) -> sigmoid
        decoder_branch = addLayers(decoder_branch, [
            convolution2dLayer(1,numOutputChannels,'Padding','same','Name','final_conv')
            sigmoidLayer('Name','final_sigmoid')
            ]);
        decoder_branch = connectLayers(decoder_branch,'d1_a2','final_conv');

        
        net = encoderDecoderNetwork(imageSize, encoder_backbone, decoder_branch, ...
            'SkipConnections', 'concatenate');


end