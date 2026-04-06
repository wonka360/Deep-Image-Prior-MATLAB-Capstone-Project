classdef reflectionPadding2dLayer < nnet.layer.Layer
    properties
        PadSize
    end
    methods
        function layer = reflectionPadding2dLayer(padsize, options)
            arguments
                padsize (1, 1) double
                options.Name string = ''
            end
            layer.PadSize = padsize;
            layer.Name = options.Name;
            layer.Description = 'Reflection Padding 2d';
        end
        function Z = predict(layer, X)
            p = layer.PadSize;
            X = cat(1, X(p+1:-1:2,:,:,:),X,X(end-1:-1:end-p,:,:,:));
            X = cat(2, X(:,p+1:-1:2,:,:),X,X(:,end-1:-1:end-p,:,:));
            Z = X;    
        end
    end
end
