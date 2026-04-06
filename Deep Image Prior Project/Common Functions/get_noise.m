function net_input = get_noise(input_depth, method, spatial_size, noise_type, var)

        % Function returns a matlab noise tensor of size H x W x...
        % input_depth x 1

        if nargin < 4
            noise_type = 'u';
        end
        
        if nargin < 5
            var = 1/10;
        end
        if isscalar(spatial_size)
            spatial_size = [spatial_size, spatial_size];
        end
        H = spatial_size(1);
        W = spatial_size(2);

        if strcmp(method, 'noise')
            if  strcmp(noise_type, 'u')
                net_input = rand(H, W, input_depth, 1);
            else
                net_input = randn(H, W, input_depth, 1);
            end
            net_input = net_input * var; % Scale the noise by the variance
        elseif strcmp(method, 'meshgrid')
            %assert(input_depth==2)

            [X, Y] = meshgrid((0:W-1)/(W-1), (0:H-1)/(H-1));
            
            net_input = cat(3, X, Y);
            
            net_input = reshape(net_input, H, W, 2, 1);
         else
            error('Invalid method specified. Use ''noise'' or ''meshgrid''.');
        end
end