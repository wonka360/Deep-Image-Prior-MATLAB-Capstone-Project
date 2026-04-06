function resized_image = get_image(image, imsize)
        
        if nargin < 2
            imsize = -1;
        end
        
        % Check if imsize is a scaler
        if isscalar(imsize)
            imsize = [imsize, imsize];
        end
        resized_image = image; % Default to original image if no resizing is needed

        % Resize Image to specified dimensions if needed
        if imsize(1) ~= -1
            current_size = [size(image, 2), size(image, 1)]; % W, H
            if ~isequal(current_size, imsize)
                if imsize(1) > current_size(1)
                    % bicubic upsampling
                    resized_image = imresize(image, imsize, 'bicubic');
                else
                    % Downsampling anti-aliasing
                    resized_image = imresize(image, imsize, 'Antialiasing',true);
                end
            end
        end
            
              
end