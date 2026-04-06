function img_cropped = crop_image(image, d)

    if nargin < 2
        d = 32;
    end

    % Get image size
    H = size(image, 1);
    W = size(image, 2);

    % Compute largest size divisible by d
    newH = H - mod(H, d);
    newW = W - mod(W, d);

    % Compute crop start (center crop)
    top  = floor((H - newH) / 2) + 1;
    left = floor((W - newW) / 2) + 1;

    % Compute crop end
    bottom = top  + newH - 1;
    right  = left + newW - 1;

    % Perform crop
    if ndims(image) == 2
        img_cropped = image(top:bottom, left:right);
    else
        img_cropped = image(top:bottom, left:right, :);
    end
end
