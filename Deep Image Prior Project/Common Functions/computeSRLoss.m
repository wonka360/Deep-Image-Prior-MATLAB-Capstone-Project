function loss = computeSRLoss(out_HR, img_LR_dl, ratio, lambda_tv)
        % Downsample HR output to LR size via average pooling
        %out_LR = avgpool(out_HR, [factor factor], 'Stride', [factor factor], 'Padding', 'same');
        out_LR = dlresize(out_HR, 'OutputSize', [128, round(128*ratio)], 'Method', 'linear');
        mse_loss = mean((out_LR - img_LR_dl).^2, 'all');
        
        %Total Variation: penalize spatial gradients in output  
        dx = out_HR(:, 2:end, :, :) - out_HR(:, 1:end-1, :, :);
        dy = out_HR(2:end, :, :, :) - out_HR(1:end-1, :, :, :);
        tv_loss = mean(abs(dx), 'all') + mean(abs(dy), 'all');
        
        loss = mse_loss + lambda_tv * tv_loss;
end
