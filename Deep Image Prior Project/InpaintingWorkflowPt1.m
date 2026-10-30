%[text] # Inpainting Workflow Part 1
%%
%[text] ## Add File Paths
addpath 'Inpainting'
addpath 'Models'
addpath 'Common Functions'
%%
%[text] ## Load Files
Image_name = 'library.png';
mask_name = 'library_mask.png';
% Load image
image = im2double(imread(Image_name));
mask  = imread(mask_name);

mask = im2double(mask);
mask = mask > 0.5;
mask = repmat(mask, [1 1 3]);

img = image .* mask;
img = imresize(img, [256, 256]);
mask = imresize(mask, [256, 256], 'nearest');
inpainted_image = img;
imshow(inpainted_image);
%%
%[text] ## Define CNN and Noise Tensor
% --- Create Noise Tensor ---
rng(42); % Seed
input_depth = 32;
%method = 'meshgrid';
method = 'noise';
if strcmp(method, 'meshgrid')
    input_depth = 2;
end
spatial_size = size(inpainted_image, 1);
net_input = get_noise(input_depth, method, spatial_size);

% Visualize Noise
imagesc(net_input(:,:,2))
axis image
colormap gray
colorbar

% --- Create Unet with ResNet Encoder ---
num_classes = 3; % Number of color channels
[a, b, ~] = size(inpainted_image);
imgSize = [a, b, input_depth]; % Set the image size for the CNN

numFilters = [16, 32, 64, 128];
stackdepth = [1, 1, 1, 1];
net = InpaintUnet4(imgSize, num_classes, numFilters, stackdepth);
% Count number of parameters
totalParams = 0;
lrnables = net.Learnables;
for i = 1:height(lrnables)
    param = lrnables.Value{i};
    totalParams = totalParams + numel(extractdata(param));
end
fprintf('Number of params: %d\n', totalParams);

% --- GPU setup (only if a usable GPU exists) ---
useGPU = canUseGPU;                      
if useGPU
    fprintf('Using GPU: %s\n', gpuDevice().Name);
    gpurng(42);                           % rng(42) does not seed GPU random numbers
else
    fprintf('No GPU available, running on CPU.\n');
end

net_input_saved = single(net_input);
img_in = single(inpainted_image);
mask_in = single(mask);           % 'mask' itself stays a CPU logical (used for PSNR indexing)

if useGPU
    net_input_saved = gpuArray(net_input_saved);
    img_in = gpuArray(img_in);
    mask_in = gpuArray(mask_in);
    net = dlupdate(@gpuArray, net);       % learnables to GPU
    if ~isempty(net.State)                % batch-norm state to GPU
        net.State.Value = cellfun(@gpuArray, net.State.Value, 'UniformOutput', false);
    end
end

% Convert to dlarray 
net_input_saved = dlarray(net_input_saved, 'SSCB');
img_in = dlarray(img_in, 'SSCB');
mask_in = dlarray(mask_in, 'SSCB');

%--- Define Loss Function ---
%loss = mean((out - inpainted_image).^2,'all');
function loss = computeLoss(out, img_noisy, lambda_tv) 
        mse_loss = mean((out - img_noisy).^2, 'all'); 
        %Total Variation: penalize spatial gradients in output 
        dx = out(:, 2:end, :, :) - out(:,1:end-1, :, :); 
        dy = out(2:end, :, :, :) - out(1:end-1, :, :, :); 
        tv_loss = mean(abs(dx),'all') + mean(abs(dy), 'all'); 
        loss = mse_loss + lambda_tv * tv_loss; 
end
%%
%[text] ## Gradient Function
function [loss, gradients, out, out_avg, state] = modelGradients(net, net_input_saved, img_inpaint, mask, out_avg, exp_weight, reg_noise_std)
        
        if reg_noise_std > 0
            noise = randn(size(net_input_saved), 'like', extractdata(net_input_saved));
            noise = dlarray(noise, 'SSCB');
            net_input = net_input_saved + noise*reg_noise_std;
        else
            net_input = net_input_saved;
        end
        
        % Forward pass
        [out, state] = forward(net, net_input);
        
        % Loss (MSE)
        %loss = mean((out.*mask - img_inpaint.*mask).^2,'all');
        
        loss = computeLoss(out.*mask, img_inpaint.*mask, 1e-3);
        
        % Output smoothing (detach)
        out_detached = dlarray(extractdata(out),'SSCB');
        
        if isempty(out_avg)
            out_avg = out_detached;
        else
            out_avg = out_avg*exp_weight + out_detached*(1-exp_weight);
        end
        
        % Gradients
        gradients = dlgradient(loss, net.Learnables);
end
%%
%[text] ## Optimization Step
trailingAvg   = [];
trailingAvgSq = [];
reg_noise_std = 0.06;

PLOT = 1; % 1 for plotting and 0 otherwise
LR = 0.01;
show_every = 100;
exp_weight = 0.99;
num_iter = 7000;

out_avg = [];
last_net = [];
last_learnables = [];
psnr_inpaint_last = -inf;
psnr_best = -inf;
best = []; % Checkpoint struct
drop_iter = 200;

beta = 0.9;
min_delta = 0.01;
iter_const = 30;
cooldown = 30;
drop_thresh  = 1.0;
lr_decay = 0.5;
LR_min = 1e-4;
max_restores = 5;
param_noise = true;

psnr_s = [];  cooldown_left = 0;  n_restores = 0;  psnr_best_const = 0;

% --- Initialize Training Progress Monitor ---
monitor = trainingProgressMonitor( ...
    Metrics=["Loss", "PSNR", "Smoothed_PSNR"], ...
    Info=["Iteration", "LearningRate", "Best_Smoothed_PSNR"], ...
    XLabel="Iteration");



%--- Training Loop ---
for iter = 1:num_iter

    % Call function for forward pass
    %accGradients = dlaccelerate(@modelGradients);
    %[loss, gradients, out, out_avg, state] = dlfeval(accGradients, net, net_input_saved, inpainted_image, mask, out_avg, exp_weight, reg_noise_std);
    [loss, gradients, out, out_avg, state] = dlfeval(@modelGradients, net, net_input_saved, img_in, mask_in,  out_avg, exp_weight, reg_noise_std);
    


    [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients,trailingAvg, trailingAvgSq, iter, LR);

    % Update Batch Norm state
    net.State = state;

    % --- Convert dlarray to numeric ---
    out_num = double(gather(extractdata(out)));
    out_num = out_num(:,:,:,1);
    out_avg_num = double(gather(extractdata(out_avg)));
    out_avg_num = out_avg_num(:,:,:,1);


    %--- Compute PSNR ---
    psnr_inpaint = psnr(out_num(mask), inpainted_image(mask));

    %--- Checkpointing & Back Tracking ---
    if isempty(psnr_s)
        psnr_s = psnr_inpaint;
    else
        psnr_s = beta*psnr_s + (1-beta)*psnr_inpaint;
    end

    if psnr_inpaint > psnr_best + min_delta
        psnr_best = psnr_s;
        psnr_best_const = 0;
        best.learnables = net.Learnables.Value;
        best.trailingAvg = trailingAvg;
        best.trailingAvgSq = trailingAvgSq;
        best.out_avg = out_avg;
        best.iter = iter;
        best.State = net.State;
    elseif cooldown_left > 0
        cooldown_left = cooldown_left-1;
    else
        psnr_best_const = psnr_best_const + 1;
    end

    if iter >= drop_iter && psnr_best_const >= iter_const
        if (psnr_best - psnr_s) > drop_thresh
            if n_restores < max_restores
                n_restores = n_restores + 1;
                fprintf('\nIter %d: degraded %.2f dB — restoring iter %d.\n', ...
                    iter, psnr_best - psnr_s, best.iter);
                net.Learnables.Value = best.learnables;
                net.State = best.State;
                trailingAvg = best.trailingAvg;
                trailingAvgSq = best.trailingAvgSq;
                out_avg = best.out_avg;
                psnr_s = psnr_best;
            else
                fprintf('\nIter %d: restore budget used up, continuing without restore.\n', iter);
            end
        else
            fprintf('\nIter %d: plateau, decaying LR only.\n', iter);
        end
        LR = max(LR * lr_decay, LR_min);
        psnr_best_const = 0;
        cooldown_left = cooldown;
        continue;
        
    end

    % --- Update UI Monitor ---
    recordMetrics(monitor, iter, Loss=gather(extractdata(loss)), PSNR=psnr_inpaint, Smoothed_PSNR=psnr_s);
    updateInfo(monitor, ...
        "Iteration", iter + " / " + num_iter, ...
        "LearningRate", LR, ...
        "Best_Smoothed_PSNR", psnr_best);

    % Check if Stop button was pressed
    if monitor.Stop
        fprintf('\nTraining stopped by user at iteration %d.\n', iter);
        break;
    end

    fprintf('Iter %05d Loss %.6f PSNR_LR: %.4f | smoothed: %.4f | best(smoothed): %.4f\r', ...
        iter, gather(extractdata(loss)), psnr_inpaint, psnr_s, psnr_best);

     %--- Start Plotting ---
     if PLOT && mod(iter, show_every) == 0
         % Clip to [0, 1]
         out_plt = min(max(out_num, 0), 1);
         out_avg_plt = min(max(out_avg_num, 0), 1);
         % Plot side by side
         fig = figure(1); clf
         set(fig, 'Visible', 'on');
         subplot(1,2,1);
         imagesc(out_plt); colormap gray; axis image off;

         title(sprintf('Output (Iter %d; PSNR %.4f)', iter,psnr_inpaint));

         subplot(1,2,2);
         imagesc(out_avg_plt); colormap gray; axis image off;
         title('Smoothed output')
         drawnow
     end  
end 

% --- Restore best checkpoint before final inference ---
if ~isempty(best)
    net.Learnables.Value = best.learnables;
    out_avg = best.out_avg;
    net.State = best.State;
    fprintf('Using best checkpoint from iter %d (PSNR_LR %.4f)\n', best.iter, psnr_best);
end
if useGPU
    net = dlupdate(@gather, net);
end
%%
%[text] ## Save Results
% --- Save the Model and Artifacts ---
save('Inpainting_Removal_ModelWeights1.mat', 'net', 'net_input_saved', 'out_avg');
fprintf('Model and noise tensor saved to Inpainting_Removal_ModelWeights1.mat\n'); %[output:0fcef7fc]


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":27.8}
%---
%[output:0fcef7fc]
%   data: {"dataType":"text","outputData":{"text":"Model and noise tensor saved to Inpainting_Removal_ModelWeights1.mat\n","truncated":false}}
%---
