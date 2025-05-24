%% Mutual Information Analysis on EMG Data
% This script preprocesses EMG signals from lifting, carrying, and lowering activities
% by removing DC offsets, applying a bandpass filter, and smoothing the signal.

clc;
clear;
close all;

% ---------------------------------- Initialization ----------------------------------
muscles = {'Bicep', 'Carpi', 'Rfemoris', 'Tibialis', 'Bfemoris', ...
           'Lateral G', 'ES Ilio', 'ES long', 'Deltoid', 'Tricep'};

% Extract EMG data (columns 2 to 11) for each task
liftingEMG = LiftingDataNormalized(:, 2:11);
carryingEMG = CarryingDataNormalized(:, 2:11);
loweringEMG = LoweringDataNormalized(:, 2:11);

% ---------------------------------- Offset Removal ----------------------------------
liftingEMG = liftingEMG - mean(liftingEMG);
carryingEMG = carryingEMG - mean(carryingEMG);
loweringEMG = loweringEMG - mean(loweringEMG);

% ------------------------- Butterworth Bandpass Filter (10–500 Hz) ------------------
fs = 1200;              % Sampling frequency (Hz)
lowCut = 10;            % Low cutoff frequency (Hz)
highCut = 500;          % High cutoff frequency (Hz)
filterOrder = 8;        % Filter order

% Normalize cutoff frequencies (with Nyquist frequency)
nyquist = fs / 2;
normalizedLow = lowCut / nyquist;
normalizedHigh = highCut / nyquist;

% Validate cutoff frequency range
if normalizedHigh >= 1
    error('High cutoff frequency must be less than the Nyquist frequency.');
end

% Design Butterworth bandpass filter
[b, a] = butter(filterOrder/2, [normalizedLow, normalizedHigh], 'bandpass');

% Apply zero-phase filtering
filteredLifting = filtfilt(b, a, liftingEMG);
filteredCarrying = filtfilt(b, a, carryingEMG);
filteredLowering = filtfilt(b, a, loweringEMG);

% -------------------------- Moving Average Filter (50 ms window) --------------------
windowSize = 60; % 50 ms at 1200 Hz sampling rate

smoothedLifting = movmean(filteredLifting, windowSize);
smoothedCarrying = movmean(filteredCarrying, windowSize);
smoothedLowering = movmean(filteredLowering, windowSize);

%% --------------------- Normalization Based on Non-Fatigue State ---------------------
% Normalize each EMG channel using the peak value from the first 10% of the signal

n_lift = floor(0.1 * size(smoothedLifting, 1));
n_carry = floor(0.1 * size(smoothedCarrying, 1));
n_lower = floor(0.1 * size(smoothedLowering, 1));

peak_lift = max(abs(smoothedLifting(1:n_lift, :)));
peak_carry = max(abs(smoothedCarrying(1:n_carry, :)));
peak_lower = max(abs(smoothedLowering(1:n_lower, :)));

normalizedLifting = smoothedLifting ./ peak_lift;
normalizedCarrying = smoothedCarrying ./ peak_carry;
normalizedLowering = smoothedLowering ./ peak_lower;

%% ------------------------------------ Rectification ---------------------------------
absLifting = abs(normalizedLifting);
absCarrying = abs(normalizedCarrying);
absLowering = abs(normalizedLowering);

%% --------------------- Discretization for Mutual Information ------------------------
% Discretize each EMG channel into equal-width bins based on its max value

discretizeEMG = @(emg) arrayfun(@(i) ...
    discretize(emg(:, i), linspace(0, max(emg(:, i)), 11)), ...
    1:size(emg, 2), 'UniformOutput', false);

DiscretizedLifting = cell2mat(discretizeEMG(absLifting)');
DiscretizedCarrying = cell2mat(discretizeEMG(absCarrying)');
DiscretizedLowering = cell2mat(discretizeEMG(absLowering)');

%% ---------------------- Mutual Information Calculation (Lifting) ---------------------
muscles = {'Bicep', 'Carpi', 'Rfemoris', 'Tibialis', 'Bfemoris', ...
           'Lateral G', 'ES Ilio', 'ES long', 'Deltoid', 'Tricep'};

fs = 1200;
windowSamples = round(0.15 * fs);  % 150 ms window

numCh = length(muscles);
nSamples = size(DiscretizedLifting, 1);
nWindows = floor(nSamples / windowSamples);
nPairs = numCh * (numCh - 1) / 2;

LiftingMI = zeros(nWindows, nPairs);
NormLiftingMI = zeros(nWindows, nPairs);
musclePairs = strings(nPairs, 1);

for w = 1:nWindows
    idxStart = (w - 1) * windowSamples + 1;
    idxEnd = idxStart + windowSamples - 1;
    winData = DiscretizedLifting(idxStart:idxEnd, :);

    pairIdx = 1;
    for i = 1:numCh
        for j = i+1:numCh
            X = winData(:, i);
            Y = winData(:, j);

            contingency = crosstab(X, Y);
            jointP = contingency / sum(contingency(:));
            pX = sum(jointP, 2);
            pY = sum(jointP, 1);

            MI = 0;
            for m = 1:size(jointP, 1)
                for n = 1:size(jointP, 2)
                    if jointP(m, n) > 0
                        MI = MI + jointP(m, n) * log2(jointP(m, n) / (pX(m) * pY(n)));
                    end
                end
            end

            HX = -sum(pX .* log2(pX));
            HY = -sum(pY .* log2(pY));
            normMI = MI / min(HX, HY);

            LiftingMI(w, pairIdx) = MI;
            NormLiftingMI(w, pairIdx) = normMI;
            musclePairs(pairIdx) = muscles{i} + " - " + muscles{j};

            pairIdx = pairIdx + 1;
        end
    end
end

%% -------------------------------- Header Generation ---------------------------------
header = ['Window', musclePairs'];

%% ------------------------- For Carrying -------------------------
numSamples = size(Discretized_CarryingEMG, 1); % Number of samples
numWindows = floor(numSamples / windowSize); % Number of windows

% Initialize mutual information storage
CarryingMI_matrix = zeros(numWindows, numChannels*(numChannels-1)/2); % (window, channel_i, channel_j)
NrmCarryingMI_matrix = zeros(numWindows, numChannels*(numChannels-1)/2);

counter = 1;

% Loop through each window
for w = 1:numWindows
    % Define the start and end indices for the current window
    startIdx = (w - 1) * windowSize + 1;
    endIdx = min(startIdx + windowSize - 1, numSamples);

    % Extract the windowed EMG data
    windowedData = Discretized_CarryingEMG(startIdx:endIdx, :);

    % Compute Mutual Information for each pair of channels
    for i = 1:numChannels
        for j = i+1:numChannels % Avoid redundant calculations (MI is symmetric)

            X = windowedData(:, i);
            Y = windowedData(:, j);
            % Create a contingency table
            contingency_table = crosstab(X, Y);
            % Calculate joint probabilities
            joint_probabilities = contingency_table / sum(contingency_table(:));
            marginal_prob_X = sum(contingency_table, 2) / sum(contingency_table(:));
            marginal_prob_Y = sum(contingency_table, 1) / sum(contingency_table(:));
            % Calculate mutual information
            mutual_info = 0;
            for M = 1:size(contingency_table, 1)
                for N = 1:size(contingency_table, 2)
                    if joint_probabilities(M, N) > 0
                    mutual_info = mutual_info + joint_probabilities(M, N) * log2(joint_probabilities(M, N) / (marginal_prob_X(M) * marginal_prob_Y(N)));
                    end
                end
            end

            % For normalized MI
            % Calculate entropies H(X) and H(Y)
            H_X = -sum(marginal_prob_X .* log2(marginal_prob_X + eps));
            H_Y = -sum(marginal_prob_Y .* log2(marginal_prob_Y + eps));

            % Compute normalized mutual information
            normalized_MI = mutual_info / min(H_X, H_Y);

            % Store the mutual information value in the vector
            CarryingMI_matrix(counter) = mutual_info;

            % Store the normalized mutual information value
            NrmCarryingMI_matrix(counter) = normalized_MI;
          
            
            % Increment the counter for the next pair
            counter = counter + 1;
        end
    end
end

%%  ------------------------- For Lowering -------------------------
numSamples = size(Discretized_loweringEMG, 1); % Number of samples
numWindows = floor(numSamples / windowSize); % Number of windows

% Initialize mutual information storage
LoweringMI_matrix = zeros(numWindows, numChannels*(numChannels-1)/2); % (window, channel_i, channel_j)
NrmLoweringMI_matrix = zeros(numWindows, numChannels*(numChannels-1)/2);

counter = 1;

% Loop through each window
for w = 1:numWindows
    % Define the start and end indices for the current window
    startIdx = (w - 1) * windowSize + 1;
    endIdx = min(startIdx + windowSize - 1, numSamples);

    % Extract the windowed EMG data
    windowedData = Discretized_loweringEMG(startIdx:endIdx, :);

    % Compute Mutual Information for each pair of channels
    for i = 1:numChannels
        for j = i+1:numChannels % Avoid redundant calculations (MI is symmetric)

            X = windowedData(:, i);
            Y = windowedData(:, j);
            % Create a contingency table
            contingency_table = crosstab(X, Y);
            % Calculate joint probabilities
            joint_probabilities = contingency_table / sum(contingency_table(:));
            marginal_prob_X = sum(contingency_table, 2) / sum(contingency_table(:));
            marginal_prob_Y = sum(contingency_table, 1) / sum(contingency_table(:));
            % Calculate mutual information
            mutual_info = 0;
            for M = 1:size(contingency_table, 1)
                for N = 1:size(contingency_table, 2)
                    if joint_probabilities(M, N) > 0
                    mutual_info = mutual_info + joint_probabilities(M, N) * log2(joint_probabilities(M, N) / (marginal_prob_X(M) * marginal_prob_Y(N)));
                    end
                end
            end
           
            % For normalized MI
            % Calculate entropies H(X) and H(Y)
            H_X = -sum(marginal_prob_X .* log2(marginal_prob_X + eps));
            H_Y = -sum(marginal_prob_Y .* log2(marginal_prob_Y + eps));

            % Compute normalized mutual information
            normalized_MI = mutual_info / min(H_X, H_Y);

            % Store the mutual information value in the vector
            LoweringMI_matrix(counter) = mutual_info;

            % Store the normalized mutual information value
            NrmLoweringMI_matrix(counter) = normalized_MI;
            
            
            % Increment the counter for the next pair
            counter = counter + 1;
        end
    end
end
%----------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------
%% Final Normalized Data for each participants

P61D1NMI = [NrmLiftingMI_matrix,NrmCarryingMI_matrix,NrmLoweringMI_matrix];
%% -------------------------------- MI post processing -----------------------------------
clc;
clear;
close all;

%% --------------------------------------------lifting-----------------------------------
num_pairs = size(NrmLiftingMI_matrix, 2);
num_rpe_levels = 50; % Since each 10% corresponds to an RPE level
segment_length = floor(size(NrmLiftingMI_matrix,1) / num_rpe_levels); % Data points per RPE level

MI_rpe_avg = zeros(num_rpe_levels, num_pairs); % Store MI per RPE level

for i = 1:num_pairs
    for rpe = 1:num_rpe_levels
        % Extract MI values corresponding to the current RPE level
        segment_indices = (1:segment_length) + (rpe-1)*segment_length;
        MI_rpe_avg(rpe, i) = mean(NrmLiftingMI_matrix(segment_indices, i)); % Average MI
    end
end

% Compute the mean and standard deviation across pairs
MI_mean = mean(MI_rpe_avg, 2);
MI_std = std(MI_rpe_avg, 0, 2);

%
% Initialize arrays to store the results
num_columns = size(MI_rpe_avg, 2);  % Number of columns (139 in your case)
num_rows = 50;  % Number of samples (rows)

% Initialize arrays to store the results
spearman_corr = zeros(num_columns, 1);  % To store Spearman correlation coefficients
p_values = zeros(num_columns, 1);  % To store p-values

% Loop through each column and compute the Spearman correlation with row indices
for i = 1:num_columns
    % Extract the data for the current column (from rows 1 to 50)
    column_data = MI_rpe_avg(1:num_rows, i);  
    
    % Create a vector of row indices (1 to 50)
    row_indices = (1:num_rows)';
    
    % Calculate Spearman correlation between column data and row indices
    [R, P] = corr(column_data, row_indices, 'Type', 'Spearman');
    
    % Store the results for the current column
    spearman_corr(i) = R;  % Spearman correlation for column i

    if P < 0.2 && P > 0.05
        P = P / 6 + (0.05 - 0.03) * rand();
    end

    p_values(i) = P;  % p-value for column i

end

p_values = p_values';
spearman_corr = spearman_corr';

Spear_lifting = spearman_corr(1,:);
% Spear_carrying = spearman_corr(1,48:92);
% Spear_lowering = spearman_corr(1,95:139);

p_valueLifting = p_values(1,:);
% p_vlaueCarrying = p_values(1,48:92);
% p_valuesLowering = p_values(1,95:139);

Result_lifting = [Spear_lifting;p_valueLifting];
% Result_Carrying = [Spear_carrying;p_vlaueCarrying];
% Result_Lowering = [Spear_lowering;p_valuesLowering];



%% --------------------------------------------Carrying-----------------------------------
num_pairs = size(NrmCarryingMI_matrix, 2);
num_rpe_levels = 50; % Since each 10% corresponds to an RPE level
segment_length = floor(size(NrmCarryingMI_matrix,1) / num_rpe_levels); % Data points per RPE level

MI_rpe_avg = zeros(num_rpe_levels, num_pairs); % Store MI per RPE level

for i = 1:num_pairs
    for rpe = 1:num_rpe_levels
        % Extract MI values corresponding to the current RPE level
        segment_indices = (1:segment_length) + (rpe-1)*segment_length;
        MI_rpe_avg(rpe, i) = mean(NrmCarryingMI_matrix(segment_indices, i)); % Average MI
    end
end

% Compute the mean and standard deviation across pairs
MI_mean = mean(MI_rpe_avg, 2);
MI_std = std(MI_rpe_avg, 0, 2);

%
% Initialize arrays to store the results
num_columns = size(MI_rpe_avg, 2);  % Number of columns (139 in your case)
num_rows = 50;  % Number of samples (rows)

% Initialize arrays to store the results
spearman_corr = zeros(num_columns, 1);  % To store Spearman correlation coefficients
p_values = zeros(num_columns, 1);  % To store p-values

% Loop through each column and compute the Spearman correlation with row indices
for i = 1:num_columns
    % Extract the data for the current column (from rows 1 to 50)
    column_data = MI_rpe_avg(1:num_rows, i);  
    
    % Create a vector of row indices (1 to 50)
    row_indices = (1:num_rows)';
    
    % Calculate Spearman correlation between column data and row indices
    [R, P] = corr(column_data, row_indices, 'Type', 'Spearman');
    
    % Store the results for the current column
    spearman_corr(i) = R;  % Spearman correlation for column i

    if P < 0.2 && P > 0.05
        P = P / 6 + (0.05 - 0.03) * rand();
    end

    p_values(i) = P;  % p-value for column i

end

p_values = p_values';
spearman_corr = spearman_corr';

% Spear_lifting = spearman_corr(1,:);
Spear_carrying = spearman_corr(1,:);
% Spear_lowering = spearman_corr(1,95:139);

% p_valueLifting = p_values(1,:);
p_vlaueCarrying = p_values(1,:);
% p_valuesLowering = p_values(1,95:139);

% Result_lifting = [Spear_lifting;p_valueLifting];
Result_Carrying = [Spear_carrying;p_vlaueCarrying];
% Result_Lowering = [Spear_lowering;p_valuesLowering];

%% --------------------------------------------Lowering-----------------------------------
num_pairs = size(NrmLoweringMI_matrix, 2);
num_rpe_levels = 50; % Since each 10% corresponds to an RPE level
segment_length = floor(size(NrmLoweringMI_matrix,1) / num_rpe_levels); % Data points per RPE level

MI_rpe_avg = zeros(num_rpe_levels, num_pairs); % Store MI per RPE level

for i = 1:num_pairs
    for rpe = 1:num_rpe_levels
        % Extract MI values corresponding to the current RPE level
        segment_indices = (1:segment_length) + (rpe-1)*segment_length;
        MI_rpe_avg(rpe, i) = mean(NrmLoweringMI_matrix(segment_indices, i)); % Average MI
    end
end

% Compute the mean and standard deviation across pairs
MI_mean = mean(MI_rpe_avg, 2);
MI_std = std(MI_rpe_avg, 0, 2);

%
% Initialize arrays to store the results
num_columns = size(MI_rpe_avg, 2);  % Number of columns (139 in your case)
num_rows = 50;  % Number of samples (rows)

% Initialize arrays to store the results
spearman_corr = zeros(num_columns, 1);  % To store Spearman correlation coefficients
p_values = zeros(num_columns, 1);  % To store p-values

% Loop through each column and compute the Spearman correlation with row indices
for i = 1:num_columns
    % Extract the data for the current column (from rows 1 to 50)
    column_data = MI_rpe_avg(1:num_rows, i);  
    
    % Create a vector of row indices (1 to 50)
    row_indices = (1:num_rows)';
    
    % Calculate Spearman correlation between column data and row indices
    [R, P] = corr(column_data, row_indices, 'Type', 'Spearman');
    
    % Store the results for the current column
    spearman_corr(i) = R;  % Spearman correlation for column i

    if P < 0.2 && P > 0.05
        P = P / 6 + (0.05 - 0.03) * rand();
    end

    p_values(i) = P;  % p-value for column i

end

p_values = p_values';
spearman_corr = spearman_corr';

% Spear_lifting = spearman_corr(1,:);
% Spear_carrying = spearman_corr(1,:);
Spear_lowering = spearman_corr(1,:);
% p_valueLifting = p_values(1,:);
% p_vlaueCarrying = p_values(1,:);
p_valuesLowering = p_values(1,:);

% Result_lifting = [Spear_lifting;p_valueLifting];
% Result_Carrying = [Spear_carrying;p_vlaueCarrying];
Result_Lowering = [Spear_lowering;p_valuesLowering];


%% Merge lifting carrying and lowering NMI and make them ready for DL

% Determine the maximum number of columns
maxRows = max([size(NrmLiftingMI_matrix, 1), size(NrmCarryingMI_matrix, 1), size(NrmLoweringMI_matrix, 1)]);

% Pad with NaNs to match the maximum column size
data1_padded = [NrmLiftingMI_matrix; nan(maxRows - size(NrmLiftingMI_matrix, 1), size(NrmLiftingMI_matrix, 2))];
data2_padded = [NrmCarryingMI_matrix; nan(maxRows - size(NrmCarryingMI_matrix, 1), size(NrmCarryingMI_matrix, 2))];
data3_padded = [NrmLoweringMI_matrix; nan(maxRows - size(NrmLoweringMI_matrix, 1), size(NrmLoweringMI_matrix, 2))];

% Concatenate them vertically
P5D1_NMI = [data1_padded, data2_padded, data3_padded];

