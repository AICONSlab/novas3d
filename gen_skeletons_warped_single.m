rng('shuffle');
rngState = rng;
deltaSeed = uint32(feature('getpid'));
seed = rngState.Seed + deltaSeed;
rng(seed);

in_dir = 'test_data/raw_warped_seg';
out_dir = 'test_data/raw_warped_seg';
in_suffix = '_seg_warped_single.mat';
out_suffix = '_skel_warped_single.mat';

files = dir(fullfile(in_dir,'*_seg_warped_single.mat'));
for k=randperm(length(files))
    basefilename = files(k).name;
    if  ~isfile(fullfile(in_dir,replace(basefilename,in_suffix,out_suffix)))
        file =  fullfile(in_dir,basefilename);
        load(file);
        FilteredImage = bwskel(imfill(imbinarize(FinalImage),'holes'),'MinBranchLength', 40);
        outputFileName = fullfile(out_dir,replace(basefilename,in_suffix,out_suffix));
        save(outputFileName,'FilteredImage','-mat')
    end
    %for K=1:length(FilteredImage(1, 1, :))
    %    imwrite(FilteredImage(:, :, K), outputFileName, 'WriteMode', 'append',  'Compression','none');
    %end
end

% run through bwareopen function to segment
