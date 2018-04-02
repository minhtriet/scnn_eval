% process_video('/media/data/mtriet/dataset/fb_frames_eval_split/', 'Brazil_Spain_002_000', 'fb', 1)
function [ seg_swin ] = process_video(framedir, videoname, sport, split )
THRESHOLD = 0.7;
num_class = containers.Map({'fb','bb'},[6,8]);
addpath(genpath('../../lib/'));
framerate = 25;

if ~exist([framedir videoname])
    error([framedir videoname ' not existed']);
end
% system(['mkdir ' framedir videoname]);
if exist([ 'network_localization/output'])
    system('rm -R network_localization/output');
end
system(['mkdir network_localization/output']);
if exist('network_proposal/output')
    system('rm -R network_proposal/output');
end
system('mkdir network_proposal/output');

%% init sliding window
% 1:video_name 2:frame_size_type 3:start_frame 4:end_frame 5:start_time 6:end_time 12:win_overlap_rate
fprintf(['init sliding window starts']);
tic;
seg_swin = zeros(0,12);
win_overlap_rate = 0.75;
img = dir([ framedir videoname '/*.jpg']);
num_fr = length(img);
for window_stride=[16,32,64,128,256,512]
    win_overlap = window_stride*(1-win_overlap_rate);
    start_frame = 1;
    end_frame = window_stride;
    while end_frame <= length(img)       
        tmp = strsplit(videoname,'_');
        if split 
          seg_swin(end+1,1) = str2num(tmp{end-1});
        else
          seg_swin(end+1,1) = str2num(tmp{end});
        end
        seg_swin(end,2) = window_stride;
        seg_swin(end,3) = start_frame;
        seg_swin(end,4) = end_frame;
        seg_swin(end,5) = start_frame/framerate;
        seg_swin(end,6) = end_frame/framerate;
        seg_swin(end,12) = 1-win_overlap_rate;   
        % next
        start_frame = start_frame + win_overlap;
        end_frame = end_frame + win_overlap;
    end
end

fprintf(['init sliding window done in ' num2str(toc) ' s\n']);

%% generate proposal list
fprintf(['generate proposal starts']);
tic;
fout1 = fopen('network_proposal/test_prefix_proposal.lst','w');     
fout2 = fopen('network_proposal/test_uniform16_proposal.lst','w');
for i=1:size(seg_swin, 1)
    fprintf(fout1,['network_proposal/output/' num2str(i,'%06d') '\n']);
    fprintf(fout2,[framedir videoname '/ ' num2str(seg_swin(i,3)) ' 0 ' num2str(seg_swin(i,2)/16) '\n']);            
end
fclose(fout1);
fclose(fout2);

fprintf(['generate proposal list done in ' num2str(toc) ' s\n']);

%% run proposal network
fprintf(['run proposal network starts\n']);
tic;
system(['network_proposal/feature_extract.sh ', sport]);
fprintf(['run proposal network done in ' num2str(toc) ' s\n']);

%% read proposal results
fprintf(['read proposal results starts']);
tic;
prob = zeros(size(seg_swin,1),2);
img = dir( ['network_proposal/output/'] ); % be careful whether all are jpg
for img_index = 3:size(img,1)
    [~,prob(img_index-2,:)] = read_binary_blob([ 'network_proposal/output/' img(img_index).name]);
end
seg_swin(:,10) = prob(:,2);

save(['network_proposal/seg_swin_' videoname '.mat'],'seg_swin','-v7.3');

fprintf(['read proposal results done in ' num2str(toc) ' s\n']);

%% generate localization list
fprintf(['generate localization list starts\n']);
tic;
seg_swin = seg_swin(seg_swin(:,10)>=THRESHOLD,:);

fout3 = fopen('network_localization/test_prefix_localization.lst','w'); 
fout4 = fopen('network_localization/test_uniform16_localization.lst','w');
for i=1:size(seg_swin, 1)
    fprintf(fout3,['network_localization/output/' num2str(i,'%06d') '\n']);
    fprintf(fout4,[framedir videoname '/ ' num2str(seg_swin(i,3)) ' 0 ' num2str(seg_swin(i,2)/16) ' 0\n']);            
end
fclose(fout3);
fclose(fout4);

fprintf(['generate localization list done in ' num2str(toc) ' s\n']);

res = zeros( num_class(sport), num_fr );

if isempty(seg_swin)
  % early exit   
  res(1, :) = 1;  
  save(['final/seg_swin_', videoname, '.mat'],'seg_swin','res','-v7.3');
  fprintf(['No action detected, exiting\n']);
  return;
end
%% run localization results
fprintf(['run localization results starts\n']);
tic;

system(['bash network_localization/feature_extract.sh ', sport]);
fprintf(['run localization results done in ' num2str(toc) ' s\n']);

%% read localization results
fprintf(['read localization results starts\n']);
tic;
prob = zeros(size(seg_swin,1), num_class(sport) );
img = dir( ['network_localization/output/'] ); % be careful whether all are jpg
for img_index = 3:size(img,1)
    [~,prob(img_index-2,:)] = read_binary_blob([ 'network_localization/output/' img(img_index).name]);
end

% reweight
load(['/media/data/mtriet/dataset/script/scnn/' sport '_freq.mat'])
freq=double(freq)/sum(freq);
for i=(1:size(prob,1))
  prob(i,:) = prob(i,:) ./ freq;
end
[a,b] = max(prob(:,:)');
seg_swin(:,9) = a;
seg_swin(:,11) = b-1;
seg_swin=seg_swin(seg_swin(:,11)~=0,:);
if isempty(seg_swin)
  % early exit   
  res(1, :) = 1;  
  save(['final/seg_swin_', videoname, '.mat'],'seg_swin','res','-v7.3');
  fprintf(['No action detected, exiting\n']);
  return;
end
for i=size(seg_swin,1):-1:1
  res(seg_swin(i,11),seg_swin(i,3):seg_swin(i,4)) = seg_swin(i,9);
end
if split
  save(['/media/data/mtriet/scnn/experiments/huawei_c3d1.0_split/final/', sport, '/prob_', videoname, '.mat'],'prob','-v7.3');
  save(['/media/data/mtriet/scnn/experiments/huawei_c3d1.0_split/final/', sport, '/seg_swin_', videoname, '.mat'],'seg_swin', 'res', '-v7.3');
else
  save(['network_localization/prob_', videoname, '.mat'],'prob','-v7.3');
  save(['network_localization/seg_swin_', videoname, '.mat'],'seg_swin','res','-v7.3');
end

fprintf(['read localization results done in ' num2str(toc) ' s\n']);

%cd('eval');
end
