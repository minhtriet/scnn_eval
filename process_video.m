function [ seg_swin ] = process_video( videoname, sport )

THRESHOLD = 0.7;

videodir = ['/media/data/mtriet/raw_video/', sport,'_eval/'];
framedir = 'frame/';
num_class = containers.Map({'fb','bb'},[6,8]);
addpath(genpath('../../lib/'));
framerate = 25;
tmp1 = strsplit(videoname,'.');
videoname = tmp1{1};
videotype = tmp1{2};

if exist([framedir videoname])
    %system(['rm -R ' framedir videoname]);
end
system(['mkdir ' framedir videoname]);
if exist([ 'network_localization/output'])
    system('rm -R network_localization/output');
end
system(['mkdir network_localization/output']);
if exist('network_proposal/output')
    system('rm -R network_proposal/output');
end
system('mkdir network_proposal/output');

if exist([ 'final'])
    system('rm -R final');
end
system('mkdir final');

%% frame extract
fprintf(['frame extract starts']);
tic;
cmd = ['../../lib/preprocess/ffmpeg -i ' videodir videoname '.' videotype ' -r ' num2str(framerate) ...
	' -f image2 ' framedir videoname '/' '%06d.jpg 2>' framedir 'frame_extract.log'];
%system(cmd);

fprintf(['frame extract done in ' num2str(toc) ' s\n']);

%% init sliding window
% 1:video_name 2:frame_size_type 3:start_frame 4:end_frame 5:start_time 6:end_time 12:win_overlap_rate
fprintf(['init sliding window starts']);
tic;
seg_swin = zeros(0,12);
win_overlap_rate = 0.75;
img = dir([ framedir videoname '/*.jpg']);
for window_stride=[16,32,64,128,256,512]
    win_overlap = window_stride*(1-win_overlap_rate);
    start_frame = 1;
    end_frame = window_stride;
    while end_frame <= length(img)
        tmp = strsplit(videoname,'_');
        seg_swin(end+1,1) = str2num(tmp{end});
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
for i=1:length(seg_swin)
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
save('network_proposal/seg_swin.mat','seg_swin','-v7.3');

fprintf(['read proposal results done in ' num2str(toc) ' s\n']);

%% generate localization list
fprintf(['generate localization list starts\n']);
tic;
seg_swin = seg_swin(seg_swin(:,10)>=THRESHOLD,:);

fout3 = fopen('network_localization/test_prefix_localization.lst','w'); 
fout4 = fopen('network_localization/test_uniform16_localization.lst','w');
for i=1:length(seg_swin)
    fprintf(fout3,['network_localization/output/' num2str(i,'%06d') '\n']);
    fprintf(fout4,[framedir videoname '/ ' num2str(seg_swin(i,3)) ' 0 ' num2str(seg_swin(i,2)/16) ' 0\n']);            
end
fclose(fout3);
fclose(fout4);

fprintf(['generate localization list done in ' num2str(toc) ' s\n']);

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
[a,b] = max(prob(:,:)');
seg_swin(:,9) = a;
seg_swin(:,11) = b-1;
save(['network_localization/prob_', videoname, '.mat'],'prob','-v7.3');
save(['network_localization/seg_swin_', videoname, '.mat'],'seg_swin','-v7.3');

fprintf(['read localization results done in ' num2str(toc) ' s\n']);

cd('eval');

end

