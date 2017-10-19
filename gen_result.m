function [  ] = gen_result( sport )
videodir = ['/media/data/mtriet/raw_video/', sport,'_eval/'];
framedir = ['../frame/'];
videonames = dir([videodir '*.mp4']);
num_class = containers.Map({'fb','bb'},[6,8]);
detfilename = ['tmp_', sport, '.txt'];
gtpath = ['../annotation/',sport,'/annotation_val/'];
subset ='val';
thres = (0.3:0.1:0.7);
result = zeros(length(thres),7);
%% segment, then propose for each video
for i = 1:length(videonames)
    tmp1 = strsplit(videonames(i).name,'.');
    videoname = tmp1{1};
    %% process switch
    if exist(['network_localization/seg_swin_', videoname ,'.mat'], 'file')
        load(['network_localization/seg_swin_', videoname ,'.mat']);
        load(['network_localization/prob_', videoname ,'.mat']);
        cd('network_localization/');
    else
        seg_swin = process_video(videonames(i).name, sport);
    end
    %% post-processing
    fprintf(['post-processing starts\n']);
    tic;
    seg_swin=seg_swin(seg_swin(:,11)~=0,:);
    cd('..');
    % refine score via window length weights
    load(['window_weight/',sport,'_weight.mat']);
    for i=1:length(seg_swin)        
        seg_swin(i,9)=seg_swin(i,9).*weight(log2(seg_swin(i,2)/16)+1,seg_swin(i,11));
    end    
    
    cd('eval');
    for j = 1:length(thres)
        % ===============================
        % NMS after detection - per video
        threshold = thres(j)
%         overlap_nms = threshold - 0.1;
%         pick_nms = [];
%         for cls=1:num_class(sport)
%             inputpick = find((seg_swin(:,11)==cls));
%             pick_nms = [pick_nms; inputpick(nms_temporal([seg_swin(inputpick,5), seg_swin(inputpick,6),seg_swin(inputpick,9)],overlap_nms))]; 
%         end
%        seg_swin = seg_swin(pick_nms,:);
        if isempty(seg_swin)
            break;
        end
        % rank score by overlap score
        [~,order] = sort(-seg_swin(:,9));
        seg_swin = seg_swin(order,:);
        fout = fopen(detfilename ,'w');
        for i=1:size(seg_swin,1)
            fprintf(fout,['video_test_' num2str(seg_swin(i,1),'%07d') ' ' num2str(seg_swin(i,5),'%.1f') ' ' num2str(seg_swin(i,6),'%.1f') ' ' num2str(seg_swin(i,11)) ' ' num2str(seg_swin(i,9)) ' ' '\n']);
        end
        
        res = zeros(num_class(sport) ,length(dir([framedir videoname '/']))-2);        
        for i=size(seg_swin,1):-1:1
           res(seg_swin(i,11),seg_swin(i,3):seg_swin(i,4)) = seg_swin(i,9);           
        end
        
        save(['../final/seg_swin_', videoname ,'_',num2str(thres(j)),'.mat'],'seg_swin','res');
        fclose(fout);
        fprintf(['post-processing done in ' num2str(toc) ' s\nEvaluation start!']);        
        %% evaluation
        [true_p, false_p, gt_0match, gt_instance, prop_instance] = huaweievalProposal(detfilename,gtpath,subset, threshold);        
        %score = struct('tp', 0, 'fp', 0, 'gt_0match', 0, 'gt_i', 0, 'prop_i', 0);        
        result(j,1) = result(j,1) + true_p;
        result(j,2) = result(j,2) + false_p;
        result(j,3) = result(j,3) + gt_0match;
        result(j,4) = result(j,4) + gt_instance;
        result(j,5) = result(j,5) + prop_instance;
        fprintf('True positive: num of proposal that has overlaps > %f: %f\n', threshold, true_p);
        fprintf('False positive: num of proposal that has sum of overlaps score < 0.5: %f %d\n', threshold, false_p);
        fprintf('Num of ground truth that does not have any match: %d\n', gt_0match);
        fprintf('GT instances: %d\nProposal instances: %d\n', gt_instance, prop_instance);

        fprintf('Precision 1: %f\n', true_p/prop_instance);
        fprintf('Precision 2: %f\n', true_p/(true_p+false_p));        
    end
    cd('..');
end
disp('   TP     FP       Precision');
for i = 1:length(thres)
    result(i, 6) = result(i,1) / (result(i,1) + result(i,2));
    disp([result(i,1), result(i,2), result(i,6)]);
end
save('result.mat', 'result');
