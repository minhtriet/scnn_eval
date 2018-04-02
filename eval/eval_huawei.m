function [] = eval_huawei(sport)
AP_all=cell(0);
PR_all=cell(0);
mAP=[];
REC_all=[];
detfilename = ['tmp_', sport, '.txt'];
gtpath = ['../annotation/',sport,'/annotation_val/'];
subset = 'val';
num_class = containers.Map({'fb','bb'},[5,7]);
for threshold = 0.3:0.1:0.7

    load(['../final/' sport '/seg_swin.mat']);
    seg_swin=seg_swin';
    seg_swin=seg_swin(seg_swin(:,11)~=0,:);
        
    % adjust conf via the distribution of window length
    % no need since can be slo-mo video
    %load(['../window_weight/', sport, '_weight.mat']);    
    %for i=1:length(seg_swin)
    %    i
    %    keyboard 
    %    seg_swin(i,9)=seg_swin(i,9).*weight(log2(seg_swin(i,2)/16)+1,seg_swin(i,11));
    %end
    
	% ===============================
	% NMS after detection - per video
	overlap_nms = threshold - 0.1;
	videoid = unique(seg_swin(:,1));
	tic; 
	pick_nms = [];
	for id=videoid'
		for cls=1:num_class(sport)
			inputpick = find((seg_swin(:,1)==id)&(seg_swin(:,11)==cls));
			pick_nms = [pick_nms; inputpick(nms_temporal([seg_swin(inputpick,5) ...
				,seg_swin(inputpick,6),seg_swin(inputpick,9)],overlap_nms))]; 
		end
	end
	toc;

	% ===============================

	%% rank score by overlap score
	seg_swin = seg_swin([pick_nms],:);
	[~,order]=sort(-seg_swin(:,9));
	seg_swin = seg_swin(order,:);

	%% eval mAP
	numkeep = length(seg_swin);    
	fout = fopen(detfilename ,'w');
	eval_seg_swin = seg_swin(1:numkeep,:);  % use det score for computing
	
	for i=1:size(eval_seg_swin,1)
		fprintf(fout,['video_test_' num2str(eval_seg_swin(i,1),'%07d') ' ' num2str(eval_seg_swin(i,5),'%.1f') ' ' num2str(eval_seg_swin(i,6),'%.1f') ' ' num2str(eval_seg_swin(i,11)) ' ' num2str(eval_seg_swin(i,9)) ' ' '\n']);
	end
	fclose(fout);

    %[pr_all,ap_all,map] = huaweievalDet_woclass(detfilename,gtpath,subset,threshold);
    [pr_all,ap_all,map] = huaweievalDet(detfilename,gtpath,subset,threshold);
    mAP=[mAP,map];
    AP_all{end+1}=ap_all;
    PR_all{end+1}=pr_all;
    ave_rec = 0;
    for ii=1:num_class(sport)
      ave_rec = ave_rec + pr_all(ii).rec(end);
    end
    ave_rec = ave_rec/num_class(sport);
    REC_all=[REC_all,ave_rec];
    
    
end

save('res_scnn_thumos14.mat','PR_all','AP_all','REC_all','mAP');

