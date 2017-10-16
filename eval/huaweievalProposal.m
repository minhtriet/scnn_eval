function [true_p, false_p, gt_nomatch, gt_instance, prop_instance]=huaweievalProposal(detfilename,gtpath,subset, threshold)

[classids,classnames]=textread([gtpath '/detclasslist.txt'],'%d%s');
  
% read ground truth
clear gtevents
gteventscount=0;

for i=1:length(classnames) 
  class=classnames{i};
  gtfilename=[gtpath '/' class '_' subset '.txt'];
  if exist(gtfilename,'file')~=2
    error(['evaldet: Could not find GT file ' gtfilename])
  end
  [videonames,t1,t2]=textread(gtfilename,'%s%f%f');
  for j=1:length(videonames)
    gteventscount=gteventscount+1;
    vid_name = strsplit(strjoin(videonames(j)), '_');
    vid_name = strjoin(vid_name(end));
    vid_name = str2num(vid_name);
    vid_name = sprintf('%07d', vid_name);
    gtevents(gteventscount).videoname= ['video_test_', vid_name];
    gtevents(gteventscount).timeinterval=[t1(j) t2(j)];    
    gtevents(gteventscount).conf=1;
  end
end


% parse detection results
%

if exist(detfilename,'file')~=2
  error(['evaldet: Could not find file ' detfilename])
end

[videonames,t1,t2,clsid,conf]=textread(detfilename,'%s%f%f%d%f');
videonames=regexprep(videonames,'\.mp4','');

clear detevents
for i=1:length(videonames)  
    detevents(i).videoname=videonames{i};
    detevents(i).timeinterval=[t1(i) t2(i)];     
end

% Evaluate per-class PR for multiple overlap thresholds
name = detevents(1).videoname;
mask=arrayfun(@(a) all(a.videoname==name), gtevents);
ov=intervaloverlapvalseconds(cat(1,gtevents(mask).timeinterval),cat(1,detevents(:).timeinterval));

true_p = sum(ov(:) > threshold);
false_p = sum( sum(ov,1)< threshold );
gt_nomatch = sum( sum(ov,2)<0.2 );
gt_instance = length(gtevents(mask)); 
prop_instance = length(detevents);
fprintf('Maximum: %f\n',max(ov(:)));
