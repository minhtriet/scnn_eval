% ----------------------------------------------------------------------------------------------------------------
% Segment-CNN
% Copyright (c) 2016 , Digital Video & Multimedia (DVMM) Laboratory at Columbia University in the City of New York.
% Licensed under The MIT License [see LICENSE for details]
% Written by Zheng Shou, Dongang Wang, and Shih-Fu Chang.
% ----------------------------------------------------------------------------------------------------------------
function [  ] = cal_weight( sport )

load([sport, '_class.mat']);
filepath = ['../annotation/',sport,'/annotation_val/'];  % script is written for football at /dataset/script
window_length = [0.6, 1.24, 2.52, 5.08, 10.20, 22.44];
window_cnt = zeros(6,length(classid));

for i = 2:length(classid)
    f = fopen([filepath, classid{i}, '_val.txt']);    
    while ~feof(f)
        line = fgetl(f);
        line = regexp(line,' ','split');
        delta = str2double(line{3})-str2double(line{2});
        for j = 1:6
            if delta>=(window_length(j)/2) && delta <= (window_length(j)*2)
                window_cnt(j,i) = window_cnt(j,i)+1;
            end
        end
    end
end

weight = window_cnt./repmat(sum(window_cnt,1),6,1);
save([sport,'_weight.mat'],'weight');
save([sport,'_count.mat'],'window_cnt');



