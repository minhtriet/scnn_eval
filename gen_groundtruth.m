function [ ] = gen_groundtruth(sport)
videodir = ['/media/data/mtriet/raw_video/', sport,'_eval/'];
framedir = 'frame/';
subtitles = dir([videodir '*.aqt']);

for i = 1:length(subtitles)
    name = subtitles(i).name;
    name = strsplit(name, '.');
    name = name{1};
    gt = cell(1, length(dir([framedir name '/']))-2);
    gt(:) = {'bg'};
    [class, time] = textread([videodir name '.aqt'],'%s%d');    
    for line = 1:3:length(class)
        if ~( strcmp(strjoin(class(line+1)), 'bg') )
            if strcmp(strjoin(class(line+1)), 'fkwg') || strcmp(strjoin(class(line+1)), 'fkwog') 
                gt(time(line):time(line+2)) = {'fk'};
            else
                gt(time(line):time(line+2)) = class(line+1);
            end
        end        
    end
    save(['ground_truth/' name '.mat'], 'gt');
end

end