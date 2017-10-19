function [ ] = draw_graph(video, sport)

fb = containers.Map( {'bg', 'gs', 'sot', 'pkwg', 'fk' 'pkwog'}, [1:6] );
bb = containers.Map( {'bg', 'ps', 'sd', 'bs', 'ao', 'fs', 'or', 'dr'}, [1:8] );

load(['ground_truth/' video '.mat'], 'gt');

if strcmp(sport, 'fb')
    sport_map = fb;
else
    sport_map = bb;
end

x = [1:length(gt)];
y = zeros(length(sport_map), length(gt));
for i=1:length(gt)
    y( i, fb( strjoin(gt(i)) ) ) = 1;
end
figure
area(y)
ylim([0 1.1])

end