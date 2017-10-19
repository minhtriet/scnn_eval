import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sys
import pdb
import json

MAP_PATH = "/media/data/mtriet/dataset/script"

gt = scipy.io.loadmat('ground_truth/%s.mat' % sys.argv[1])
y = gt['gt']
x = np.arange(0, 1.2, 1)

if sys.argv[2] == 'bb':
  with open("%s/bb_classes.json" % MAP_PATH) as data_file:
    sport_map = json.load(data_file)
    sport_map = dict((v,k) for k,v in sport_map.iteritems())
  with open("%s/bb_color.json" % MAP_PATH) as data_file:
    color_map = json.load(data_file)
elif sys.argv[2] == 'fb':
  with open("%s/fb_classes.json" % MAP_PATH) as data_file:
    sport_map = json.load(data_file)
    sport_map = dict((v,k) for k,v in sport_map.iteritems())
  with open("%s/fb_color.json" % MAP_PATH) as data_file:
    color_map = json.load(data_file)
else:
  print "draw_graph video_without_extension <fb/bb>"
  sys.exit(0)

fig, axes = plt.subplots(6)
y = [i[0] for i in y[0]]
axes[0].set_ylim([0, 1])
axes[0].set_xlim([0, len(y)])
for i in range(len(y)-1):
  axes[0].fill_betweenx(x, i, i+1, color=color_map[y[i]])

axes[0].set_ylabel('Class')
axes[0].set_xlabel('Ground truth')

# load threshold
for index, value in enumerate(np.linspace(0.3, 0.7, num=5)):
  axes[index+1].set_ylim([0, 1])
  axes[index+1].set_xlim([0, len(y)])
  print('Read file final/seg_swin_%s_%1.1f.mat' % (sys.argv[1], value))
  gt = scipy.io.loadmat('final/seg_swin_%s_%1.1f.mat' % (sys.argv[1], value))
  gt = gt['res']
  y = gt.T
  for i in range(len(y)-1):
    for klass in range(len(color_map)): 
      axes[index+1].fill_betweenx(x, i, i+1, color=color_map[ sport_map[klass] ])
    
  axes[index+1].set_ylabel('Class')
  axes[index+1].set_xlabel('Threshold: %f' % value)

plt.savefig('fig.png')
