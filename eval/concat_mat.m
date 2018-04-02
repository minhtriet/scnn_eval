function concat_mat(path, output)
  if nargin < 2
      disp('concat_mat path output_file');
      disp('concat_mat /media/data/mtriet/scnn/experiments/huawei_c3d1.0/network_proposal/*.mat fb');
      return
  end
  
  mat_files = dir(path);
  if size(mat_files, 1) == 0
    disp('No mat files found');
    return
  end
  base_path = fileparts(path);
  seg_swin = [];
  gt = [];
  
  for i=(1:length(mat_files))
    variable = matfile([base_path '/' mat_files(i).name]);
    try
      variable = variable.seg_swin;
      seg_swin = [seg_swin, variable];  % concat first variable of the matfile
    catch
      try
        variable = variable.gt;
        gt = [gt, variable];  % concat first variable of the matfile
      catch
        seg_swin = [seg_swin; variable];
      end
    end
  end
  if isempty(seg_swin)
    gt = gt';
    save([output '.mat'], 'gt');
  else
    seg_swin = seg_swin';
    save([output '.mat'], 'seg_swin');
  end
end
