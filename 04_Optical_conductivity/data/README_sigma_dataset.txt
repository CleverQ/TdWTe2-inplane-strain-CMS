================ README_sigma_dataset ================
数据根目录：当前 repository 的 data/source_components/ 与 data/processed/
生成时间：2026-05-06 01:29:50

一、光电导率单位说明
------------------------------------------------------
当前脚本中用于显示与说明的光电导率单位为：S/cm
本脚本不会对输入 CSV 的数值做单位换算，输出文件中的单位与原始输入文件保持一致。

二、原始输入文件说明
------------------------------------------------------
每个工况目录中通常包含以下原始 CSV：
  S_xx.csv : 对称部分 S_{xx}(omega)
  S_yy.csv : 对称部分 S_{yy}(omega)
  S_xy.csv : 对称部分 S_{xy}(omega)
  S_yx.csv : 对称部分 S_{yx}(omega)，若不存在则脚本用 S_xy 代替
  A_xy.csv : 反对称部分 A_{xy}(omega)
  A_yx.csv : 反对称部分 A_{yx}(omega)，若不存在则脚本按 -A_xy 补全
  _stats_per_dir.csv : 提取阶段生成的局部统计信息，仅作参考，不参与核心组装

三、脚本在每个工况目录中生成的文件
------------------------------------------------------
  sigma_tensor_xy.csv
      统一频率轴上的复电导张量数据表，列为：
      omega_eV, Re_xx, Im_xx, Re_yy, Im_yy, Re_xy, Im_xy, Re_yx, Im_yx
      光电导率单位：S/cm

  sigma_tensor_xy.mat
      MATLAB 二进制文件，变量包括：
      omega_eV, sigma_xx, sigma_yy, sigma_xy, sigma_yx
      其中 sigma_ij 为复数，单位仍为：S/cm

  sigma_tensor_xy_quicklook.png
      快速检查图，展示 xx/yy/xy/yx 四个分量的 Re/Im 曲线。

四、grids 目录输出说明
------------------------------------------------------
grids 目录下会按分组写出：
  grids\all\...
  grids\ref\...
  grids\a\...
  grids\b\...
  以及若存在旧命名风格，还可能有 zigzag / armchair / default 等分组。

每个分组目录中包含：
  grids.mat
      汇总后的 MATLAB 数据文件，适合后续脚本直接读取。
      主要变量：eps_percent, case_ids, case_labels, direction, source_dir,
      omega_eV, E_eV, lambda_nm,
      Re_xx, Im_xx, Re_yy, Im_yy, Re_xy, Im_xy, Re_yx, Im_yx,
      rho, eta_signed
      其中 Re/Im 电导率分量单位：S/cm

  cases_metadata.csv
      每个工况的元数据表。

  grid_Re_xx.csv, grid_Im_xx.csv, ...
      第一列为 omega_eV，后续每一列对应一个工况 case_id。
      光电导率单位：S/cm

  grid_rho.csv
      rho = Re_xx / Re_yy（带数值保护）的网格数据，为无量纲量。

  grid_eta_signed.csv
      eta_signed = sign(Re_xy) * |sigma_xy| / sqrt(|sigma_xx||sigma_yy|)
      为无量纲量。

  summary_sigma_Re.png
      每个分组内所有工况的 Re(sigma_xx), Re(sigma_yy), Re(sigma_xy), Re(sigma_yx) 汇总折线图。

  summary_sigma_Im.png
      每个分组内所有工况的 Im(sigma_xx), Im(sigma_yy), Im(sigma_xy), Im(sigma_yx) 汇总折线图。

五、当前工况命名解析规则
------------------------------------------------------
  00_Optimized_Structure           -> direction=ref, eps=0
  Strain_a_Tension_0.1pct          -> direction=a,   eps=+0.1
  Strain_a_Compression_0.1pct      -> direction=a,   eps=-0.1
  Strain_b_Tension_0.2pct          -> direction=b,   eps=+0.2
  Strain_b_Compression_0.2pct      -> direction=b,   eps=-0.2

六、当前识别到的工况
------------------------------------------------------
  00_Optimized_Structure           | direction=ref      | eps_percent=  0.0000
  Strain_a_Tension_0.1pct          | direction=a        | eps_percent=  0.1000
  Strain_a_Tension_0.2pct          | direction=a        | eps_percent=  0.2000
  Strain_a_Tension_0.3pct          | direction=a        | eps_percent=  0.3000
  Strain_a_Tension_0.4pct          | direction=a        | eps_percent=  0.4000
  Strain_a_Tension_0.5pct          | direction=a        | eps_percent=  0.5000
  Strain_b_Tension_0.1pct          | direction=b        | eps_percent=  0.1000
  Strain_b_Tension_0.2pct          | direction=b        | eps_percent=  0.2000
  Strain_b_Tension_0.3pct          | direction=b        | eps_percent=  0.3000
  Strain_b_Tension_0.4pct          | direction=b        | eps_percent=  0.4000
  Strain_b_Tension_0.5pct          | direction=b        | eps_percent=  0.5000

======================================================
