function prepare_sigma_dataset(dataset_name, base_root)
% Public-repository copy: only path examples/defaults and public directory-name
% aliases were adapted. Tensor assembly, interpolation, units, and numerical
% algorithms are unchanged from the 2026-05-06 canonical local script.
% =====================================================================
% prepare_sigma_dataset
%
% 统一入口脚本：
%   1) 在指定数据总目录下自动寻找工况目录（存在 S_xx.csv + S_yy.csv）
%   2) 组装 sigma_tensor_xy.csv / .mat / quicklook
%   3) 自动汇总成 grids.mat / cases_metadata.csv / grid_*.csv
%   4) 额外写出 README_sigma_dataset.txt，说明输入/输出文件含义
%   5) 在每个 grids/<group>/ 目录下输出汇总折线图：
%        summary_sigma_Re.png
%        summary_sigma_Im.png
%
% 当前推荐目录风格（你现在正在用）：
%   optics_sigma_xy/
%       00_Optimized_Structure/
%           S_xx.csv
%           S_yy.csv
%           S_xy.csv
%           A_xy.csv
%           _stats_per_dir.csv
%
%       Strain_a_Tension_0.1pct/
%       Strain_a_Tension_0.2pct/
%       ...
%       Strain_b_Tension_0.5pct/
%
%   根目录下还可能存在：
%       optics_summary.csv
%       optics_summary_Re.png
%       optics_summary_Im.png
%
% 本脚本会自动忽略这些根目录汇总文件，只识别真正的工况子目录。
%
% 兼容的旧目录风格：
%   Graphene/
%       ref_0.0/
%       zigzag_p0.001/
%       armchair_p0.001/
%
%   WTe2/
%       00_Optimized_Structure/
%       Compression_1%/
%       Tension_1%/
%
% 输出：
%   <data_root>/<case>/sigma_tensor_xy.csv
%   <data_root>/<case>/sigma_tensor_xy.mat
%   <data_root>/<case>/sigma_tensor_xy_quicklook.png
%
%   <data_root>/grids/all/grids.mat
%   <data_root>/grids/all/cases_metadata.csv
%   <data_root>/grids/all/grid_*.csv
%   <data_root>/grids/all/summary_sigma_Re.png
%   <data_root>/grids/all/summary_sigma_Im.png
%
%   同时按 direction 自动分组输出：
%   <data_root>/grids/a/...
%   <data_root>/grids/b/...
%   <data_root>/grids/ref/...
%   <data_root>/grids/default/...
%   <data_root>/grids/zigzag/...
%   <data_root>/grids/armchair/...
%
%   另外输出：
%   <data_root>/README_sigma_dataset.txt
%
% 用法（当前推荐）：
%   prepare_sigma_dataset
%       % public copy 默认处理 ../data/source_components
%
%   prepare_sigma_dataset('optics_sigma_xy')
%       % 处理脚本同目录下的 optics_sigma_xy
%
%   prepare_sigma_dataset(fullfile('..', 'data', 'source_components'))
%       % 从 repository 的 code/ 目录显式指定相对数据路径
%
%   prepare_sigma_dataset('source_components', fullfile('..', 'data'))
%       % 按 base_root + dataset_name 拼接
%
% 说明：
%   - 若只给一个参数，且该参数本身是一个存在的文件夹路径，则直接把它当 data_root。
%   - 若不给参数，则默认使用 repository 中的 ../data/source_components。
%   - 光电导率单位显示由脚本前部的 SIGMA_UNIT_TEXT 控制，脚本本身不做单位换算。
% =====================================================================

% ===== 光电导率单位说明（请按你的数据实际情况修改） =====
SIGMA_UNIT_TEXT = 'S/cm';
% 例如可改成：
% SIGMA_UNIT_TEXT = 'S/cm';
% SIGMA_UNIT_TEXT = 'arb. units';
% SIGMA_UNIT_TEXT = 'same as input CSV';

if nargin < 1 || isempty(dataset_name)
    dataset_name = fullfile('..', 'data', 'source_components');
end

if nargin < 2 || isempty(base_root)
    % 如果 dataset_name 本身就是一个存在的目录，则直接作为 data_root
    if isfolder(dataset_name)
        data_root = char(java.io.File(dataset_name).getCanonicalPath());
    else
        base_root = fileparts(mfilename('fullpath'));
        data_root = fullfile(base_root, dataset_name);
    end
else
    data_root = fullfile(base_root, dataset_name);
end

if ~isfolder(data_root)
    error('数据总目录不存在：%s', data_root);
end

fprintf('=============================================\n');
fprintf('prepare_sigma_dataset 开始处理：%s\n', data_root);
fprintf('=============================================\n');

% ---------------------------------------------------------------------
% 1) 找到所有“工况目录”：必须至少包含 S_xx.csv 和 S_yy.csv
% ---------------------------------------------------------------------
case_dirs = find_case_dirs(data_root);
if isempty(case_dirs)
    error('未在 %s 下找到任何包含 S_xx.csv 和 S_yy.csv 的工况目录。', data_root);
end

fprintf('[INFO] 发现工况目录数：%d\n', numel(case_dirs));

% ---------------------------------------------------------------------
% 2) 逐工况组装 sigma_tensor_xy
% ---------------------------------------------------------------------
ABS_TOL = 1e-6;
REL_TOL = 1e-3;

Cases = struct( ...
    'case_label', {}, ...
    'case_id', {}, ...
    'direction', {}, ...
    'eps_percent', {}, ...
    'source_dir', {}, ...
    'omega', {}, ...
    'Re_xx', {}, 'Im_xx', {}, ...
    'Re_yy', {}, 'Im_yy', {}, ...
    'Re_xy', {}, 'Im_xy', {}, ...
    'Re_yx', {}, 'Im_yx', {});

for k = 1:numel(case_dirs)
    cdir = char(case_dirs(k));
    meta = parse_case_meta(data_root, cdir);

    fprintf('\n[CASE %d/%d] %s\n', k, numel(case_dirs), meta.case_label);

    % ---- 读取组件 ----
    S_xx = read_comp(cdir, 'S_xx.csv');
    S_yy = read_comp(cdir, 'S_yy.csv');
    S_xy = read_comp(cdir, 'S_xy.csv');
    S_yx = read_comp(cdir, 'S_yx.csv');
    A_xy = read_comp(cdir, 'A_xy.csv');
    A_yx = read_comp(cdir, 'A_yx.csv');

    if isempty(S_xx) || isempty(S_yy)
        fprintf(2, '[跳过] 缺少 S_xx/S_yy：%s\n', cdir);
        continue;
    end

    % ---- 联合频率轴 ----
    W = unique([S_xx.w; S_yy.w]);
    if ~isempty(S_xy), W = unique([W; S_xy.w]); end
    if ~isempty(S_yx), W = unique([W; S_yx.w]); end
    if ~isempty(A_xy), W = unique([W; A_xy.w]); end
    if ~isempty(A_yx), W = unique([W; A_yx.w]); end
    W = sort(W(:));

    Sxx = interp_comp(S_xx, W);
    Syy = interp_comp(S_yy, W);

    % ---- 对称部分 S_xy / S_yx ----
    if ~isempty(S_xy), Sxy = interp_comp(S_xy, W); else, Sxy = []; end
    if ~isempty(S_yx), Syx = interp_comp(S_yx, W); else, Syx = []; end

    if isempty(Sxy) && isempty(Syx)
        Sxy = struct('w',W,'re',zeros(size(W)),'im',zeros(size(W)));
        Syx = Sxy;
    elseif isempty(Sxy)
        Sxy = Syx;
    elseif isempty(Syx)
        Syx = Sxy;
    else
        warn_if_inconsistent('S_{xy} vs S_{yx} (应相等)', Sxy, Syx, ABS_TOL, REL_TOL);
        Savg.re = 0.5 * (Sxy.re + Syx.re);
        Savg.im = 0.5 * (Sxy.im + Syx.im);
        Sxy = struct('w',W,'re',Savg.re,'im',Savg.im);
        Syx = Sxy;
    end

    % ---- 反对称部分 A_xy / A_yx ----
    if ~isempty(A_xy), Axy = interp_comp(A_xy, W); else, Axy = []; end
    if ~isempty(A_yx), Ayx = interp_comp(A_yx, W); else, Ayx = []; end

    if isempty(Axy) && isempty(Ayx)
        Axy = struct('w',W,'re',zeros(size(W)),'im',zeros(size(W)));
        Ayx = struct('w',W,'re',zeros(size(W)),'im',zeros(size(W)));
    elseif isempty(Axy)
        Axy = Ayx;
        Axy.re = -Axy.re;
        Axy.im = -Axy.im;
    elseif isempty(Ayx)
        Ayx = Axy;
        Ayx.re = -Ayx.re;
        Ayx.im = -Ayx.im;
    else
        Axy_neg = struct('w',W,'re',-Axy.re,'im',-Axy.im);
        warn_if_inconsistent('A_{yx} vs -A_{xy} (应相反)', Ayx, Axy_neg, ABS_TOL, REL_TOL);

        Aavg.re = 0.5 * (Axy.re - Ayx.re);
        Aavg.im = 0.5 * (Axy.im - Ayx.im);

        Axy = struct('w',W,'re', Aavg.re,'im', Aavg.im);
        Ayx = struct('w',W,'re',-Aavg.re,'im',-Aavg.im);
    end

    % ---- 组合 sigma = S + A ----
    sig_xx.re = Sxx.re;
    sig_xx.im = Sxx.im;

    sig_yy.re = Syy.re;
    sig_yy.im = Syy.im;

    sig_xy.re = Sxy.re + Axy.re;
    sig_xy.im = Sxy.im + Axy.im;

    sig_yx.re = Syx.re + Ayx.re;
    sig_yx.im = Syx.im + Ayx.im;

    % ---- 写 sigma_tensor_xy.csv ----
    T = table(W, ...
        sig_xx.re, sig_xx.im, ...
        sig_yy.re, sig_yy.im, ...
        sig_xy.re, sig_xy.im, ...
        sig_yx.re, sig_yx.im, ...
        'VariableNames', {'omega_eV', ...
        'Re_xx','Im_xx','Re_yy','Im_yy','Re_xy','Im_xy','Re_yx','Im_yx'});

    out_csv = fullfile(cdir, 'sigma_tensor_xy.csv');
    writetable(T, out_csv, 'Delimiter', ',', 'WriteVariableNames', true);

    % ---- 写 sigma_tensor_xy.mat ----
    Ssave = struct();
    Ssave.omega_eV = W;
    Ssave.sigma_xx = sig_xx.re + 1i*sig_xx.im;
    Ssave.sigma_yy = sig_yy.re + 1i*sig_yy.im;
    Ssave.sigma_xy = sig_xy.re + 1i*sig_xy.im;
    Ssave.sigma_yx = sig_yx.re + 1i*sig_yx.im;
    save(fullfile(cdir, 'sigma_tensor_xy.mat'), '-struct', 'Ssave');

    % ---- quicklook ----
    quicklook(fullfile(cdir, 'sigma_tensor_xy_quicklook.png'), ...
        W, sig_xx, sig_yy, sig_xy, sig_yx, meta.case_label, SIGMA_UNIT_TEXT);

    fprintf('[OK] 已写出 sigma_tensor_xy.* 到 %s\n', cdir);

    % ---- 存入 Cases ----
    Cases(end+1).case_label  = meta.case_label; %#ok<AGROW>
    Cases(end).case_id       = meta.case_id;
    Cases(end).direction     = meta.direction;
    Cases(end).eps_percent   = meta.eps_percent;
    Cases(end).source_dir    = meta.source_dir;

    Cases(end).omega = W;
    Cases(end).Re_xx = sig_xx.re;
    Cases(end).Im_xx = sig_xx.im;
    Cases(end).Re_yy = sig_yy.re;
    Cases(end).Im_yy = sig_yy.im;
    Cases(end).Re_xy = sig_xy.re;
    Cases(end).Im_xy = sig_xy.im;
    Cases(end).Re_yx = sig_yx.re;
    Cases(end).Im_yx = sig_yx.im;
end

if isempty(Cases)
    error('没有成功组装出任何 sigma_tensor_xy 数据。');
end

% ---------------------------------------------------------------------
% 3) 自动分组并输出 grids
% ---------------------------------------------------------------------
grids_root = fullfile(data_root, 'grids');
if ~isfolder(grids_root)
    mkdir(grids_root);
end

group_names = build_group_names(Cases);

for ig = 1:numel(group_names)
    gname = group_names{ig};
    outdir = fullfile(grids_root, gname);
    if ~isfolder(outdir)
        mkdir(outdir);
    end

    if strcmpi(gname, 'all')
        idx = true(1, numel(Cases));
    else
        idx = strcmpi({Cases.direction}, gname);
    end

    subCases = Cases(idx);
    if isempty(subCases)
        continue;
    end

    fprintf('\n[GROUP] 写出 grids：%s （%d 个工况）\n', outdir, numel(subCases));
    write_group_outputs(outdir, subCases, SIGMA_UNIT_TEXT);
end

% ---------------------------------------------------------------------
% 4) 写 README 说明文件
% ---------------------------------------------------------------------
write_readme_sigma_dataset(data_root, Cases, SIGMA_UNIT_TEXT);

fprintf('\n=============================================\n');
fprintf('prepare_sigma_dataset 处理完成：%s\n', data_root);
fprintf('=============================================\n');

end


% =====================================================================
% 下面是辅助函数
% =====================================================================

function case_dirs = find_case_dirs(data_root)
% 任意目录，只要同时包含 S_xx.csv 与 S_yy.csv，就认定为工况目录
% 兼容当前布局：
%   optics_sigma_xy/
%       00_Optimized_Structure/
%       Strain_a_Tension_0.1pct/
%       ...
%   根目录下的 optics_summary.csv / png 会被自动忽略

    L = dir(fullfile(data_root, '**', '*.csv'));
    if isempty(L)
        case_dirs = strings(0,1);
        return;
    end

    folders = string({L.folder});
    names   = lower(string({L.name}));

    all_dirs = unique(folders);
    keep = false(size(all_dirs));

    for i = 1:numel(all_dirs)
        d = all_dirs(i);

        % 跳过 grids 目录
        rel = strrep(relpath_local(char(d), data_root), '\', '/');
        if startsWith(lower(string(rel)), "grids/")
            continue;
        end
        if strcmpi(rel, 'grids')
            continue;
        end

        % 只把真正含有 S_xx.csv 和 S_yy.csv 的目录视为工况目录
        names_here = names(folders == d);
        has_xx = any(names_here == "s_xx.csv");
        has_yy = any(names_here == "s_yy.csv");
        keep(i) = has_xx && has_yy;
    end

    case_dirs = all_dirs(keep);
end

function meta = parse_case_meta(data_root, cdir)
% 自动从目录路径解析：
%   case_label
%   case_id
%   direction
%   eps_percent
%   source_dir
%
% 当前新增支持命名：
%   00_Optimized_Structure
%   Strain_a_Tension_0.1pct
%   Strain_b_Tension_0.2pct
%   Strain_a_Compression_0.1pct

    rel = strrep(relpath_local(cdir, data_root), '\', '/');
    parts = split(string(rel), '/');
    seg = lower(parts{end});

    meta = struct();
    meta.case_label = char(rel);
    meta.source_dir = char(rel);

    % 默认
    meta.direction   = 'default';
    meta.eps_percent = NaN;

    % -------- public repository naming: unstrained / a_0p1 / b_0p5 --------
    if strcmp(seg, 'unstrained')
        meta.direction   = 'ref';
        meta.eps_percent = 0;
        meta.case_id     = make_case_id(meta.case_label);
        return;
    end

    tok = regexp(seg, '^([ab])_0p([0-9]+)$', 'tokens', 'once');
    if ~isempty(tok)
        meta.direction   = char(tok{1});
        meta.eps_percent = str2double(tok{2}) / 10;
        meta.case_id     = make_case_id(meta.case_label);
        return;
    end

    % -------- 当前命名：Strain_a_Tension_0.1pct / Strain_b_Compression_0.2pct --------
    tok = regexp(seg, '^strain_([ab])_(tension|compression)_([0-9]+(\.\d+)?)pct$', 'tokens', 'once');
    if ~isempty(tok)
        meta.direction = char(tok{1});   % 'a' or 'b'
        val = str2double(tok{3});
        if strcmp(tok{2}, 'compression')
            val = -val;
        end
        meta.eps_percent = val;
        meta.case_id = make_case_id(meta.case_label);
        return;
    end

    % -------- 当前命名：00_Optimized_Structure --------
    if strcmp(seg, '00_optimized_structure')
        meta.direction   = 'ref';
        meta.eps_percent = 0;
        meta.case_id     = make_case_id(meta.case_label);
        return;
    end

    % -------- 旧式 WTe2 命名 --------
    tok = regexp(seg, '^compression_(\d+(\.\d+)?)%$', 'tokens', 'once');
    if ~isempty(tok)
        meta.direction   = 'default';
        meta.eps_percent = -str2double(tok{1});
        meta.case_id     = make_case_id(meta.case_label);
        return;
    end

    tok = regexp(seg, '^tension_(\d+(\.\d+)?)%$', 'tokens', 'once');
    if ~isempty(tok)
        meta.direction   = 'default';
        meta.eps_percent = str2double(tok{1});
        meta.case_id     = make_case_id(meta.case_label);
        return;
    end

    % -------- Graphene: ref_0.0 --------
    tok = regexp(seg, '^ref[_\-]?(\d*\.?\d+)?$', 'tokens', 'once');
    if ~isempty(tok)
        meta.direction = 'ref';
        if isempty(tok{1})
            meta.eps_percent = 0;
        else
            val = str2double(tok{1});
            if abs(val) <= 1
                val = val * 100;
            end
            meta.eps_percent = val;
        end
        meta.case_id = make_case_id(meta.case_label);
        return;
    end

    % -------- Graphene: zigzag_p0.001 / armchair_m0.001 --------
    tok = regexp(seg, '^([a-z0-9]+)[_\-](p|m|plus|minus|pos|neg)?(\d*\.?\d+)$', 'tokens', 'once');
    if ~isempty(tok)
        meta.direction = char(tok{1});
        sign_tag = tok{2};
        val = str2double(tok{3});

        if ismember(sign_tag, {'m','minus','neg'})
            val = -val;
        end

        % 小数应变 -> 百分数
        if abs(val) <= 1
            val = val * 100;
        end
        meta.eps_percent = val;
        meta.case_id = make_case_id(meta.case_label);
        return;
    end

    % -------- 最后兜底：按关键字识别方向 --------
    if contains(seg, 'zigzag')
        meta.direction = 'zigzag';
    elseif contains(seg, 'armchair')
        meta.direction = 'armchair';
    elseif contains(seg, 'strain_a')
        meta.direction = 'a';
    elseif contains(seg, 'strain_b')
        meta.direction = 'b';
    elseif contains(seg, 'ref') || contains(seg, 'optimized')
        meta.direction = 'ref';
        meta.eps_percent = 0;
    end

    meta.case_id = make_case_id(meta.case_label);
end

function case_id = make_case_id(case_label)
    case_id = matlab.lang.makeValidName(case_label, 'ReplacementStyle', 'hex');
end

function group_names = build_group_names(Cases)
% 总是输出 all，再输出每个 unique direction
    dirs = unique({Cases.direction}, 'stable');
    group_names = [{'all'}, dirs];
end

function write_group_outputs(outdir, Cases, sigma_unit_text)
% 按一个 group 输出：
%   grids.mat
%   cases_metadata.csv
%   grid_Re_*.csv / grid_Im_*.csv
%   summary_sigma_Re.png
%   summary_sigma_Im.png

    % ---- 排序：先 eps，再 case_label ----
    Tmeta = struct2table(rmfield(Cases, ...
        {'omega','Re_xx','Im_xx','Re_yy','Im_yy','Re_xy','Im_xy','Re_yx','Im_yx'}));
    [~, ord] = sortrows(Tmeta, {'eps_percent','case_label'});
    Cases = Cases(ord);
    Tmeta = Tmeta(ord,:);

    % ---- 保证 case_id 唯一 ----
    case_ids = cellstr(matlab.lang.makeUniqueStrings(Tmeta.case_id));
    Tmeta.case_id = case_ids;

    % ---- 联合频率轴 ----
    allw = cell2mat(arrayfun(@(s) s.omega(:).', Cases, 'UniformOutput', false));
    omega = unique(round(allw * 1e6) / 1e6, 'stable');
    omega = omega(:).';  % 行向量

    nC = numel(Cases);
    nW = numel(omega);

    % ---- 分量矩阵：按 [Ncase × Nomega] 保存，便于 sigma2PSHE 直接读取 ----
    Re_xx = nan(nC, nW); Im_xx = nan(nC, nW);
    Re_yy = nan(nC, nW); Im_yy = nan(nC, nW);
    Re_xy = nan(nC, nW); Im_xy = nan(nC, nW);
    Re_yx = nan(nC, nW); Im_yx = nan(nC, nW);

    tol = 1e-6;

    for j = 1:nC
        wj = round(Cases(j).omega * 1e6) / 1e6;
        [lia, locb] = ismembertol(wj, omega, tol, 'DataScale', 1);
        idx = locb(lia);

        Re_xx(j, idx) = Cases(j).Re_xx(lia);
        Im_xx(j, idx) = Cases(j).Im_xx(lia);
        Re_yy(j, idx) = Cases(j).Re_yy(lia);
        Im_yy(j, idx) = Cases(j).Im_yy(lia);
        Re_xy(j, idx) = Cases(j).Re_xy(lia);
        Im_xy(j, idx) = Cases(j).Im_xy(lia);
        Re_yx(j, idx) = Cases(j).Re_yx(lia);
        Im_yx(j, idx) = Cases(j).Im_yx(lia);
    end

    % ---- 派生量 ----
    eps0 = 1e-12;
    rho = Re_xx ./ max(abs(Re_yy), eps0) .* sign(Re_yy);

    mag_xx = hypot(Re_xx, Im_xx);
    mag_yy = hypot(Re_yy, Im_yy);
    mag_xy = hypot(Re_xy, Im_xy);
    eta_signed = sign(Re_xy) .* (mag_xy ./ sqrt(max(mag_xx .* mag_yy, eps0)));

    % ---- 波长 ----
    lambda_nm = nan(size(omega));
    mask_pos = omega > 0;
    lambda_nm(mask_pos) = 1239.841984 ./ omega(mask_pos);

    % ---- metadata ----
    writetable(Tmeta, fullfile(outdir, 'cases_metadata.csv'));

    % ---- 保存 grids.mat ----
    eps_percent = Tmeta.eps_percent(:);
    case_labels = Tmeta.case_label(:);
    direction   = Tmeta.direction(:);
    source_dir  = Tmeta.source_dir(:);

    omega_eV = omega;
    E_eV     = omega;
    save(fullfile(outdir, 'grids.mat'), ...
        'eps_percent', 'case_ids', 'case_labels', 'direction', 'source_dir', ...
        'omega_eV', 'E_eV', 'lambda_nm', ...
        'Re_xx','Im_xx','Re_yy','Im_yy','Re_xy','Im_xy','Re_yx','Im_yx', ...
        'rho', 'eta_signed');

    % ---- 写网格 CSV：按旧习惯输出成“第一列 omega，后面各 case”为列 ----
    write_grid_csv(fullfile(outdir,'grid_Re_xx.csv'),  omega, case_ids, Re_xx.');
    write_grid_csv(fullfile(outdir,'grid_Im_xx.csv'),  omega, case_ids, Im_xx.');
    write_grid_csv(fullfile(outdir,'grid_Re_yy.csv'),  omega, case_ids, Re_yy.');
    write_grid_csv(fullfile(outdir,'grid_Im_yy.csv'),  omega, case_ids, Im_yy.');
    write_grid_csv(fullfile(outdir,'grid_Re_xy.csv'),  omega, case_ids, Re_xy.');
    write_grid_csv(fullfile(outdir,'grid_Im_xy.csv'),  omega, case_ids, Im_xy.');
    write_grid_csv(fullfile(outdir,'grid_Re_yx.csv'),  omega, case_ids, Re_yx.');
    write_grid_csv(fullfile(outdir,'grid_Im_yx.csv'),  omega, case_ids, Im_yx.');
    write_grid_csv(fullfile(outdir,'grid_rho.csv'),    omega, case_ids, rho.');
    write_grid_csv(fullfile(outdir,'grid_eta_signed.csv'), omega, case_ids, eta_signed.');

    % ---- 新增：汇总折线图 ----
    write_group_summary_plots(outdir, omega, Tmeta.case_label, ...
        Re_xx, Im_xx, Re_yy, Im_yy, Re_xy, Im_xy, Re_yx, Im_yx, sigma_unit_text);

    fprintf('[OK] 已写出 %s\n', outdir);
end

function write_group_summary_plots(outdir, omega, case_labels, ...
    Re_xx, Im_xx, Re_yy, Im_yy, Re_xy, Im_xy, Re_yx, Im_yx, sigma_unit_text)
% 在每个 grids/<group>/ 下输出两张图：
%   summary_sigma_Re.png
%   summary_sigma_Im.png
%
% 每张图是 2×2 子图，对应 xx / yy / xy / yx
% 每条线代表一个 case

    if isempty(omega)
        return;
    end

    xlo = min(omega);
    xhi = max(omega);

    % -------- Re 图 --------
    f1 = figure('Visible','off','Position',[100 100 1200 850]);
    comps_re = {
        Re_xx, '\sigma_{xx}';
        Re_yy, '\sigma_{yy}';
        Re_xy, '\sigma_{xy}';
        Re_yx, '\sigma_{yx}'
    };

    for i = 1:4
        subplot(2,2,i);
        M = comps_re{i,1};
        for j = 1:size(M,1)
            plot(omega, M(j,:), 'LineWidth', 1.0); hold on;
        end
        grid on;
        xlim([xlo, xhi]);
        xlabel('Energy (eV)');
        ylabel(sprintf('Re conductivity (%s)', sigma_unit_text), 'Interpreter', 'none');
        title(sprintf('Re %s', comps_re{i,2}), 'Interpreter', 'tex');

        if numel(case_labels) <= 12
            legend(case_labels, 'Location', 'best', 'Interpreter', 'none');
        end
    end
    sgtitle('Summary of Re(\sigma) for all cases', 'Interpreter', 'tex');
    set(f1, 'Color', 'w');
    exportgraphics(f1, fullfile(outdir, 'summary_sigma_Re.png'), 'Resolution', 220);
    close(f1);

    % -------- Im 图 --------
    f2 = figure('Visible','off','Position',[100 100 1200 850]);
    comps_im = {
        Im_xx, '\sigma_{xx}';
        Im_yy, '\sigma_{yy}';
        Im_xy, '\sigma_{xy}';
        Im_yx, '\sigma_{yx}'
    };

    for i = 1:4
        subplot(2,2,i);
        M = comps_im{i,1};
        for j = 1:size(M,1)
            plot(omega, M(j,:), 'LineWidth', 1.0); hold on;
        end
        grid on;
        xlim([xlo, xhi]);
        xlabel('Energy (eV)');
        ylabel(sprintf('Im conductivity (%s)', sigma_unit_text), 'Interpreter', 'none');
        title(sprintf('Im %s', comps_im{i,2}), 'Interpreter', 'tex');

        if numel(case_labels) <= 12
            legend(case_labels, 'Location', 'best', 'Interpreter', 'none');
        end
    end
    sgtitle('Summary of Im(\sigma) for all cases', 'Interpreter', 'tex');
    set(f2, 'Color', 'w');
    exportgraphics(f2, fullfile(outdir, 'summary_sigma_Im.png'), 'Resolution', 220);
    close(f2);
end

function write_grid_csv(fp, omega, case_ids, M)
% M: [Nomega × Ncase]
    T = array2table([omega(:), M], 'VariableNames', [{'omega_eV'}, case_ids(:).']);
    writetable(T, fp);
end

function C = read_comp(cdir, name)
    fp = fullfile(cdir, name);
    if ~isfile(fp)
        C = [];
        return;
    end

    T = readtable(fp);
    vars = string(T.Properties.VariableNames);

    w_names  = ["omega_eV","omega","w","freq_eV","Energy_eV"];
    re_names = ["Re","Real","real","re"];
    im_names = ["Im","Imag","imag","im"];

    C = struct();
    w_col  = find(ismember(vars, w_names), 1);
    re_col = find(ismember(vars, re_names), 1);
    im_col = find(ismember(vars, im_names), 1);

    if ~isempty(w_col) && ~isempty(re_col) && ~isempty(im_col)
        C.w  = T{:, w_col};
        C.re = T{:, re_col};
        C.im = T{:, im_col};
    else
        if width(T) < 3
            C = [];
            return;
        end
        C.w  = T{:,1};
        C.re = T{:,2};
        C.im = T{:,3};
    end

    C.w  = C.w(:);
    C.re = C.re(:);
    C.im = C.im(:);

    good = ~(isnan(C.w) | isnan(C.re) | isnan(C.im));
    C.w  = C.w(good);
    C.re = C.re(good);
    C.im = C.im(good);

    [C.w, idx] = sort(C.w, 'ascend');
    C.re = C.re(idx);
    C.im = C.im(idx);

    [C.w, ia] = unique(C.w, 'stable');
    C.re = C.re(ia);
    C.im = C.im(ia);
end

function C = interp_comp(C0, W)
    C.w = W(:);

    if numel(C0.w) < 2
        C.re = nan(size(W));
        C.im = nan(size(W));
        return;
    end

    C.re = interp1(C0.w, C0.re, W, 'pchip', 'extrap');
    C.im = interp1(C0.w, C0.im, W, 'pchip', 'extrap');

    mask = (W < min(C0.w)) | (W > max(C0.w));
    C.re(mask) = NaN;
    C.im(mask) = NaN;
end

function warn_if_inconsistent(tag, C1, C2, ABS_TOL, REL_TOL)
    z1 = C1.re + 1i * C1.im;
    z2 = C2.re + 1i * C2.im;

    if ~isequal(numel(z1), numel(z2))
        fprintf(2, '[警告] %s 频点数不一致（%d vs %d）。\n', tag, numel(z1), numel(z2));
        return;
    end

    dz = z1 - z2;
    mag = abs(z1);
    rel_err = abs(dz) ./ max(mag, ABS_TOL);

    if any((abs(dz) > ABS_TOL) & (rel_err > REL_TOL))
        fprintf(2, '[警告] %s 存在超阈值差异：ABS_TOL=%g, REL_TOL=%g。\n', tag, ABS_TOL, REL_TOL);
    end
end

function quicklook(png, W, sxx, syy, sxy, syx, case_label, sigma_unit_text)
    f = figure('Visible','off','Position',[100 100 1100 800]);
    comps = {sxx,'\sigma_{xx}'; syy,'\sigma_{yy}'; sxy,'\sigma_{xy}'; syx,'\sigma_{yx}'};

    for i = 1:4
        subplot(2,2,i);
        plot(W, comps{i,1}.re, '-',  'LineWidth', 1.1); hold on;
        plot(W, comps{i,1}.im, '--', 'LineWidth', 1.1);
        grid on;
        xlim([min(W), max(W)]);
        xlabel('Energy (eV)');
        ylabel(sprintf('Conductivity (%s)', sigma_unit_text), 'Interpreter', 'none');
        title(sprintf('%s | %s', comps{i,2}, case_label), 'Interpreter', 'tex');
        legend('Re','Im','Location','best');
    end

    set(gcf, 'Color', 'w');
    exportgraphics(f, png, 'Resolution', 220);
    close(f);
end

function write_readme_sigma_dataset(data_root, Cases, sigma_unit_text)
% 写出一个 txt，说明：
%   - 原始数据文件分别代表什么
%   - 脚本生成的输出文件分别代表什么
%   - grids 目录中的文件分别代表什么
%   - 当前识别到的工况有哪些
%   - 光电导率单位是什么

    fp = fullfile(data_root, 'README_sigma_dataset.txt');
    fid = fopen(fp, 'w', 'n', 'UTF-8');
    if fid < 0
        warning('无法写出 README_sigma_dataset.txt：%s', fp);
        return;
    end

    c = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, '================ README_sigma_dataset ================\r\n');
    fprintf(fid, '数据根目录：%s\r\n', data_root);
    fprintf(fid, '生成时间：%s\r\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, '\r\n');

    fprintf(fid, '一、光电导率单位说明\r\n');
    fprintf(fid, '------------------------------------------------------\r\n');
    fprintf(fid, '当前脚本中用于显示与说明的光电导率单位为：%s\r\n', sigma_unit_text);
    fprintf(fid, '本脚本不会对输入 CSV 的数值做单位换算，输出文件中的单位与原始输入文件保持一致。\r\n');
    fprintf(fid, '\r\n');

    fprintf(fid, '二、原始输入文件说明\r\n');
    fprintf(fid, '------------------------------------------------------\r\n');
    fprintf(fid, '每个工况目录中通常包含以下原始 CSV：\r\n');
    fprintf(fid, '  S_xx.csv : 对称部分 S_{xx}(omega)\r\n');
    fprintf(fid, '  S_yy.csv : 对称部分 S_{yy}(omega)\r\n');
    fprintf(fid, '  S_xy.csv : 对称部分 S_{xy}(omega)\r\n');
    fprintf(fid, '  S_yx.csv : 对称部分 S_{yx}(omega)，若不存在则脚本用 S_xy 代替\r\n');
    fprintf(fid, '  A_xy.csv : 反对称部分 A_{xy}(omega)\r\n');
    fprintf(fid, '  A_yx.csv : 反对称部分 A_{yx}(omega)，若不存在则脚本按 -A_xy 补全\r\n');
    fprintf(fid, '  _stats_per_dir.csv : 提取阶段生成的局部统计信息，仅作参考，不参与核心组装\r\n');
    fprintf(fid, '\r\n');

    fprintf(fid, '三、脚本在每个工况目录中生成的文件\r\n');
    fprintf(fid, '------------------------------------------------------\r\n');
    fprintf(fid, '  sigma_tensor_xy.csv\r\n');
    fprintf(fid, '      统一频率轴上的复电导张量数据表，列为：\r\n');
    fprintf(fid, '      omega_eV, Re_xx, Im_xx, Re_yy, Im_yy, Re_xy, Im_xy, Re_yx, Im_yx\r\n');
    fprintf(fid, '      光电导率单位：%s\r\n', sigma_unit_text);
    fprintf(fid, '\r\n');
    fprintf(fid, '  sigma_tensor_xy.mat\r\n');
    fprintf(fid, '      MATLAB 二进制文件，变量包括：\r\n');
    fprintf(fid, '      omega_eV, sigma_xx, sigma_yy, sigma_xy, sigma_yx\r\n');
    fprintf(fid, '      其中 sigma_ij 为复数，单位仍为：%s\r\n', sigma_unit_text);
    fprintf(fid, '\r\n');
    fprintf(fid, '  sigma_tensor_xy_quicklook.png\r\n');
    fprintf(fid, '      快速检查图，展示 xx/yy/xy/yx 四个分量的 Re/Im 曲线。\r\n');
    fprintf(fid, '\r\n');

    fprintf(fid, '四、grids 目录输出说明\r\n');
    fprintf(fid, '------------------------------------------------------\r\n');
    fprintf(fid, 'grids 目录下会按分组写出：\r\n');
    fprintf(fid, '  grids\\all\\...\r\n');
    fprintf(fid, '  grids\\ref\\...\r\n');
    fprintf(fid, '  grids\\a\\...\r\n');
    fprintf(fid, '  grids\\b\\...\r\n');
    fprintf(fid, '  以及若存在旧命名风格，还可能有 zigzag / armchair / default 等分组。\r\n');
    fprintf(fid, '\r\n');
    fprintf(fid, '每个分组目录中包含：\r\n');
    fprintf(fid, '  grids.mat\r\n');
    fprintf(fid, '      汇总后的 MATLAB 数据文件，适合后续脚本直接读取。\r\n');
    fprintf(fid, '      主要变量：eps_percent, case_ids, case_labels, direction, source_dir,\r\n');
    fprintf(fid, '      omega_eV, E_eV, lambda_nm,\r\n');
    fprintf(fid, '      Re_xx, Im_xx, Re_yy, Im_yy, Re_xy, Im_xy, Re_yx, Im_yx,\r\n');
    fprintf(fid, '      rho, eta_signed\r\n');
    fprintf(fid, '      其中 Re/Im 电导率分量单位：%s\r\n', sigma_unit_text);
    fprintf(fid, '\r\n');
    fprintf(fid, '  cases_metadata.csv\r\n');
    fprintf(fid, '      每个工况的元数据表。\r\n');
    fprintf(fid, '\r\n');
    fprintf(fid, '  grid_Re_xx.csv, grid_Im_xx.csv, ...\r\n');
    fprintf(fid, '      第一列为 omega_eV，后续每一列对应一个工况 case_id。\r\n');
    fprintf(fid, '      光电导率单位：%s\r\n', sigma_unit_text);
    fprintf(fid, '\r\n');
    fprintf(fid, '  grid_rho.csv\r\n');
    fprintf(fid, '      rho = Re_xx / Re_yy（带数值保护）的网格数据，为无量纲量。\r\n');
    fprintf(fid, '\r\n');
    fprintf(fid, '  grid_eta_signed.csv\r\n');
    fprintf(fid, '      eta_signed = sign(Re_xy) * |sigma_xy| / sqrt(|sigma_xx||sigma_yy|)\r\n');
    fprintf(fid, '      为无量纲量。\r\n');
    fprintf(fid, '\r\n');
    fprintf(fid, '  summary_sigma_Re.png\r\n');
    fprintf(fid, '      每个分组内所有工况的 Re(sigma_xx), Re(sigma_yy), Re(sigma_xy), Re(sigma_yx) 汇总折线图。\r\n');
    fprintf(fid, '\r\n');
    fprintf(fid, '  summary_sigma_Im.png\r\n');
    fprintf(fid, '      每个分组内所有工况的 Im(sigma_xx), Im(sigma_yy), Im(sigma_xy), Im(sigma_yx) 汇总折线图。\r\n');
    fprintf(fid, '\r\n');

    fprintf(fid, '五、当前工况命名解析规则\r\n');
    fprintf(fid, '------------------------------------------------------\r\n');
    fprintf(fid, '  00_Optimized_Structure           -> direction=ref, eps=0\r\n');
    fprintf(fid, '  Strain_a_Tension_0.1pct          -> direction=a,   eps=+0.1\r\n');
    fprintf(fid, '  Strain_a_Compression_0.1pct      -> direction=a,   eps=-0.1\r\n');
    fprintf(fid, '  Strain_b_Tension_0.2pct          -> direction=b,   eps=+0.2\r\n');
    fprintf(fid, '  Strain_b_Compression_0.2pct      -> direction=b,   eps=-0.2\r\n');
    fprintf(fid, '\r\n');

    fprintf(fid, '六、当前识别到的工况\r\n');
    fprintf(fid, '------------------------------------------------------\r\n');
    for i = 1:numel(Cases)
        fprintf(fid, '  %-32s | direction=%-8s | eps_percent=%8.4f\r\n', ...
            Cases(i).case_label, Cases(i).direction, Cases(i).eps_percent);
    end
    fprintf(fid, '\r\n');

    fprintf(fid, '======================================================\r\n');

    fprintf('[OK] 已写出 %s\n', fp);
end

function p = relpath_local(target, base)
% Windows/Unix 通用相对路径
% 修复：统一把 split 的结果转成行向量，避免 [up, down] 维度不一致

    target = char(java.io.File(target).getCanonicalPath());
    base   = char(java.io.File(base).getCanonicalPath());

    targetParts = split(string(strrep(target,'\','/')), '/');
    baseParts   = split(string(strrep(base,'\','/')), '/');

    % ===== 关键修复：统一成行向量 =====
    targetParts = targetParts(:).';
    baseParts   = baseParts(:).';

    % 去掉空段（例如盘符前后的异常分隔）
    targetParts(targetParts == "") = [];
    baseParts(baseParts == "") = [];

    n = min(numel(targetParts), numel(baseParts));
    k = 0;
    for i = 1:n
        if strcmp(targetParts{i}, baseParts{i})
            k = i;
        else
            break;
        end
    end

    up = repmat("..", 1, max(0, numel(baseParts) - k));
    down = targetParts(k+1:end);
    down = down(:).';   % 再保险一次，强制行向量

    parts = [up, down];
    parts(parts == "") = [];

    if isempty(parts)
        p = '.';
    else
        p = char(join(parts, filesep));
    end
end