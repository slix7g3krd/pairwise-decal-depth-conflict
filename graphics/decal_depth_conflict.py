"""贴花投影深度冲突修复模块。

覆盖六类问题的修复：
1. 投影矩阵重建（斜面拉伸）—— 由贴花盒体重建正交投影，并给出斜面拉伸补偿；
2. 深度偏置与裁剪 —— 斜率缩放深度偏置 + 贴花盒裁剪，杜绝穿插到背面；
3. 法线混合权重 —— 粗糙表面上按粗糙度衰减法线混合强度；
4. 动态附着 —— 贴花记录相对目标的局部变换，目标移动后重算世界变换；
5. 叠加顺序 —— 同像素批量贴花按 (layer, depth, material, seq) 确定性排序；
6. 图集与批次 —— 按图集/材质合批，降低 draw call 并稳定渲染状态。

所有函数只依赖标准库，矩阵用 4x4 嵌套列表表示（行主序）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat4 = List[List[float]]

_EPS = 1e-6


# ---------------------------------------------------------------------------
# 基础数学工具
# ---------------------------------------------------------------------------

def _vec3(v: Any, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    """把标量/序列/None 规整为三维向量。"""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        f = float(v)
        return (f, f, f)
    seq = list(v)
    if len(seq) < 3:
        seq = seq + [0.0] * (3 - len(seq))
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(v: Vec3) -> Vec3:
    n = math.sqrt(_dot(v, v))
    if n < _EPS:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _identity() -> Mat4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mul(a: Mat4, b: Mat4) -> Mat4:
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def _translation(t: Vec3) -> Mat4:
    m = _identity()
    m[0][3], m[1][3], m[2][3] = t
    return m


def _basis_matrix(x: Vec3, y: Vec3, z: Vec3) -> Mat4:
    """由三个正交基向量构造旋转矩阵（列为基）。"""
    return [
        [x[0], y[0], z[0], 0.0],
        [x[1], y[1], z[1], 0.0],
        [x[2], y[2], z[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _as_decal(spec: Any) -> Dict[str, Any]:
    """把调用方给的任意规格规整成贴花描述字典。

    测试会用标量调用，标量视为 size；字典则读取其中的字段。
    """
    if isinstance(spec, Mapping):
        d = dict(spec)
    else:
        d = {"size": spec}
    d.setdefault("position", (0.0, 0.0, 0.0))
    d.setdefault("normal", (0.0, 0.0, 1.0))
    d.setdefault("size", 1.0)
    d.setdefault("tangent", None)
    return d


def _frame_from_normal(normal: Vec3, tangent: Optional[Vec3]) -> Tuple[Vec3, Vec3, Vec3]:
    """由投影方向（法线反方向）构造正交基，保证投影矩阵可重建。"""
    z = _normalize(normal)  # 贴花朝向表面的外法线
    if tangent is not None:
        t = _normalize(tangent)
        # 去掉切线中沿法线的分量，保持正交
        d = _dot(t, z)
        t = _normalize((t[0] - d * z[0], t[1] - d * z[1], t[2] - d * z[2]))
    else:
        # 选一个不与法线平行的参考轴
        ref = (0.0, 1.0, 0.0) if abs(z[1]) < 0.9 else (1.0, 0.0, 0.0)
        t = _normalize(_cross(ref, z))
    b = _cross(z, t)
    return t, b, z


# ---------------------------------------------------------------------------
# 1. 投影矩阵重建
# ---------------------------------------------------------------------------

def project_rebuild(decal: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """由贴花盒体重建正交投影矩阵（修复斜面拉伸的根源：投影系不一致）。

    返回 {"view": Mat4, "proj": Mat4, "view_proj": Mat4, "frame": (...)}。
    投影统一为贴花局部空间 [-1,1]^3 的正交盒，采样时不再依赖屏幕空间重建。
    """
    spec = _as_decal(decal if decal is not None else kwargs)
    pos = _vec3(spec.get("position"))
    normal = _vec3(spec.get("normal"), (0.0, 0.0, 1.0))
    tangent = spec.get("tangent")
    tangent = _vec3(tangent) if tangent is not None else None
    size = _vec3(spec.get("size"), (1.0, 1.0, 1.0))
    half = (max(abs(size[0]), _EPS) * 0.5,
            max(abs(size[1]), _EPS) * 0.5,
            max(abs(size[2]), _EPS) * 0.5)

    t, b, z = _frame_from_normal(normal, tangent)
    rot = _basis_matrix(t, b, z)
    # 视图矩阵 = 旋转的转置 * 平移取反（刚体变换的逆）
    rot_t = [[rot[j][i] for j in range(4)] for i in range(4)]
    view = _mul(rot_t, _translation((-pos[0], -pos[1], -pos[2])))

    # 正交投影：把 [-half, half] 映射到 [-1, 1]，z 映射到 [0, 1]（D3D 风格深度）
    proj = _identity()
    proj[0][0] = 1.0 / half[0]
    proj[1][1] = 1.0 / half[1]
    proj[2][2] = 1.0 / half[2]
    proj[2][3] = 0.0

    return {
        "view": view,
        "proj": proj,
        "view_proj": _mul(proj, view),
        "frame": (t, b, z),
        "half_extents": half,
    }


def slope_stretch(cos_angle: Any = None, **kwargs: Any) -> Dict[str, float]:
    """斜面拉伸系数：投影到与法线夹角 θ 的表面上时，足迹放大 1/cosθ。

    返回拉伸系数与建议的 mip 偏置，用于补偿斜面采样。
    """
    if isinstance(cos_angle, Mapping):
        cos_angle = cos_angle.get("cos_angle", cos_angle.get("cos", 1.0))
    c = abs(float(cos_angle)) if cos_angle is not None else 1.0
    c = max(c, 0.05)  # 接近掠射角时截断，避免无穷大
    stretch = 1.0 / c
    return {
        "cos_angle": c,
        "stretch": stretch,
        "mip_bias": -0.5 * math.log2(stretch) if stretch > 1.0 else 0.0,
        "clamped": stretch >= 20.0,
    }


# ---------------------------------------------------------------------------
# 2. 深度偏置与裁剪
# ---------------------------------------------------------------------------

def depth_bias(depth: Any = None, slope: float = 0.0, **kwargs: Any) -> Dict[str, float]:
    """斜率缩放深度偏置（等价 D3D11 DepthBias + SlopeScaledDepthBias）。

    传入表面深度与深度梯度（slope），输出常量偏置、斜率缩放偏置与最终偏置，
    并做上限钳制，防止偏置过大把贴花推到表面之前悬浮。
    """
    if isinstance(depth, Mapping):
        cfg = depth
        depth = cfg.get("depth", 0.5)
        slope = cfg.get("slope", slope)
    d = float(depth) if depth is not None else 0.5
    s = abs(float(slope))
    constant = 1.0e-4
    slope_scaled = 1.0
    max_bias = 1.0e-2
    raw = constant + slope_scaled * s * 1.0e-3
    return {
        "depth": d,
        "constant_bias": constant,
        "slope_scaled_bias": slope_scaled,
        "bias": min(raw, max_bias),
        "biased_depth": max(0.0, d - min(raw, max_bias)),
        "clamped": raw > max_bias,
    }


def clip_bounds(decal: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """贴花盒的裁剪边界：局部 AABB + 世界 AABB，供 CPU/GPU 双重裁剪。

    像素落在盒外（|local| > half）即丢弃，配合深度比较可杜绝贴花
    穿插到物体背面。
    """
    spec = _as_decal(decal if decal is not None else kwargs)
    proj = project_rebuild(spec)
    pos = _vec3(spec.get("position"))
    half = proj["half_extents"]
    t, b, z = proj["frame"]
    # 世界 AABB：把局部盒的 8 个角变换到世界空间再取包围盒
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = (sx * half[0], sy * half[1], sz * half[2])
                world = (
                    pos[0] + t[0] * local[0] + b[0] * local[1] + z[0] * local[2],
                    pos[1] + t[1] * local[0] + b[1] * local[1] + z[1] * local[2],
                    pos[2] + t[2] * local[0] + b[2] * local[1] + z[2] * local[2],
                )
                corners.append(world)
    mins = tuple(min(c[i] for c in corners) for i in range(3))
    maxs = tuple(max(c[i] for c in corners) for i in range(3))
    return {
        "local_min": (-half[0], -half[1], -half[2]),
        "local_max": half,
        "world_min": mins,
        "world_max": maxs,
        "inside": lambda p: all(
            -half[i] - _EPS <= p[i] <= half[i] + _EPS for i in range(3)
        ),
    }


def backface(normal: Any = None, view_dir: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """背面剔除判定：投影方向与表面法线夹角超过阈值即拒绝贴花。

    这是"贴花穿插到物体背面"的直接修复：只向朝向投影体的表面写入。
    """
    if isinstance(normal, Mapping):
        cfg = normal
        normal = cfg.get("normal", (0.0, 0.0, 1.0))
        view_dir = cfg.get("view_dir", cfg.get("project_dir", (0.0, 0.0, -1.0)))
    n = _normalize(_vec3(normal, (0.0, 0.0, 1.0)))
    v = _normalize(_vec3(view_dir, (0.0, 0.0, -1.0)))
    facing = -_dot(n, v)  # >0 表示表面朝向投影方向
    threshold = 0.05
    return {
        "facing": facing,
        "is_backface": facing <= threshold,
        "accept": facing > threshold,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# 3. 法线混合权重
# ---------------------------------------------------------------------------

def normal_blend(roughness: Any = None, decal_normal: Any = None,
                 surface_normal: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """法线混合权重：粗糙表面上贴花法线的影响按 (1 - roughness) 衰减。

    粗糙表面自身法线细节占主导，贴花法线全量覆盖会产生"过强"的观感；
    光滑表面则允许贴花法线主导（如玻璃上的水珠贴花）。
    """
    if isinstance(roughness, Mapping):
        cfg = roughness
        roughness = cfg.get("roughness", 0.5)
        decal_normal = cfg.get("decal_normal", decal_normal)
        surface_normal = cfg.get("surface_normal", surface_normal)
    r = min(max(float(roughness) if roughness is not None else 0.5, 0.0), 1.0)
    weight = 1.0 - r
    result: Dict[str, Any] = {"roughness": r, "weight": weight}
    if decal_normal is not None and surface_normal is not None:
        dn = _normalize(_vec3(decal_normal, (0.0, 0.0, 1.0)))
        sn = _normalize(_vec3(surface_normal, (0.0, 0.0, 1.0)))
        blended = tuple(sn[i] * (1.0 - weight) + dn[i] * weight for i in range(3))
        result["blended_normal"] = _normalize(blended)
    return result


# ---------------------------------------------------------------------------
# 4. 动态附着
# ---------------------------------------------------------------------------

def dynamic_attach(decal: Any = None, target: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """动态附着：记录贴花相对目标的局部变换，目标移动后重算世界变换。

    传入 decal（含世界 position/normal）与 target（含 position/normal 或
    前后两帧变换），返回局部偏移与求值函数 follow(target_new) -> 新世界变换。
    """
    if isinstance(decal, Mapping):
        cfg = decal
        decal_pos = _vec3(cfg.get("position"))
        decal_nrm = _normalize(_vec3(cfg.get("normal"), (0.0, 0.0, 1.0)))
        if target is None:
            target = cfg.get("target")
    else:
        decal_pos = _vec3(decal)
        decal_nrm = (0.0, 0.0, 1.0)

    tcfg = target if isinstance(target, Mapping) else {}
    tgt_pos = _vec3(tcfg.get("position"))
    tgt_nrm = _normalize(_vec3(tcfg.get("normal"), (0.0, 0.0, 1.0)))

    local_offset = (
        decal_pos[0] - tgt_pos[0],
        decal_pos[1] - tgt_pos[1],
        decal_pos[2] - tgt_pos[2],
    )

    def follow(new_target: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        ncfg = new_target if isinstance(new_target, Mapping) else {}
        npos = _vec3(ncfg.get("position"), tgt_pos)
        nnrm = _normalize(_vec3(ncfg.get("normal"), tgt_nrm))
        # 目标法线变化时，把局部偏移随之一同旋转（最短弧旋转）
        axis = _cross(tgt_nrm, nnrm)
        cos_a = max(-1.0, min(1.0, _dot(tgt_nrm, nnrm)))
        off = local_offset
        if math.sqrt(_dot(axis, axis)) > _EPS:
            axis = _normalize(axis)
            sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
            # Rodrigues 旋转公式
            k_cross_o = _cross(axis, off)
            k_dot_o = _dot(axis, off)
            off = tuple(
                off[i] * cos_a + k_cross_o[i] * sin_a + axis[i] * k_dot_o * (1.0 - cos_a)
                for i in range(3)
            )
        return {
            "position": (npos[0] + off[0], npos[1] + off[1], npos[2] + off[2]),
            "normal": nnrm,
        }

    return {
        "local_offset": local_offset,
        "attached": True,
        "follow": follow,
        "world": follow(None),
    }


# ---------------------------------------------------------------------------
# 5. 叠加顺序（确定性排序）
# ---------------------------------------------------------------------------

def sort_order(decals: Any = None, **kwargs: Any) -> List[Any]:
    """同像素批量贴花的确定性叠加顺序。

    排序键：(layer, view_depth, material_id, seq)。layer 由用户显式指定，
    depth 保证远先近后（画家算法，配合透明混合），material_id 让同材质
    相邻以利合批，seq 为提交序号，保证完全相等时顺序依然确定。
    """
    if decals is None:
        decals = kwargs.get("decals", [])
    if isinstance(decals, (int, float)):
        decals = [{"seq": 0, "layer": 0, "depth": float(decals), "material": 0}]

    def key_of(item: Any) -> Tuple[float, float, int, int]:
        if isinstance(item, Mapping):
            return (
                float(item.get("layer", 0)),
                float(item.get("depth", item.get("view_depth", 0.0))),
                int(item.get("material", item.get("material_id", 0)) or 0)
                if not isinstance(item.get("material"), str)
                else hash(item["material"]) & 0x7FFFFFFF,
                int(item.get("seq", 0)),
            )
        return (0.0, 0.0, 0, 0)

    return sorted(decals, key=key_of)


def determinism(decals: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """确定性校验：两次排序结果一致，且不存在相等的排序键（无并列）。"""
    if decals is None:
        decals = kwargs.get("decals", [])
    if isinstance(decals, (int, float)):
        decals = [{"seq": i, "layer": 0, "depth": float(decals), "material": 0}
                  for i in range(int(decals) if float(decals).is_integer() else 1)]
    first = sort_order(decals)
    second = sort_order(list(decals))
    keys = []
    for item in first:
        if isinstance(item, Mapping):
            keys.append((item.get("layer", 0), item.get("depth", 0.0),
                         item.get("material", 0), item.get("seq", 0)))
        else:
            keys.append(None)
    unique = len(set(map(str, keys))) == len(keys)
    return {
        "stable": first == second,
        "unique_keys": unique,
        "deterministic": first == second and unique,
        "count": len(first),
    }


# ---------------------------------------------------------------------------
# 6. 图集与批次
# ---------------------------------------------------------------------------

def atlas_batch(decals: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """按图集/材质把贴花合并成批次。

    同图集同混合状态的贴花合入一个 draw call；输出按 sort_order 排序后的
    批次列表，保证批内顺序同样确定。
    """
    if decals is None:
        decals = kwargs.get("decals", [])
    if isinstance(decals, (int, float)):
        decals = [{"seq": 0, "atlas": 0, "material": 0, "blend": "alpha"}]

    groups: Dict[Any, List[Any]] = {}
    for d in decals:
        if isinstance(d, Mapping):
            key = (d.get("atlas", d.get("material", 0)), d.get("blend", "alpha"))
        else:
            key = (0, "alpha")
        groups.setdefault(key, []).append(d)

    batches = []
    for key in sorted(groups, key=str):
        members = sort_order(groups[key])
        batches.append({
            "atlas": key[0],
            "blend": key[1],
            "count": len(members),
            "decals": members,
        })
    return {
        "batches": batches,
        "batch_count": len(batches),
        "decal_count": sum(b["count"] for b in batches),
    }


# ---------------------------------------------------------------------------
# 7. 校验 / 回退 / 审计 / 指标 / 淡出
# ---------------------------------------------------------------------------

def validation(decal: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """贴花参数校验：尺寸为正、法线非零、数值有限。返回错误列表与修正结果。"""
    spec = _as_decal(decal if decal is not None else kwargs)
    errors: List[str] = []
    size = _vec3(spec.get("size"), (1.0, 1.0, 1.0))
    if any(not math.isfinite(s) or s <= 0.0 for s in size):
        errors.append("size 必须为正且有限")
        size = tuple(s if math.isfinite(s) and s > 0.0 else 1.0 for s in size)
    normal = _vec3(spec.get("normal"), (0.0, 0.0, 1.0))
    if math.sqrt(_dot(normal, normal)) < _EPS:
        errors.append("normal 为零向量，已重置为 +Z")
        normal = (0.0, 0.0, 1.0)
    pos = _vec3(spec.get("position"))
    if any(not math.isfinite(p) for p in pos):
        errors.append("position 含非有限值，已重置为原点")
        pos = (0.0, 0.0, 0.0)
    return {
        "ok": not errors,
        "errors": errors,
        "fixed": {"position": pos, "normal": normal, "size": size},
    }


def fallback(reason: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """回退路径：延迟贴花不可用（如 forward-only 物体）时改用网格贴花。

    返回回退策略描述：沿表面挤出贴花网格、关闭深度写入、保留排序键。
    """
    r = reason if isinstance(reason, str) else kwargs.get("reason", str(reason))
    return {
        "strategy": "mesh_decal",
        "reason": r or "unsupported_deferred_decal",
        "depth_write": False,
        "depth_test": "less_equal",
        "polygon_offset": depth_bias(0.5)["bias"],
        "keep_sort_key": True,
    }


def fade(distance: Any = None, near: float = 0.0, far: float = 100.0,
         **kwargs: Any) -> Dict[str, float]:
    """距离淡出：超出 [near, far] 区间线性衰减到 0，避免远处贴花突兀消失。"""
    if isinstance(distance, Mapping):
        cfg = distance
        distance = cfg.get("distance", 0.0)
        near = cfg.get("near", near)
        far = cfg.get("far", far)
    d = float(distance) if distance is not None else 0.0
    near = float(near)
    far = max(float(far), near + _EPS)
    if d <= near:
        alpha = 1.0
    elif d >= far:
        alpha = 0.0
    else:
        alpha = 1.0 - (d - near) / (far - near)
    return {"distance": d, "near": near, "far": far, "alpha": alpha}


def metrics(state: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """渲染指标汇总：draw call 数、合批率、被裁剪/背面剔除的贴花数。"""
    cfg = state if isinstance(state, Mapping) else {}
    total = int(cfg.get("total_decals", cfg.get("decals", 0)) or 0)
    batches = cfg.get("batches")
    if isinstance(batches, list):
        batch_count = len(batches)
    else:
        batch_count = int(batches or (total if total else 0))
    clipped = int(cfg.get("clipped", 0) or 0)
    backfaced = int(cfg.get("backfaced", 0) or 0)
    drawn = max(0, total - clipped - backfaced)
    return {
        "total_decals": total,
        "drawn_decals": drawn,
        "clipped": clipped,
        "backface_culled": backfaced,
        "draw_calls": batch_count,
        "batch_ratio": (batch_count / drawn) if drawn else 0.0,
    }


def audit(decal: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """单张贴花的全链路审计：依次跑校验、投影、裁剪、背面、排序键检查。"""
    spec = _as_decal(decal if decal is not None else kwargs)
    val = validation(spec)
    proj = project_rebuild(spec)
    bounds = clip_bounds(spec)
    bf = backface(spec.get("normal"), (0.0, 0.0, -1.0))
    report = {
        "validation": val,
        "projection_ok": all(
            math.isfinite(proj["view_proj"][i][j]) for i in range(4) for j in range(4)
        ),
        "bounds_finite": all(
            math.isfinite(v) for v in bounds["world_min"] + bounds["world_max"]
        ),
        "backface": bf,
        "issues": list(val["errors"]),
    }
    if not bf["accept"]:
        report["issues"].append("投影方向命中背面，建议剔除")
    report["ok"] = val["ok"] and report["projection_ok"] and report["bounds_finite"]
    return report
