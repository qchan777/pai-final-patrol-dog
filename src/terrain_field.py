"""犬っこ用の地形 (高さマップ)。中央 ±1.5 m は平ら、+x 方向は上り階段、-x は下り階段、±y の帯はでこぼこ。
段差は中央から離れるほど高くなる (自然なカリキュラム)。高さの単位は V_SCALE (5 mm) の整数倍。"""
import numpy as np

H_SCALE = 0.05      # 1 マス 5 cm
V_SCALE = 0.005     # 高さの単位 5 mm
FIELD_M = 20.0      # 20 m 四方
STEP_W = 0.5        # 階段の踏み面 50 cm
FLAT = 1.5          # 中央の平らな範囲 (±m)


def step_height(k):
    """k 段目の段差 [m]: 4 cm から 1 cm ずつ、最大 12 cm"""
    return min(0.03 + 0.01 * k, 0.12)


def make_height_field(kind="stairs", seed=0):
    n = int(FIELD_M / H_SCALE)
    hf = np.zeros((n, n), dtype=np.int32)
    if kind == "flat":
        return hf
    rng = np.random.default_rng(seed)
    xs = (np.arange(n) - n // 2) * H_SCALE
    for i, x in enumerate(xs):
        if abs(x) <= FLAT:
            continue
        k = int((abs(x) - FLAT) // STEP_W) + 1
        h = sum(step_height(m) for m in range(1, k + 1))
        hf[i, :] = int(round((h if x > 0 else -h) / V_SCALE))     # +x は上り、-x は下り
    for j, y in enumerate(xs):
        if abs(y) > FLAT:
            hf[:, j] += rng.integers(-6, 7, size=n)                # ±3 cm のでこぼこ
    return hf


if __name__ == "__main__":
    hf = make_height_field()
    print(hf.shape, "min/max [m]:", hf.min() * V_SCALE, hf.max() * V_SCALE)
