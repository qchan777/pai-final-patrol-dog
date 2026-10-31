"""災害現場 (屋外) の見立て: 3×3 の区画、各 6 m、中央は出発点 (平ら)。
瓦礫 = 砂利、崩れた段差 = 箱の散らばり、傾いた地面 = 坂、崩れた階段 = ピラミッド階段。"""
SUB_SIZE = 6.0
H_SCALE = 0.05
V_SCALE = 0.005
SUBTERRAINS = [
    ["random_uniform_terrain", "discrete_obstacles_terrain", "pyramid_sloped_terrain"],
    ["discrete_obstacles_terrain", "flat_terrain", "random_uniform_terrain"],
    ["pyramid_stairs_terrain", "sloped_terrain", "discrete_obstacles_terrain"],
]
PARAMS = {
    "random_uniform_terrain": {"min_height": -0.05, "max_height": 0.05, "step": 0.01, "downsampled_scale": 0.3},
    "sloped_terrain": {"slope": 0.2},
    "pyramid_sloped_terrain": {"slope": 0.2},
    "discrete_obstacles_terrain": {"max_height": 0.08, "min_size": 0.4, "max_size": 1.2, "num_rects": 25},
    "pyramid_stairs_terrain": {"step_width": 0.4, "step_height": 0.07},
}
# 見回りの経由地 (GPS 座標の見立て、x 前 / y 左)。各区画の中心寄り
WAYPOINTS = [("瓦礫地帯 A", (-4.0, 4.0)), ("崩れた段差", (4.0, 4.0)), ("崩れた階段", (-4.0, -4.0)), ("迂回路", (-3.5, 0.0)), ("帰還", (0.0, 0.0))]

# 越えられない障害物 (名前, 中心 (x, y), 大きさ (sx, sy, sz))。壁 0.5 m、瓦礫の山 0.4 m。司令役は高さスキャナで事前に迂回する
OBSTACLES = [
    ("倒れた壁", (-2.2, 2.4), (0.3, 2.4, 0.5)),       # 出発点から瓦礫地帯 A への直線上
    ("瓦礫の山", (2.4, 2.0), (1.4, 1.4, 0.4)),        # 崩れた段差への途中
    ("倒れた壁", (-2.0, -2.6), (2.4, 0.3, 0.5)),      # 崩れた階段への途中
]
FLAG_H = 1.2

# 要救助者の見立て (倒れた人形): 経由地「崩れた段差」の手前に横たわる。箱の組み合わせ (胴体・頭・脚)、衝突なし
PERSON = {"near": "瓦礫地帯 A", "pos": (-5.6, 5.4), "yaw": 2.2}  # 瓦礫 (砂利) の上、経由地の 2.1 m 先。箱の密集地だと近づいた後に足がはまる (batch 3)
