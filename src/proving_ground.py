"""訓練場 (3×3 の区画、各 6 m)。中央は平ら (出発点)。Genesis 組み込みの地形部品を使う。"""
SUB_SIZE = 6.0
H_SCALE = 0.05
V_SCALE = 0.005
SUBTERRAINS = [
    ["random_uniform_terrain", "sloped_terrain", "discrete_obstacles_terrain"],      # 砂利 / 坂 / 箱の散らばり
    ["stairs_terrain", "flat_terrain", "pyramid_stairs_terrain"],                   # 階段 / 出発点 / ピラミッド階段
    ["pyramid_sloped_terrain", "random_uniform_terrain", "discrete_obstacles_terrain"],
]
PARAMS = {
    "random_uniform_terrain": {"min_height": -0.04, "max_height": 0.04, "step": 0.01, "downsampled_scale": 0.25},
    "sloped_terrain": {"slope": 0.25},
    "pyramid_sloped_terrain": {"slope": 0.25},
    "discrete_obstacles_terrain": {"max_height": 0.08, "min_size": 0.4, "max_size": 1.2, "num_rects": 25},
    "stairs_terrain": {"step_width": 0.4, "step_height": 0.08},
    "pyramid_stairs_terrain": {"step_width": 0.4, "step_height": 0.08},
}
# 区画の中心 (x 前, y 左) と名前。評価で出発点から各区画へ向かわせる
TARGETS = {
    "砂利": (-SUB_SIZE, SUB_SIZE), "坂": (0.0, SUB_SIZE), "箱": (SUB_SIZE, SUB_SIZE),
    "階段": (-SUB_SIZE, 0.0), "ピラミッド階段": (SUB_SIZE, 0.0),
    "ピラミッド坂": (-SUB_SIZE, -SUB_SIZE), "砂利2": (0.0, -SUB_SIZE), "箱2": (SUB_SIZE, -SUB_SIZE),
}
