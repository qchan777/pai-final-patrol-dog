"""既知の地図 (壁・瓦礫の山) を避ける経路計画: 0.25 m 格子の A*。障害物は 0.6 m 膨らませる。
災害現場なら「ドローンの空撮から作った地図」に相当。未知の障害物は高さスキャナが担当する。"""
import heapq, math
CELL = 0.25
HALF = 10.0          # 現場は ±10 m


def _grid(obstacles, inflate=0.8):
    n = int(2 * HALF / CELL)
    blocked = [[False] * n for _ in range(n)]
    for ox, oy, sx, sy, sz in obstacles:
        for i in range(n):
            for j in range(n):
                x, y = -HALF + (i + 0.5) * CELL, -HALF + (j + 0.5) * CELL
                if abs(x - ox) < sx / 2 + inflate and abs(y - oy) < sy / 2 + inflate:
                    blocked[i][j] = True
    return blocked


def _cell(p):
    return (int((p[0] + HALF) / CELL), int((p[1] + HALF) / CELL))


class Planner:
    def __init__(self, obstacles):
        self.blocked = _grid(obstacles); self.n = len(self.blocked)

    def plan(self, start, goal):
        s, g = _cell(start), _cell(goal)
        if self.blocked[g[0]][g[1]]:
            return [goal]
        pq = [(0.0, s)]; came = {s: None}; cost = {s: 0.0}
        while pq:
            _, c = heapq.heappop(pq)
            if c == g:
                break
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == dj == 0:
                        continue
                    nb = (c[0] + di, c[1] + dj)
                    if not (0 <= nb[0] < self.n and 0 <= nb[1] < self.n) or self.blocked[nb[0]][nb[1]]:
                        continue
                    nc = cost[c] + math.hypot(di, dj)
                    if nc < cost.get(nb, 1e9):
                        cost[nb] = nc; came[nb] = c
                        heapq.heappush(pq, (nc + math.hypot(g[0] - nb[0], g[1] - nb[1]), nb))
        if g not in came:
            return [goal]
        path = []; c = g
        while c is not None:
            path.append((-HALF + (c[0] + 0.5) * CELL, -HALF + (c[1] + 0.5) * CELL)); c = came[c]
        path.reverse()
        return path[1:] + [goal]

    def lookahead(self, path, pos, dist=1.0):
        """経路上で、今の位置から dist 先の点を返す (そこへ向かう)"""
        for k, p in enumerate(path):
            if math.hypot(p[0] - pos[0], p[1] - pos[1]) > dist:
                return p
        return path[-1]
