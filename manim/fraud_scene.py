from manim import *
import random

class PathfindingScene(Scene):
    def construct(self):
        # -------------------------------
        # 0:00 – 0:05 | Grid setup
        # -------------------------------
        rows, cols = 10, 10
        cell_size = 0.6
        grid = VGroup()

        for i in range(rows):
            for j in range(cols):
                cell = Square(side_length=cell_size)
                cell.set_stroke(WHITE, 1)
                cell.move_to(RIGHT*j*cell_size + UP*i*cell_size)
                grid.add(cell)
        grid.move_to(ORIGIN)
        self.play(LaggedStart(*[Create(c) for c in grid], lag_ratio=0.01))
        self.wait(0.5)

        # -------------------------------
        # 0:05 – 0:15 | Obstacles
        # -------------------------------
        num_obstacles = 15
        obstacle_cells = random.sample(list(grid), num_obstacles)  # <-- convert to list
        for cell in obstacle_cells:
            self.play(cell.animate.set_fill(GRAY), run_time=0.05)
        # -------------------------------
        # 0:15 – 0:20 | Start and Goal
        # -------------------------------
        start_cell = grid[0]
        goal_cell = grid[-1]
        start_cell.set_fill(GREEN)
        goal_cell.set_fill(RED)

        start_dot = Dot(start_cell.get_center(), color=GREEN)
        self.add(start_dot)

        # -------------------------------
        # 0:20 – 0:35 | BFS Exploration Animation
        # -------------------------------
        # Simple BFS grid indices
        def get_neighbors(idx):
            i, j = divmod(idx, cols)
            neighbors = []
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    n_idx = ni*cols + nj
                    if grid[n_idx] not in obstacle_cells:
                        neighbors.append(n_idx)
            return neighbors

        visited = set()
        queue = [0]  # start index
        parent = {}

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            self.play(grid[current].animate.set_fill(BLUE), run_time=0.05)

            if current == rows*cols - 1:  # reached goal
                break

            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
                    parent[neighbor] = current

        # -------------------------------
        # 0:35 – 0:45 | Path reconstruction
        # -------------------------------
        path = []
        cur = rows*cols - 1
        while cur != 0:
            path.append(cur)
            cur = parent.get(cur, 0)
        path.append(0)
        path = path[::-1]

        for idx in path:
            self.play(grid[idx].animate.set_fill(YELLOW), run_time=0.05)

        # -------------------------------
        # 0:45 – 0:50 | Agent movement along path
        # -------------------------------
        for idx in path:
            self.play(start_dot.animate.move_to(grid[idx].get_center()), run_time=0.1)

        # -------------------------------
        # 0:50 – 1:00 | Outro fade
        # -------------------------------
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)