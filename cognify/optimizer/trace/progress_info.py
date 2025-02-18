import threading
import heapq
from termcolor import colored
import sys

from tqdm.auto import tqdm

class ProgressInfo:

    pbar_lock = threading.Lock()
    max_position = 20
    position_pool = list(range(max_position))
    heapq.heapify(position_pool)

    def __init__(self):
        self.best_score = float('-inf')
        self.lowest_cost = float('inf')
        self.lowest_exec_time = float('inf')
        self.opt_cost = 0
        self.finished = False

    def init_pbar(self, total, initial, initial_score, initial_cost, initial_exec_time, opt_cost):
        self.pbar_position = ProgressInfo.ask_for_position()
        self.pbar = tqdm(
            total=total,
            initial=initial,
            desc=self._gen_opt_bar_desc(initial_score, initial_cost, initial_exec_time, opt_cost),
            leave=True,
            position=self.pbar_position,
            colour="green",
            bar_format=r'{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]'
        )
        self.total = total
        self.current = initial

    @staticmethod
    def ask_for_position():
        with ProgressInfo.pbar_lock:
            # if no position is available, add a new one
            if len(ProgressInfo.position_pool) == 0:
                position = ProgressInfo.max_position
                ProgressInfo.max_position += 1
            else:
                position = heapq.heappop(ProgressInfo.position_pool)
            return position

    @staticmethod
    def release_position(position):
        with ProgressInfo.pbar_lock:
            heapq.heappush(ProgressInfo.position_pool, position)

    def _gen_opt_bar_desc(self, quality, cost, exec_time, total_optimization_cost):
        color = "green"
        quality = "--" if quality == 0.0 else f'{quality:.2f}'
        cost = "--" if cost == 0.0 else f'${cost*1000:.2f}'
        exec_time = "--" if exec_time == 0.0 else f'{exec_time:.2f}s'
        total_optimization_cost = "<$0.01" if total_optimization_cost < 0.01 else f'${total_optimization_cost:.2f}'
        quality_text = colored(quality, color)
        cost_text = colored(cost, color)
        exec_time_text = colored(exec_time, color)
        total_optimization_cost_text = colored(total_optimization_cost, color)

        return f"Optimization progress | best quality: {quality_text}, lowest cost@1000: {cost_text}, lowest avg latency: {exec_time_text} | Total Optimization Cost: {total_optimization_cost_text}"

    def finish(self):
        self.update_progress(self.total - self.current)
        self.finished = True

    def update_progress(self, frac: float):
        with ProgressInfo.pbar_lock:
            if not self.finished:
                if self.current + frac > self.total:
                    self.pbar.update(self.total - self.current)
                else:
                    self.pbar.update(frac)
                self.current += frac

    def update_status(self, best_score, lowest_cost, lowest_exec_time, opt_cost):
        with ProgressInfo.pbar_lock:
            if best_score is not None:
                self.best_score = max(best_score, self.best_score)

            if lowest_cost is not None:
                self.lowest_cost = min(lowest_cost, self.lowest_cost)
                
            if lowest_exec_time is not None:
                self.lowest_exec_time = min(lowest_exec_time, self.lowest_exec_time)

            if opt_cost is not None:
                self.opt_cost = max(opt_cost, self.opt_cost)

            best_score_desc = 0.0 if self.best_score == float('-inf') else self.best_score
            lowest_cost_desc = 0.0 if self.lowest_cost == float('inf') else self.lowest_cost
            lowest_exec_time_desc = 0.0 if self.lowest_exec_time == float('inf') else self.lowest_exec_time

            self.pbar.set_description(
                self._gen_opt_bar_desc(best_score_desc, lowest_cost_desc, lowest_exec_time_desc, self.opt_cost)
            )

    def release(self, hierarchy_level):
        if hierarchy_level == 0:
            ProgressInfo.release_position(self.pbar_position)

pbar = ProgressInfo()