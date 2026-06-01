"""
Small QThread task runner for keeping the UI responsive.
"""
import time
import traceback
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal

from npy_npz_viewer.core.task_result import TaskResult


class TaskWorker(QObject):
    finished = Signal(str, object, object)
    progress = Signal(str, int, str)

    def __init__(self, name: str, func: Callable):
        super().__init__()
        self.name = name
        self.func = func
        self.cancelled = False

    def run(self):
        start = time.perf_counter()
        try:
            result = self.func(self)
            if not isinstance(result, TaskResult):
                result = TaskResult.ok(result)
        except Exception as exc:
            result = TaskResult.fail(f"{exc}\n{traceback.format_exc()}")

        result.elapsed_ms = (time.perf_counter() - start) * 1000
        if self.cancelled:
            result = TaskResult.fail(
                "任务已取消",
                elapsed_ms=result.elapsed_ms,
                metadata={"cancelled": True},
            )
        self.finished.emit(self.name, result, self)

    def cancel(self):
        self.cancelled = True


class TaskRunner(QObject):
    task_started = Signal(str)
    task_progress = Signal(str, int, str)
    task_finished = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_thread = None
        self.current_worker = None
        self.tasks = []

    def run_task(self, name: str, func: Callable):
        self.cancel_current()

        thread = QThread()
        worker = TaskWorker(name, func)
        worker.moveToThread(thread)
        task = {"thread": thread, "worker": worker}

        thread.started.connect(worker.run)
        worker.progress.connect(self.task_progress)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda task_ref=task: self._on_thread_finished(task_ref))
        thread.finished.connect(thread.deleteLater)

        self.tasks.append(task)
        self.current_thread = thread
        self.current_worker = worker
        self.task_started.emit(name)
        thread.start()

    def cancel_current(self):
        if self.current_worker is not None:
            self.current_worker.cancel()

    def cancel_all(self):
        for task in list(self.tasks):
            task["worker"].cancel()

    def wait_for_all(self, timeout_ms: int = 3000):
        all_stopped = True
        for task in list(self.tasks):
            thread = task["thread"]
            if thread.isRunning():
                thread.quit()
                if not thread.wait(timeout_ms):
                    all_stopped = False
        return all_stopped

    def has_running_tasks(self):
        return any(task["thread"].isRunning() for task in self.tasks)

    def _on_worker_finished(self, name: str, result: TaskResult, worker: TaskWorker):
        if self.current_worker is worker:
            self.current_thread = None
            self.current_worker = None
        self.task_finished.emit(name, result)

    def _on_thread_finished(self, task: dict):
        if task in self.tasks:
            self.tasks.remove(task)
