"""
Async Worker Pool Utilities

Provides high-performance async worker pool for concurrent processing
of multiple tasks with progress tracking and error handling.
"""

import asyncio
from typing import Awaitable, TypeVar, Callable, List, Any, Optional, Dict
from functools import wraps
import traceback
from tqdm import tqdm

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Return type


class TaskModel:
    """Represents a task with function and arguments"""
    def __init__(self, func: Callable[..., Awaitable[R]], *args: Any, **kwargs: Any):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.extra_info: Any = None


class WorkerPool:
    """
    Async worker pool for concurrent task execution

    Provides efficient parallel processing with progress tracking,
    error handling, and resource management.
    """

    def __init__(self):
        """Initialize worker pool"""
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: List[tuple[Callable, Any, Any]] = []
        self.workers: List[tuple[asyncio.Task, Dict[str, Any]]] = []
        self.pbar: Optional[tqdm] = None

    def add_worker(self, count: int = 1, **kwargs: Any) -> None:
        """
        Add worker(s) to the pool

        Args:
            count: Number of workers to add
            **kwargs: Fixed arguments for all workers
        """
        for _ in range(count):
            worker_id = len(self.workers) + 1
            task = asyncio.create_task(self._worker(worker_id, **kwargs), name=f"worker-{worker_id}")
            self.workers.append((task, kwargs))

    def assign(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> None:
        """
        Assign a task to the pool

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        self.tasks.append((func, args, kwargs))

    def assign_task(self, task: TaskModel) -> None:
        """
        Assign a TaskModel to the pool

        Args:
            task: TaskModel instance
        """
        self.tasks.append((task.func, task.args, task.kwargs))

    def assign_tasks(self, tasks: List[TaskModel]) -> None:
        """
        Assign multiple TaskModels to the pool

        Args:
            tasks: List of TaskModel instances
        """
        self.tasks.extend([(task.func, task.args, task.kwargs) for task in tasks])

    async def _worker(self, worker_id: int, **worker_kwargs: Any) -> None:
        """
        Worker coroutine for processing tasks

        Args:
            worker_id: Unique worker identifier
            **worker_kwargs: Fixed arguments for this worker
        """
        while True:
            try:
                func, args, kwargs = await self.queue.get()

                # Merge worker-specific arguments with task arguments
                merged_kwargs = {**worker_kwargs, **kwargs}
                await func(*args, **merged_kwargs)

                if self.pbar:
                    self.pbar.update(1)
                self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"\n❌ Worker-{worker_id} error: {str(e)}")
                print("🔍 Stack trace:")
                print(traceback.format_exc())
                if self.pbar:
                    self.pbar.update(1)
                self.queue.task_done()

    async def run(self, verbose: bool = True) -> None:
        """
        Run all assigned tasks

        Args:
            verbose: Show progress bar
        """
        if not self.workers:
            raise RuntimeError("❌ No workers available. Call add_worker() first.")

        if len(self.tasks) == 0:
            print("ℹ️ No tasks to process.")
            return

        # Initialize progress bar
        if verbose:
            self.pbar = tqdm(total=len(self.tasks), desc="🚀 Processing tasks")
        else:
            self.pbar = None

        # Add all tasks to queue
        for func, args, kwargs in self.tasks:
            await self.queue.put((func, args, kwargs))

        # Wait for all tasks to complete
        await self.queue.join()

        # Clean up progress bar
        if self.pbar:
            self.pbar.close()
            self.pbar = None

    async def reset(self) -> None:
        """Reset pool state while keeping workers active"""
        # Wait for all tasks to complete
        await self.queue.join()

        # Clear task list
        self.tasks.clear()

        # Reset progress bar
        if self.pbar:
            self.pbar.close()
            self.pbar = None

    async def close(self) -> None:
        """Close worker pool and clean up resources"""
        # Cancel all workers
        for worker, _ in self.workers:
            worker.cancel()

        # Wait for all workers to finish
        await asyncio.gather(*[w[0] for w in self.workers], return_exceptions=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics

        Returns:
            Dictionary with pool statistics
        """
        return {
            "total_tasks": len(self.tasks),
            "active_workers": len(self.workers),
            "queue_size": self.queue.qsize() if hasattr(self.queue, 'qsize') else "unknown"
        }


def worker(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
    """
    Decorator for marking async worker functions

    Args:
        func: Async function to be wrapped

    Returns:
        Wrapped async function
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> R:
        return await func(*args, **kwargs)
    return wrapper


def task_fn(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
    """
    Decorator for marking task functions

    Args:
        func: Async function to be wrapped

    Returns:
        Wrapped async function
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> R:
        return await func(*args, **kwargs)
    return wrapper


def create_worker_pool() -> WorkerPool:
    """
    Create a new WorkerPool instance

    Returns:
        WorkerPool instance
    """
    return WorkerPool()


# Utility function for simple parallel execution
async def run_parallel(tasks: List[Callable[..., Awaitable[R]]],
                      max_concurrent: int = 10,
                      verbose: bool = True) -> List[R]:
    """
    Run multiple async tasks in parallel with limited concurrency

    Args:
        tasks: List of async functions to execute
        max_concurrent: Maximum concurrent tasks
        verbose: Show progress bar

    Returns:
        List of results in order of task completion
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def limited_task(task_func):
        async with semaphore:
            result = await task_func()
            return result

    if verbose:
        pbar = tqdm(total=len(tasks), desc="🚀 Running parallel tasks")

    async def wrapped_task(task_func):
        result = await limited_task(task_func)
        results.append(result)
        if verbose:
            pbar.update(1)

    # Create and run all tasks
    wrapped_tasks = [wrapped_task(task) for task in tasks]
    await asyncio.gather(*wrapped_tasks)

    if verbose:
        pbar.close()

    return results


__all__ = [
    'WorkerPool', 'TaskModel', 'worker', 'task_fn', 'create_worker_pool', 'run_parallel'
]