import os
import shutil
from typing import Dict, Optional
import contextlib
import tempfile

## Evaluator for humaneval dataset

# `problem` is a dictionary with the following keys
#   "task_id"
#   "prompt"
#   "entry_point"
#   "canonical_solution"
#   "test"

def humaneval_evaluator(problem, finalized_code):
    split_completion = finalized_code.split('\n')
    parsed_lines = []
    for line in split_completion:
        if "<result>" in line or "</result>" in line or "```" in line or "python" in line:
            continue
        parsed_lines.append(line)
    completion = '\n'.join(parsed_lines)

    result = check_correctness_thread(problem, completion, timeout=3.0)
    return 1.0 if result["passed"] else 0.0


## Adapted from: 
## https://github.com/aiwaves-cn/agents/blob/master/src/agents/datasets/humaneval.py

def check_correctness_thread(
    problem: Dict, completion: str, timeout: float, completion_id: Optional[int] = None
) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    result = []
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        original_rmtree = shutil.rmtree
        original_rmdir = os.rmdir
        original_chdir = os.chdir

        try:
            # Construct the check program and run it.
            check_program = (
                problem["prompt"]
                + completion
                + "\n"
                + problem["test"]
                + "\n"
                + f"check({problem['entry_point']})"
            )
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except BaseException as e:
            result.append(f"failed: {e}")
        finally:
            # Restore the original functions for cleanup.
            shutil.rmtree = original_rmtree
            os.rmdir = original_rmdir
            os.chdir = original_chdir

    if not result:
        result.append("timed out")
    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )


## ------- Manage code execution ------- ##
from typing import Optional
import contextlib
import faulthandler
import io
import os
import platform
import signal
import tempfile


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

