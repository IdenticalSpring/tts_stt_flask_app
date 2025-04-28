import threading
import queue

# Queue lưu các task (function, args)
task_queue = queue.Queue()

# Worker background lấy task ra xử lý
def worker():
    while True:
        func, args = task_queue.get()
        try:
            func(*args)
        except Exception as e:
            print(f"❌ Lỗi khi xử lý task: {e}")
        task_queue.task_done()

# Khởi động 1 worker thread
threading.Thread(target=worker, daemon=True).start()

# API để thêm task mới
def add_task(func, *args):
    task_queue.put((func, args))
