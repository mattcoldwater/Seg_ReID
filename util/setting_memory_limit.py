import os
import psutil


p = psutil.Process() #os.getpid()
soft, hard = p.rlimit(psutil.RLIMIT_RSS)
m1 = p.memory_full_info().rss  # in bytes B
