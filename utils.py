import sys, time

def print_progress(i, total, step=100):
    if i % step == 0:
        sys.stdout.write("\r{}/{}".format(i, total))

def is_time(epoch, trigger):
    return (trigger > 0) and (epoch % trigger == 0)

class Fps():
    def start(self, starti=0):
        self.time  = time.time()
        self.lasti = starti

    def fps(self, i):
        current = time.time()

        diff_t = current - self.time
        diff_i = i - self.lasti

        self.time  = current
        self.lasti = i

        return diff_i / diff_t
