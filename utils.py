import sys
def show_progress(step, max_iter):
    msg = '\r progress {}/{}'.format(step, max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()
