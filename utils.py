import sys

def print_progress(i, total, step=100):
	if i % step == 0:
		sys.stdout.write("\r{}/{}".format(i, total))
		
		if i >= total - step:	# new line
			print()
