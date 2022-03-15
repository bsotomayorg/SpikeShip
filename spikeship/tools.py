import numpy as np
from numba import jit, njit

"""
Some methods were adapted from https://github.com/FilippoBovo/robustats/blob/master/c/robustats.c
"""

@jit(nopython=True)
def swap_2d(x, n2, i, j):
	for k in range(0, n2):
		temp = x[i][k];
		x[i][k] = x[j][k];
		x[j][k] = temp;
	return x

@jit(nopython=True)
def partition_on_kth_element_2d(x, begin, end, n2,  m, k):
	value = x[k][m]

	x = swap_2d(x, n2, k, end);

	i = begin
	for j in range(begin, end):
		
		if (x[j][m] < value):
			x = swap_2d(x, n2, i, j);
			i += 1

	x= swap_2d(x, n2, i, end);

	return i;

@jit(nopython=True)
def partition_on_kth_smallest_2d(x, begin, end, n2, m, k):
	while (True):
		
		if (begin == end):
			return x[begin][m]
		
		pivot_index = begin + int(np.random.random()* (end - begin + 1)); #random_range(begin, end);
		pivot_index = partition_on_kth_element_2d(x, begin, end, n2, m, pivot_index);
		
		if (k == pivot_index):
			return x[k][m];
		elif (k < pivot_index):
			end = pivot_index - 1;
		else:
			begin = pivot_index + 1;


@jit(nopython=True)
def zip_(array_a, array_b, n):
	return np.column_stack((array_a,array_b))

@njit()
def weighted_median(x, w):
	begin = 0; end = len(x) - 1;
	xw_n = end - begin + 1

	xw = zip_(x, w, xw_n)
	w_sum = np.sum(w[:xw_n])

	median, w_middle = 0.0, 0.0;
	w_lower_sum, w_lower_sum_norm, w_higher_sum, w_higher_sum_norm = 0.0, 0.0, 0.0, 0.0
	while(True):
		n = end - begin + 1
		if (n == 1):
			return x[begin]
		elif (n == 2):
			if (w[begin] >= w[end]):
				return x[begin];
			else:
				return x[end];
		else:
			median_index = begin + (n - 1) // 2; # lower median index
			median = partition_on_kth_smallest_2d( xw, begin, end, 2, 0, median_index)

			w_middle = xw[median_index][1];

			w_lower_sum = 0.;
			for i in range(begin, median_index):
				w_lower_sum += xw[i][1];
			w_lower_sum_norm = w_lower_sum / w_sum


			w_higher_sum = 0.;
			for i in range(median_index+1, end+1):
				w_higher_sum += xw[i][1];
			w_higher_sum_norm = w_higher_sum / w_sum

			if (w_lower_sum_norm < 0.5 and w_higher_sum_norm < 0.5):
				return median;
			elif (w_lower_sum_norm > 0.5):
				xw[median_index][1] = xw[median_index][1] + w_higher_sum;
				end = median_index;
			else:
				xw[median_index][1] = xw[median_index][1] + w_lower_sum;
				begin = median_index;

@jit(nopython=True,cache=True)
def weighted_median_(x,w):
	argsort = np.argsort(x)
	w = w[argsort]; x = x[argsort];

	midpoint = 0.5 * np.sum(w)

	if np.any(w > midpoint):
		return (x[w == np.max(w)])[0]
	else:
		cs_weights = np.cumsum(w)
		idx = np.where(cs_weights <= midpoint)[0][-1]
		if cs_weights[idx] == midpoint:
			return np.mean(x[idx:idx+2])
		else:
			return x[idx+1]
	

if __name__ == '__main__':
	cc.compile()