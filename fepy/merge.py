

def quicksort(arr):


	if len(arr) < 2 :
		return arr

	pivot = arr[0]

	lArr = []
	rArr = []

	for i in range(1, len(arr)):

		ele = arr[i]
		if ele < pivot:
			lArr.append(ele)
		else:
			rArr.append(ele)

	
	lArr = quicksort(lArr)
	rArr = quicksort(rArr)

	newArr = lArr
	newArr.append(pivot)
	newArr.extend(rArr)

	return newArr

import random

arr = [random.randint(0,100) for p in range(0,100)]

print(arr)

print("====")
result = quicksort(arr)

print(result)
