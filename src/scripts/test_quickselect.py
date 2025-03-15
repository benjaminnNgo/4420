import random
def quickselect_median_point(points:list, dim :int = 0,select_pivot_fn=random.choice):
    return quickselect(points, len(points) // 2, select_pivot_fn,dim)
    

def quickselect(points, k, select_pivot_fn,dim):
    if len(points) == 1:
        assert k == 0
        return points[0]

    pivot = select_pivot_fn(points)

    lows = [el for el in points if el[dim] < pivot[dim]]
    highs = [el for el in points if el[dim] > pivot[dim]]
    pivots = [el for el in points if el[dim] == pivot[dim]]

    if k < len(lows):
        return quickselect(lows, k, select_pivot_fn,dim)
    elif k < len(lows) + len(pivots):
        # Find kth largest element
        return pivots[0]
    else:
        return quickselect(highs, k - len(lows) - len(pivots), select_pivot_fn,dim)
    
if __name__ == "__main__":
    # print(5//2)
    # print(quickselect_median_point([1,2,3,4,5]))
    print(quickselect_median_point([[1,2],[0,4],[2,3]],dim=0))

