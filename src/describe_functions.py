def count(values):
    i = 0
    for _ in values:
        i += 1
    return i

def mean(values):
    if not values:
        return None
    total = count(values)
    sum = 0
    for i in values:
        sum += i
    return sum / total

def max(values):
    if not values:
        return None
    max = values[0]
    for i in values:
        if i > max:
            max = i
    return max

def min(values):
    if not values:
        return None
    min = values [0]
    for i in values:
        if i < min:
            min = i
    return min
