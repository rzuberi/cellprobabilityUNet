import time
import numpy as np

lst = [
    [4, 3, 7, 2,5],
    [9, 6, 1, 8,1],
    [5, 0, 2, 9,8],
    [8, 2, 3, 1,3],
    [8, 2, 3, 1,3],
    [8, 2, 3, 1,3]
]

def sort_possible_matches(possible_matches):
    max_index = max(np.argmax(np.array(possible_matches) > 0, axis=1)) #find the last index of the second image to loop through them

    cells_from_first_img_matches = [-1 for i in range(len(possible_matches))]

    for i in range(0,max_index):
        cell_index = []
        cell_score = []
        for j in range(len(possible_matches)):
            if i in np.where(possible_matches[j] > 0)[0]: #the cell in the first image does have cell i (from the second image) as a contender for a match
                cell_index.append(j)
                cell_score.append(possible_matches[j][i])
        if len(cell_index) > 0:
            cells_from_first_img_matches[cell_index[np.argmax(cell_score)]] = i
    
    return cells_from_first_img_matches
#print(sort_possible_matches(lst))

start = time.time()

pairs = []
length = min(len(lst),len(lst[0]))
for i in range(length):
    lst = np.array(lst)
    max_index = np.argmax(lst)
    row, col = np.unravel_index(max_index, lst.shape)
    pairs.append((row,col))
    lst[:,col] = 0
    lst[row,:] = 0

print('Time taken:', time.time() - start)
#print(pairs)  # Output: [(1, 0), (2, 3), (0, 2), (3, 0), (1, 3), (2, 0), (0, 0), (3, 2), (2, 2), (3, 1), (0, 3), (1, 1)]
