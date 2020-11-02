from annoy import AnnoyIndex
import random

f = 5
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
for i in range(1000):
    v = [random.gauss(-1, 1) for z in range(f)]
    t.add_item(i, v)

t.build(10) # 10 trees
t.save('test.ann')

# ...

u = AnnoyIndex(f, 'angular')
u.load('test.ann') # super fast, will just mmap the file
item_index = 2
q = [random.gauss(-1, 1) for z in range(f)]
vector, distance = u.get_nns_by_item(item_index, 10, include_distances=True)
# vector, distance =u.get_nns_by_vector(q, 2, include_distances=True)
# print(vector) # will find the 1000 nearest neighbors
for i, item in enumerate(zip(vector, distance)):
    print(i, ' -> ', item)
    # print(i, ': ', u.get_item_vector(i))
    print(u.get_distance(item[0], item_index))












