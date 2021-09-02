import numpy as np

np.set_printoptions(threshold=np.inf)
a=np.array([[9,-4,1,0,0,0,0],
			[-4,6,-4,1,0,0,0]])
b=np.array([[0,0,0,1,-4,5,-2],
			[0,0,0,0,1,-2,1]],)
np_base=np.array([[1,-4,6,-4,1,0,0],
				[0,1,-4,6,-4,1,0],
				[0,0,1,-4,6,-4,1]])

# print(a)
# print(b)
# print(np_base)
c=np.tile(np_base,(196,1))
d=np.append(c,b,axis=0)
e=np.insert(d,0,a,axis=0)

print(e)



