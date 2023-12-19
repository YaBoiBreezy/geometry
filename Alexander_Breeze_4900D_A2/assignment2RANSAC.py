#COMP4900D Assignment 1
#Alexander Breeze 101 143 291

import geomproc
import numpy as np
import math

numIterations=50
numCandidates=3
inlierDThreshold=9e-2   #to be inlier, must be within D of point in base
inlierNThreshold=1
epsilon=2e-2  #this is to add random noise to the pointclouds

base = geomproc.load('meshes/bunny.obj')
base.normalize()

#choose the transformation
[q, r] = np.linalg.qr(np.random.random((3, 3)))
rotation = q
transformation = 2 * np.random.random((3, 1)) - 1

print(f"rot {rotation}")
print(f"trans {transformation}")

shifted = base.copy()  #shifted will be the pointcloud I move to realign
shifted.vertex = geomproc.apply_transformation(shifted.vertex, rotation, transformation)
shifted.compute_vertex_and_face_normals()  #library says to do this for spin images

pc1 = base.sample(int(len(base.vertex)/2))
pc2 = shifted.sample(int(len(shifted.vertex)/2))  #take random sample of half the points, this gets different point sets for the 2 clouds


pt1 = geomproc.create_points(pc1.point, color=[1, 0, 0]) #save base pointcloud before noise

def addNoise(arr):
  shape = arr.shape
  randomNumbers = np.random.uniform(-epsilon, epsilon, shape) #shift every dim of every point by rand([-e,e])
  return arr + randomNumbers
pc1.point=addNoise(pc1.point) #adding noise
pc2.point=addNoise(pc2.point)

pt2 = geomproc.create_points(pc1.point, color=[0, 1, 0]) #save base pointcloud after noise
result = geomproc.mesh()
result.append(pt1)
result.append(pt2)
wo = geomproc.write_options()
wo.write_vertex_colors = True
result.save('output/bunny_noised.obj', wo) 

opt = geomproc.spin_image_options()
pc1desc = geomproc.spin_images(pc1, pc1, opt)  #get spin image descriptors
pc2desc = geomproc.spin_images(pc2, pc2, opt)


bestCount=0
bestTransformation=0
samplePC1=pc1.copy()
samplePC2=pc2.copy()
for i in range(numIterations):  #try to align point clouds multiple times
  print(f"ITERATION {i+1}/{numIterations}")
  randomIndices1 = np.random.choice(pc1.point.shape[0], size=numCandidates, replace=False)  #choose the base sample
  randomIndices2 = np.random.choice(pc2.point.shape[0], size=numCandidates, replace=False)  #choose the shifted sample

  samplePC1.point = pc1.point[randomIndices1]  #need to get points AND spin images for all of sample
  sampleDescriptor1 = pc1desc[randomIndices1]

  samplePC2.point = pc2.point[randomIndices2]  #need to get points AND spin images for all of sample
  sampleDescriptor2 = pc2desc[randomIndices2]

  # Match the descriptors
  corr = geomproc.best_match(sampleDescriptor2, pc1desc)  #swapped args here and below so I transform back to original
  corr = corr.astype(int)
  #corr = geomproc.closest_points(samplePC2, pc1)[0]  #samplePC1
  [rot, trans] = geomproc.transformation_from_correspondences(samplePC2, pc1, corr)  #derive a transformation from the samples


  pc2tr = pc2.copy()
  pc2tr.point = geomproc.apply_transformation(pc2tr.point, rot, trans)  #Apply T to all points in S;
  
  inliers=geomproc.closest_points(pc2tr,pc1)[0] #need first value, which is the list of point pairs.

  newInliers = []   #for some reason inliers is int array, so all distance=0. filtering points here
  for i in range(inliers.shape[0]):
    p1 = pc1.point[inliers[i, 1]]
    p2 = pc2tr.point[inliers[i, 0]]
    distance = np.linalg.norm(p1 - p2)
    if distance < inlierDThreshold:
        newInliers.append([inliers[i, 0], inliers[i, 1], distance])
  inliers = np.array(newInliers).astype(int)
  print(f"#inliers: {inliers.shape}")

  if inliers.shape[0]>bestCount:
    bestCount=inliers.shape[0]
    bestTransformation=geomproc.transformation_from_correspondences(pc2, pc1, inliers)  #derive a transformation from the inliers, 
    #using closest point (inlier [0], [1]) as argument
    #bestTransformation is almost identical to transformation that generated inliers
    
    

print("finished iterating")

if not bestTransformation:  #can't just keep any, because if no inliers are found then can't make a good sample for the good transformation
  print("no inliers found")
  exit()

pc3=pc2.copy()
pc3.point = geomproc.apply_transformation(pc2.point, bestTransformation[0], bestTransformation[1])  #Apply best T to all points in S;

#get the inliers so I can make them light blue in final return. This is the old inlier method.
inliers=[]
for idx, p2 in enumerate(pc3.point):
  numClosePoints = 0  #number of close points
  for p1 in pc1.point:
    if np.linalg.norm(p1 - p2) < inlierDThreshold:
      numClosePoints += 1 
      if numClosePoints >= inlierNThreshold:
        inliers.append(idx)
        break
print(f"#inliers: {len(inliers)}")

in_array = pc3.point[inliers]
out_array = np.delete(pc3.point, inliers, axis=0)

pt1 = geomproc.create_points(pc1.point, color=[1, 0, 0]) #base is red
pt2 = geomproc.create_points(pc2.point, color=[0, 1, 0]) #shifted is green
pt3 = geomproc.create_points(out_array, color=[0, 0, 1]) #aligned is blue
pt4 = geomproc.create_points(in_array, color=[0.5,0.8,1])#aligned inlier is lightblue
# Combine everything together
result = geomproc.mesh()
result.append(pt1)
result.append(pt2)
result.append(pt3)
result.append(pt4)
# Save the mesh
wo = geomproc.write_options()
wo.write_vertex_colors = True
result.save('output/bunny_corr.obj', wo)

# Print some information
print('Original transformation = ')
print(rotation)
print(transformation)
print('Alignment result = ')
print(bestTransformation[0])
print(bestTransformation[1])
