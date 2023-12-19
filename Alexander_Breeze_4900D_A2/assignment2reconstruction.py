# Test reconstruction of a set of samples with the RBF method


import geomproc

# Import numpy for data arrays
import numpy as np

# Import math functions
import math

# Measure execution time
import time

c=0
for name in ["bunny","sphere"]:
 for h in [0,0.00001,0.5,1]: #0 is RBF
  for sparsity in [0.05,0.1,0.25]:

   # Load and normalize the mesh
   tm = geomproc.load(f'meshes/{name}.obj')
   tm.normalize()

   print(tm.vertex.shape)

   # Compute normal vectors
   tm.compute_vertex_and_face_normals()

   # Sample a point cloud from the mesh
   n = int(tm.vertex.shape[0] * sparsity)
   pc = tm.sample(n)

   # Define kernel for reconstruction
   if not h:
    kernel = lambda x, y: math.pow(np.linalg.norm(x - y), 3)
   else:
    wendland = lambda x, y, h: (math.pow(1 - np.linalg.norm(x - y)/h, 4))*(4.0*np.linalg.norm(x - y)/h + 1)
    kernel = lambda x, y: wendland(x, y, 0.01)

   # Define epsilon for displacing samples
   epsilon = 0.01

   # Run RBF reconstruction
   print('Reconstructing implicit function')
   start_time = time.time()
   surf = geomproc.impsurf()
   surf.setup_rbf(pc, epsilon, kernel)

   # Run marching cubes
   print('Running marching cubes')
   rec = geomproc.marching_cubes(np.array([-1.5, -1.5, -1.5]), np.array([1.5, 1.5, 1.5]), 16, surf.evaluate)

   # Report time
   end_time = time.time()
   print('Execution time = ' + str(end_time - start_time) +'s')
   c+=1
   print(f'done {c}')

   # Save output mesh
   rec.save(f'output/{name}/{h}_{sparsity}_rec.obj')
