#Alexander Breeze
#101 143 291
#COMP4900D geometry Assignment 1

#Smoothing a mesh with Taubin Smoothing

import geomproc
import numpy as np


tm = geomproc.load('meshes/bunny.obj')  #get the mesh

tm.compute_connectivity()  #get connections between vertices

num_iterations = 50
lamb = 0.1 #[0,1]
mu = -0.1 #[0,-1]
smooth = np.zeros(tm.vertex.shape)  #holds new vertices to update all at once

if lamb<0 or lamb>1 or mu>0 or mu<-1:
    print("WARN: lamb or mu out of range")

for it in range(num_iterations):
    for vi in range(tm.vertex.shape[0]):  #for each vertex
        smooth[vi, :] = ((1-lamb)*tm.vertex[vi]) + (lamb*np.average(tm.vertex[tm.viv[vi], :], axis=0))  #smooth
    tm.vertex = smooth  #replace all vertices with smoothed versions

    for vi in range(tm.vertex.shape[0]):  #for each vertex
        smooth[vi, :] = ((1-mu)*tm.vertex[vi]) + (mu*np.average(tm.vertex[tm.viv[vi], :], axis=0))  #expand
    tm.vertex = smooth  #replace all vertices with expanded versions


# Save the mesh
wo = geomproc.write_options()
tm.save(f'output/bunny_taubin_{lamb}_{mu}_{num_iterations}.obj', wo)
