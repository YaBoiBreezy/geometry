#COMP4900D Assignment 3
#Alexander Breeze 101 143 291


import geomproc
import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
import random


# Parametrize mesh "tm" with LSCM
def lscm(tm, constr_indx, constr_coord):
    """Compute a parameterization of a mesh with LSCM

    Parameters
    ----------
    tm : mesh
        Mesh object for the input shape
    constr_indx : array_like
        Index of each vertex to be used as a constraint
    constr_coord : array_like
        Constrained coordinates of each vertex listed in constr_indx

    Returns
    -------
    u : array_like
        Array of shape n x 2, where n is the number of vertices in the
        mesh. Each row of the array represents the 2D coordinates of a
        vertex in the parameterization
    A : array_like
        Parameterization matrix used for computing the embedding
    constr : array_like
        Right-hand side constraints used for solving the linear system
    """

    # Create matrix for linear system
    m = tm.face.shape[0]
    n = tm.vertex.shape[0]
    # We have 2*n columns since we are merging u and v into one vector
    # We need to merge them into one vector as they are inter-dependent
    # on each other
    # We have two constraints (equations) per triangle, 
    # and four extra fixed constraints (u and v for 2 points = 4 in total)
    A = np.zeros((2*m + 4, 2*n))

    # Set coefficients of linear system
    # Create two equations (t, and m + t) per triangle
    for t in range(m):
        # Get vertices of triangle
        [i0, i1, i2] = tm.face[t, :]
        v0 = tm.vertex[i0, :]
        v1 = tm.vertex[i1, :]
        v2 = tm.vertex[i2, :]
        # Compute edge lengths (norms)
        vec1 = v1 - v0
        vec2 = v2 - v0
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        # Compute sine and cosine of angle at v0
        vec1 /= n1
        vec2 /= n2
        cs = np.dot(vec1, vec2)
        sn = np.linalg.norm(np.cross(vec1, vec2))
        # Fill entries of matrix A according to formulas
        A[t, i0] = n2*cs - n1 
        A[t, n + i0] = -n2*sn
        A[m + t, i0] = n2*sn
        A[m + t, n + i0] = n2*cs -n1
        
        A[t, i1] = -n2*cs 
        A[t, n + i1] = n2*sn
        A[m + t, i1] = -n2*sn
        A[m + t, n + i1] = -n2*cs
        
        A[t, i2] = n1
        A[m + t, n + i2] = n1
        
    # Set constraints
    constr = np.zeros(2*m + 4)
    
    # Assign constraints to A and constr vector
    for i in range(len(constr_indx)):
        constr[2*m + 2*i] = constr_coord[i][0]
        constr[2*m + 2*i + 1] = constr_coord[i][1]
        A[2*m + 2*i, constr_indx[i]] = 1
        A[2*m + 2*i + 1, n + constr_indx[i]] = 1
    
    # Solve linear system in least-squares sense
    #sol = np.linalg.solve(A, constr)
    [sol, _, _, _] = np.linalg.lstsq(A, constr, rcond=None)
    
    # Assign coordinates to output vector in proper order
    # Map (u, v)'s in 1D array to 3D array
    u = np.zeros((tm.vertex.shape[0], 3))
    u[:, 0] = sol[0:n]
    u[:, 1] = sol[n:2*n]
    # Third coordinate is left as all zeros, so that we can save the mesh

    # Return parameterized coordinates
    return [u, A, constr]

#get area of triangle (any number of dimensions)
def triangleArea(vertices):
    crossProduct = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    area = 0.5 * np.linalg.norm(crossProduct)
    return area

#get 3D coordinates for 2D point on 2D triangle
def barycentric(point,triangle2D,triangle3D):
    '''
    print(" ")
    #point=np.mean(triangle2D, axis=0)
    #print(point)
    #print(triangle2D)
    #print(triangle3D)
    x=point[0]
    y=point[1]
    x1=triangle2D[0,0]
    x2=triangle2D[1,0]
    x3=triangle2D[2,0]
    y1=triangle2D[0,1]
    y2=triangle2D[1,1]
    y3=triangle2D[2,1]
    
    detT=(y2-y3)*(x1-x3) + (x3-x2)*(y1-y3)
    lamb1=((y2-y3)*(x-x3)+(x3-x2)*(y-y3)) / detT
    lamb2=((y3-y1)*(x-x3)+(x1-x2)*(y-y3)) / detT
    #lamb2=((y3-y1)*(x-x1)+(x1-x3)*(y-y1)) / detT #this whole method was wrong, couldn't get it to work so used dot and cross instead
    lamb3=1-(lamb1+lamb2)

    detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    lamb1 = ((y2 - y3) * (x - x3) + (x3 - x1) * (y - y3)) / detT
    lamb2 = ((y3 - y1) * (x - x3) + (x1 - x2) * (y - y3)) / detT
    lamb3 = 1 - (lamb1 + lamb2)

    point3D=lamb1*triangle3D[0]+lamb2*triangle3D[1]+lamb3*triangle3D[2]
    '''

    #better method using dot and cross products
    tri1=triangle2D[0]
    tri2=triangle2D[1]
    tri3=triangle2D[2]
    v0=tri3-tri1
    v1=tri2-tri1
    v2=point-tri1
    d00=np.dot(v0, v0)
    d01=np.dot(v0, v1)
    d11=np.dot(v1, v1)
    d20=np.dot(v2, v0)
    d21=np.dot(v2, v1)
    detT=d00*d11 - d01*d01

    # Calculate barycentric coordinates
    lamb1=(d11*d20 - d01*d21) / detT
    lamb2=(d00*d21 - d01*d20) / detT
    lamb3=1-lamb1-lamb2
    if lamb1<-0.1 or lamb2<-0.1 or lamb3<-0.1 or lamb1>1 or lamb2>1 or lamb3>1:
        print(f"=========================== WARN: BAD LAMBDA VALUES  {lamb1} {lamb2} {lamb3} =======================")

    point3D=lamb1*triangle3D[0]+lamb2*triangle3D[1]+lamb3*triangle3D[2]
    # print(point3D)
    return point3D




#Choose input mesh here
#mesh_name = 'crater'
#mesh_name = 'bunny'
mesh_name = 'camel_cut'

#Set filenames
input_mesh = 'meshes/' + mesh_name + '.obj'
input_boundary = 'meshes/' + mesh_name + '.bnd'
output_prefix_texture = 'output/' + mesh_name + '_textured'
output_prefix_param = 'output/' + mesh_name + '_param'
output_prefix_boundary = 'output/' + mesh_name + '_boundary'

# Choose n and b (percents of faces/bndEdges to get vertices)
n=50
b=100

# Initialize write options
wo = geomproc.write_options()

# Load and normalize mesh
tm = geomproc.load(input_mesh)
tm.normalize()
tm.save(output_prefix_param + '_orig.obj', wo)

# Load boundary
bnd = np.loadtxt(input_boundary)
bnd = bnd.astype(int)

# Set constraints
# Points to be used as fixed constraints
constr_indx = [bnd[0], bnd[int(len(bnd)/2)]]
# Respective coordinates of constrained points
constr_coord = [[0.5, 1], [0.5, 0]]


# Compute LSCM embedding (slow so only if needed)
try:
    print("loading LSCM")
    u = np.load("meshes/"+mesh_name+"_2D.npy")
except FileNotFoundError:
    print("LSCM not found, computing LSCM")
    [u, A, constr] = lscm(tm, constr_indx, constr_coord)
    np.save("meshes/"+mesh_name+"_2D.npy", u)




# Save mesh parametrized to 2D
param_tm = tm.copy()
param_tm.vertex = u # Assign parameterized coordinates
param_tm.save(output_prefix_param + '_lscm.obj', wo)

#make a list of triangles in the mesh, weighted by area !!in 3D mesh!!
triangleWeights=np.zeros((tm.face.shape[0]))
for i, face in enumerate(tm.face):
    vertices = tm.vertex[face]
    area = triangleArea(vertices)
    triangleWeights[i] = area

print(triangleWeights)

#sample n triangles based on area
#normalizing weights to probabilities for numpy random choice
probabilities = triangleWeights / np.sum(triangleWeights)
nSampledIndices = np.random.choice(len(triangleWeights), size=int((param_tm.face.shape[0]*n)/100), p=probabilities)

#Take random point in given triangle for each triangle in sampled indices
npoints=np.zeros((nSampledIndices.shape[0], 3), dtype=float)
for index, faceIndex in enumerate(nSampledIndices):
    vertexIndices=tm.face[faceIndex]
    vertices=param_tm.vertex[vertexIndices]
    randomBarycentric=np.random.rand(3)
    randomBarycentric=randomBarycentric/np.sum(randomBarycentric)
    randomPoint=randomBarycentric[0]*vertices[0] + randomBarycentric[1]*vertices[1] + randomBarycentric[2]*vertices[2]
    npoints[index]=randomPoint

print(npoints)

#find pairs of vertices in bnd that share a face
#note that without bnd, you can simply check each pair of vertices in the mesh and take any pair that share only one face
#so minor alteration, but significant time waste ( O(|F|*|bnd|) -> O(|F|*|V|) )
boundv=[]
boundf=[]
for v1 in range(len(bnd)):
    rowsWithV1=[]
    for i, row in enumerate(tm.face):  #Find all faces with V1 before checking on V2 for significant time saving ( O(|F|*|bnd|**2) -> O(|F|*|bnd|) )
        if bnd[v1] in row:
            rowsWithV1.append(i)
    for v2 in range(v1+1, len(bnd)):
        for i in rowsWithV1:
            row=tm.face[i]
            if bnd[v2] in row:
                boundv.append([bnd[v1], bnd[v2]])
                boundf.append(i)
print(boundv) #pairs of boundary vertices that share an edge
print(boundf) #faces corresponding

#make a list of edge vertices weighted by length
edgeWeights=[]
for v in boundv:
    vertex1 = tm.vertex[v[0]]
    vertex2 = tm.vertex[v[1]]
    edgeWeights.append(np.linalg.norm(vertex2 - vertex1))

#sample b edges based on length
#normalizing weights to probabilities for numpy random choice
probabilities = edgeWeights / np.sum(edgeWeights)
bSampledIndices = np.random.choice(len(edgeWeights), size=int((len(boundv)*b)/100), p=probabilities)
bSampledFaces = np.zeros((bSampledIndices.shape))

#Take random point on edge for every edge in sampled indices
bpoints=np.zeros((bSampledIndices.shape[0], 3), dtype=float)
for index,i in enumerate(bSampledIndices):
    print("h")
    randomDist = np.random.uniform(0, 1)  # generate a random relative distance along the edge
    v1index, v2index = boundv[i]
    vertex1 = param_tm.vertex[v1index]
    vertex2 = param_tm.vertex[v2index]
    randomPoint = vertex1 + (vertex2 - vertex1)*randomDist  #go randomDist from vertex1 towards vertex2
    bpoints[index]=randomPoint
    print(vertex1)
    print(vertex2)
    print(randomPoint)

    bSampledFaces[index]=boundf[i]

print(bpoints)
print(bSampledFaces)

#make my new mesh to move to 3D
points=np.concatenate((npoints, bpoints), axis=0)
faceIndices=np.concatenate((nSampledIndices, bSampledFaces), axis=0)  #concatenate n and b samples
print(faceIndices)

#make good mesh to reconstruct
ptg=tm.copy()
ptg.vertex=points
tri = sp.spatial.Delaunay(ptg.vertex[:,0:2])  #triangulation to get faces
ptg.face=tri.simplices
ptg.save('output/'+mesh_name + '_sampled.obj', wo)  #save the sampled 2D mesh


#use barycentric coordinates to lift points to 3D
for vi, face in enumerate(faceIndices):
    vertex=ptg.vertex[vi]
    triangle2D=param_tm.vertex[param_tm.face[int(face)]]
    triangle3D=tm.vertex[tm.face[int(face)]]
    ptg.vertex[vi]=barycentric(vertex,triangle2D,triangle3D)
    #ptg.vertex[vi]=np.mean(triangle3D, axis=0) #inferior thing, mean of each triangle instead of barycentric cords

ptg.save(f'output/{mesh_name}_reconstructed_{n}_{b}.obj', wo)
print(mesh_name+"\a") # \a = noise so you know it's done (good if you had to generate u for LSCM, can take minutes/hours)
