#Alexander Breeze 101 143 291
#COMP4900D Final Project

import geomproc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import LearningRateScheduler
import keras_tuner as kt
import shutil
import json
import os
import copy

def intersectsTriangle(ray_origin, ray_direction, triangle):
    epsilon = 1e-6

    # Möller–Trumbore intersection algorithm
    vertex0, vertex1, vertex2 = triangle
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = np.cross(ray_direction, edge2)
    a = edge1.dot(h)

    if a > -epsilon and a < epsilon:
        return False  # Ray is parallel to the triangle

    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * s.dot(h)

    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * ray_direction.dot(q)

    if v < 0.0 or u + v > 1.0:
        return False

    t = f * edge2.dot(q)

    return t > epsilon

def raycast(point, mesh):
    winding_number = 0
    #NOTE: CHOOSE RAY DIRECTION **BEFORE** LOOPING THROUGH FACES, NOT ONCE PER FACE
    ray_direction = np.random.normal(size=3)
    ray_direction /= np.linalg.norm(ray_direction)

    for face in mesh.face:
        triangle_vertices = mesh.vertex[face]
        if intersectsTriangle(point, ray_direction, triangle_vertices):
            winding_number += 1

    return winding_number % 2 == 1

#get every point that will be used by marchingCubes, in a 3D array to preserve orderings
def genPoints3D(start, end, num_cubes_per_dim):
    #determine dimensions of subcubes
    cube_size = [(end[i] - start[i]) / num_cubes_per_dim for i in range(3)]
    points = np.zeros((num_cubes_per_dim + 1, num_cubes_per_dim + 1, num_cubes_per_dim + 1, 3), dtype=float)
    for x in range(num_cubes_per_dim + 1):
        for y in range(num_cubes_per_dim+ 1):
            for z in range(num_cubes_per_dim + 1):
                points[x, y, z] = [start[0] + x * cube_size[0], start[1] + y * cube_size[1], start[2] + z * cube_size[2]]
    return points

def encodeToModel(mesh_name, quality=16):
    # Create a folder to save the model files, if it already exists that's ok
    folder = f'output/{mesh_name}_model_{quality}/'
    os.makedirs(folder, exist_ok=True)

    print(f'Encoding {mesh_name}.obj with resolution of {quality}')
    start=np.array([-1, -1, -1])
    end=np.array([1, 1, 1])

    tm = geomproc.load(f'meshes/{mesh_name}.obj')
    tm.normalize()
    tm.save(f'{folder}original.obj')

    #points = genPoints(start, end, quality)  #numpy array of points marching_cubes will consider
    points = genPoints3D(start, end, quality)  #numpy array of points marching_cubes will consider
    try:  #read file from {folder_path}/binary.npy  (this saves time on reruns)
        #binary = np.load(f'{folder}/binary.npy')
        binary = np.load(f'{folder}/binary3D.npy')
        print(f"Occupancy function loaded")
    except:  #if file doesn't exits, make it
        #binary = np.array([raycast(point,tm) for point in points])  #boolean for each point representing "is point in mesh"
        binary=np.zeros((points.shape[0],points.shape[1],points.shape[2]), dtype=float)
        for x in range(points.shape[0]):
            for y in range(points.shape[1]):
                for z in range(points.shape[2]):
                    binary[x,y,z]=raycast(points[x,y,z],tm)
        #np.save(f'{folder}/binary.npy', binary)
        np.save(f'{folder}/binary3D.npy', binary)
        print(f"Occupancy function generated via raycasting")

    # Function to get neighbors for a given point
    def get_neighbors(x, y, z):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    new_x, new_y, new_z = x + i, y + j, z + k
                    if 0 <= new_x < points.shape[0] and 0 <= new_y < points.shape[1] and 0 <= new_z < points.shape[2]:
                        neighbors.append((new_x, new_y, new_z))
        return neighbors

    edge_points = []
    edge_binary = []
    other_points = []
    other_binary = []
    for x in range(points.shape[0]):
        for y in range(points.shape[1]):
            for z in range(points.shape[2]):
                current_point = (x, y, z)
                current_binary = binary[current_point]

                # Get neighbors for the current point
                neighbors = get_neighbors(x, y, z)
                edge=False
                for neighbor in neighbors:
                    if binary[neighbor]!=current_binary:
                        edge_points.append(current_point)
                        edge_binary.append(current_binary)
                        edge=True
                        break
                if not edge:
                    other_points.append(current_point)
                    other_binary.append(current_binary)
    print(f'Number of points on edge of mesh: {len(edge_points)} / {(quality+1)**3}')
    print(f'Number of points outside of mesh: {len(other_points)} / {(quality+1)**3}')
    edge_points=np.array(edge_points).reshape((-1, 3))
    edge_binary=np.array(edge_binary).astype(int).flatten()
    other_points=np.array(other_points).reshape((-1, 3))
    other_binary=np.array(other_binary).astype(int).flatten()

    def train(X,Y):
        model = keras.Sequential()
        model.add(keras.layers.Dense(units=2048, activation='tanh', input_shape=(3, ), dtype='float16')) #use float16 for half size weights
        model.add(keras.layers.Dense(units=1024, activation='tanh', dtype='float32'))
        model.add(keras.layers.Dense(units=515, activation='tanh', dtype='float32'))
        model.add(keras.layers.Dense(units=64, activation='tanh', dtype='float32'))
        model.add(keras.layers.Dense(1, activation='sigmoid', dtype='float32')) #float32 here bc many inputs to this layer
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0000001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
        acc=[] #save all accuracies
        loss=[] #save all losses
        while True:
            indices = np.random.permutation(X.shape[0]) #shuffle dataset for smoother training
            X, Y = X[indices], Y[indices]
            class_1_weight = np.mean(Y)
            class_0_weight = 1 - class_1_weight
            class_weight = {0: class_0_weight, 1: class_1_weight}  # Adjust the weights based on the class distribution
            history = model.fit(X, Y, epochs=1, batch_size=1000, verbose=1, class_weight=class_weight)
            acc.append(history.history["accuracy"][-1])
            if len(acc)>100 and min(loss[-10:])>max(loss[-100:-90])-0.0001:
                break
            loss.append(history.history["loss"][-1])
            #print(np.max(acc))
        return model

    def trainSimple(X,Y):
        model = keras.Sequential()
        model.add(keras.layers.Dense(units=512, activation='tanh', input_shape=(3, ), dtype='float16')) #use float16 for half size weights
        model.add(keras.layers.Dense(units=64, activation='tanh', dtype='float32'))
        model.add(keras.layers.Dense(1, activation='sigmoid', dtype='float32')) #float32 here bc many inputs to this layer
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        acc=[] #save all accuracies
        loss=[] #save all losses
        while True:
            indices = np.random.permutation(X.shape[0]) #shuffle dataset for smoother training
            X, Y = X[indices], Y[indices]
            class_1_weight = np.mean(Y)
            class_0_weight = 1 - class_1_weight
            class_weight = {0: class_0_weight, 1: class_1_weight}  # Adjust the weights based on the class distribution
            history = model.fit(X, Y, epochs=1, batch_size=1000, verbose=1, class_weight=class_weight)
            acc.append(history.history["accuracy"][-1])
            if acc[-1]>0.5:
                break
            loss.append(history.history["loss"][-1])
            #print(np.max(acc))
        return model

    edgeModel=train(edge_points, edge_binary)
    #edgeModel.save_weights(f'{folder}model_weights_edge.h5')
    otherModel=trainSimple(other_points, other_binary)
    otherModel.save_weights(f'{folder}model_weights_other.h5')


    
    # Save model architecture to a JSON file
    edge_model_json = edgeModel.to_json()
    other_model_json = otherModel.to_json()
    with open(f'{folder}edge_model_architecture.json', 'w') as json_file:
        json_file.write(edge_model_json)
    with open(f'{folder}other_model_architecture.json', 'w') as json_file:
        json_file.write(other_model_json)
        
    # Save model weights to separate files
    np.save(f'{folder}model_edge.npy', edge_points)

def reconstructFromModel(mesh_name, quality=16):
    folder = f'output/{mesh_name}_model_{quality}/'
    print("beginning reconstruction")
    start=np.array([-1, -1, -1])
    end=np.array([1, 1, 1])
    # Load the model architecture
    with open(f'{folder}edge_model_architecture.json', 'r') as json_file:
        edgeModelJson = json_file.read()
    with open(f'{folder}other_model_architecture.json', 'r') as json_file:
        otherModelJson = json_file.read()
    edgeModel = keras.models.model_from_json(edgeModelJson)
    otherModel = keras.models.model_from_json(otherModelJson)

    #load model weights
    edgeModel.load_weights(f'{folder}model_weights_edge.h5')
    otherModel.load_weights(f'{folder}model_weights_other.h5')
    edge_points=np.load(f'{folder}model_edge.npy')

    def MLfunc(point):
        if point in edge_points:
            prediction = edgeModel.predict(np.array([point]), verbose=0)[0]
            print("EDGE")
            prediction=1
        else:
            prediction = otherModel.predict(np.array([point]), verbose=0)[0]
            print("OTHER")
            prediction=0
        prediction = edgeModel.predict(np.array([point]), verbose=0)[0]
        return (prediction > 0.5)*-2+1  #-1 if prediction=true, +1 if prediction=false
    reconstructed_naive = geomproc.marching_cubes(start, end, quality, MLfunc)
    reconstructed_naive.save(f'{folder}ML_reconstruction.obj')

if __name__ == "__main__":
    import sys
    
    match sys.argv[1:]:
        case [file_path]:
            encodeToModel(file_path)
            reconstructFromModel(file_path)

        case [file_path, quality]:
            quality=int(quality)
            if (quality & (quality - 1)) != 0:
                print("WARNING: MarchingCubes may fail when quality is not a power of 2 (e.g. 8, 16, 32, 64)")
            #encodeToModel(file_path, quality)
            reconstructFromModel(file_path, quality)

        case _:
            print(f"Usage: {sys.argv[0]} <file_path> [<quality>]")
            sys.exit(1)