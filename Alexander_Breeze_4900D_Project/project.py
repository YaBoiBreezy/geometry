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

#casts a ray from a point and determines if it intersects a triangle, using the Möller–Trumbore algorithm
def intersectsTriangle(ray_origin, ray_direction, triangle):
    epsilon = 1e-6
    #Möller–Trumbore intersection algorithm
    vertex0, vertex1, vertex2 = triangle
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = np.cross(ray_direction, edge2)
    a = edge1.dot(h)

    if a > -epsilon and a < epsilon:
        return False  #ray is parallel to the triangle

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

#checks intersection of point with every triangle in the mesh
def raycast(point, mesh):
    winding_number = 0
    #NOTE: CHOOSE RAY DIRECTION **BEFORE** LOOPING THROUGH FACES, NOT ONCE PER FACE
    ray_direction = np.random.normal(size=3)
    ray_direction /= np.linalg.norm(ray_direction)

    for face in mesh.face:
        triangle_vertices = mesh.vertex[face]
        if intersectsTriangle(point, ray_direction, triangle_vertices):  #winding number counts # of intersections
            winding_number += 1

    return winding_number % 2 == 1 #return bool isPointInMesh

#get every point that will be used by marchingCubes
def genPoints(start, end, num_cubes_per_dim):
    #determine dimensions of each cube based on input parameters
    cube_size = (end - start) / num_cubes_per_dim
    points=[]
    z = start[2]
    while z <= end[2]:
        y = start[1]
        while y <= end[1]:
            x = start[0]
            while x <= end[0]:
                points.append([x,y,z])
                x += cube_size[0]
            y += cube_size[1]
        z += cube_size[2]
    return np.array(points)

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

    points = genPoints(start, end, quality)  #numpy array of points marching_cubes will consider
    try:  #read file from {folder_path}/binary.npy  (this saves time on reruns)
        binary = np.load(f'{folder}/binary.npy')
        print(f"Occupancy function loaded, {sum(binary)}/{len(binary)} points are inside mesh")
    except:  #if file doesn't exits, make it
        binary = np.array([raycast(point,tm) for point in points])  #boolean for each point representing "is point in mesh"
        np.save(f'{folder}/binary.npy', binary)
        print(f"Occupancy function generated via raycasting, {sum(binary)}/{len(binary)} points are inside mesh")

    #save a pointcloud of each sample point, colored by occupancy
    pc = geomproc.pcloud()
    pc.point = points
    pc.color = np.array([[1,0,0] if is_inside else [0,1,0] for point, is_inside in zip(points, binary)])
    wo = geomproc.write_options()
    wo.write_point_colors = True
    pc.save(f'{folder}occupancy_pointcloud.obj', wo)
    print(f"Saved occupancy function pointcloud")

    #use marchingcubes to reconstruct mesh from binary array occupancy function
    def reconstructFromPC(mesh_name, quality, points, binary):
        def func(point):
            index = np.where((points == point).all(axis=1))[0]
            if index.size > 0:
                return binary[index[0]]*-2+1 #get binary for this point, map [1, 0] to [-1, 1]
            print("WARNING: POINT NOT FOUND IN SAVED ARRAY, THIS IS BAD")
            return False #this is just in case something went wrong
        reconstructed_naive = geomproc.marching_cubes(start, end, quality, func)
        reconstructed_naive.save(f'{folder}naive_reconstruction.obj')
    reconstructFromPC(mesh_name, quality, points, binary)
    print(f"Reconstructed {mesh_name} from occupancy pointcloud")

    # Define a container to hold the best model and objective
    class BestModelContainer:
        def __init__(self):
            self.best_model = None
            self.best_objective = float('inf')
            self.best_acc = 0

    # Create an instance of the container
    best_model_container = BestModelContainer()
    print()

    #define model builder/trainer for hyperparameter optimizing
    class MyHyperModel(kt.HyperModel):
        def __init__(self, best_model_container):
            self.best_model_container = best_model_container
        
        def build(self, hp):
            '''
            model = keras.Sequential()
            units = hp.Int(f'units_layer_first', min_value=64, max_value=512, step=8)
            model.add(keras.layers.Dense(units=units, activation='relu', input_shape=(3, ), dtype='float16')) #use float16 for half size weights
            max_layers=4
            units=[]
            for i in range(max_layers):  #have to do this separately to make sure all hp are initialized
                units.append(hp.Int(f'units_layer_{i}', min_value=64, max_value=512, step=8))
            num_layers = hp.Int('num_layers', min_value=0, max_value=max_layers, step=1)
            for i in range(num_layers):
                model.add(keras.layers.Dense(units=units[i], activation='relu', dtype='float16'))
            model.add(keras.layers.Dense(1, activation='sigmoid', dtype='float16'))
            self.batch_size = hp.Int('batch_size', min_value=4, max_value=128, step=2, sampling='log')
            #self.learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
            #model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            '''
            
            model = keras.Sequential()
            model.add(keras.layers.Dense(units=2048, activation='tanh', input_shape=(3, ), dtype='float16')) #use float16 for half size weights
            model.add(keras.layers.Dense(units=1024, activation='tanh', dtype='float32'))
            model.add(keras.layers.Dense(units=515, activation='tanh', dtype='float32'))
            model.add(keras.layers.Dense(units=64, activation='tanh', dtype='float32'))
            model.add(keras.layers.Dense(1, activation='sigmoid', dtype='float32')) #float32 here bc many inputs to this layer
            #model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

        def fit(self, hp, model, X, Y, **kwargs):  # custom objective function to optimize hyperparameters

            class_1_weight = np.mean(Y)
            class_0_weight = 1 - class_1_weight
            class_weight = {0: class_0_weight, 1: class_1_weight}  # Adjust the weights based on the class distribution
            #fit model with auto, then recompile with low learning rate to optimize
            model.fit(X, Y, epochs=5, batch_size=200, verbose=1, class_weight=class_weight)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
            
            acc=[] #save all accuracies
            loss=[] #save all losses
            #train while <3 epochs, then until loss (or acc in case loss==nan) stops improving, stopping early if acc=100%
            while True: #len(acc)<3 or loss[-1] != np.max(loss[-3:]) or acc[-1]  != np.min(acc[-3:]):
                #shuffle dataset for smoother training
                indices = np.random.permutation(len(X))
                X, Y = X[indices], Y[indices]
                history = model.fit(X, Y, epochs=1, batch_size=40000, verbose=1, class_weight=class_weight)
                loss.append(history.history["loss"][-1])
                acc.append(history.history["accuracy"][-1])
                if len(acc)>5 and acc[-1]>self.best_model_container.best_acc: #new best
                    self.best_model_container.best_model=copy.deepcopy(model) #only for single hp run
                    self.best_model_container.best_acc=acc[-1] #only for single hp run
                    print("new best")
                if len(acc)>20 and (min(loss[-10:])>max(loss[-20:-10])-0.0001 or acc[-1]==1): #loss not decreasing, converged
                    break
            # Get the model size
            model_size = model.count_params()
            '''
            # Want 1.0 accuracy, so multiply reciprocal by large constant so if <100% accuracy then it is always worse than 100% accuracy
            objective = (1-acc[-1]) #10000000 * (1 - currAcc) + model_size  #add 100 000 per 1% accuracy loss
            if objective < self.best_model_container.best_objective:
                self.best_model_container.best_model = model
                self.best_model_container.best_objective = objective
                self.best_model_container.best_acc = acc[-1]*100
                print("NEW BEST MODEL!")
            '''
            print(f"CURRENT BEST: {self.best_model_container.best_acc}")
            #print(f"Returning objective {objective} with accuracy={int(acc[-1]*100)} and modelSize={model_size}")
            #return objective
            return self.best_model_container.best_acc
    
    # Create an instance of MyHyperModel with the best_model_container
    my_hyper_model = MyHyperModel(best_model_container)

    # Create a tuner with the custom objective function
    tuner = kt.tuners.BayesianOptimization(
        hypermodel=my_hyper_model,
        #objective=kt.Objective("loss", "min"),
        max_trials=1,
        project_name=f"{folder}optimizing_logs",
        overwrite=True,
    )

    #Search for best hyperparameters
    tuner.search(X=points, Y=binary.astype(int), max_trials=3)
    #get best model from the search
    model=best_model_container.best_model
    #print(f"got model of size {best_model_container.best_model.count_params()} with accuracy of {best_model_container.best_acc}")


    # Save model architecture to a JSON file
    model_json = model.to_json()
    with open(f'{folder}model_architecture.json', 'w') as json_file:
        json_file.write(model_json)
        
    # Save model weights to a separate file
    model.save_weights(f'{folder}model_weights.h5')

#use only the ML model to reconstruct the original mesh
def reconstructFromModel(mesh_name, quality=16):
    folder = f'output/{mesh_name}_model_{quality}/'
    print("beginning reconstruction")
    start=np.array([-1, -1, -1])
    end=np.array([1, 1, 1])
    # Load the model architecture
    with open(f'{folder}model_architecture.json', 'r') as json_file:
        loadedModelJson = json_file.read()
    loadedModel = keras.models.model_from_json(loadedModelJson)

    #load model weights
    loadedModel.load_weights(f'{folder}model_weights.h5')

    def MLfunc(point): #mask for marchingCubes predictions, maps point to occupancy via ML model
        prediction = loadedModel.predict(np.array([point]), verbose=0)[0]
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
            encodeToModel(file_path, quality)
            reconstructFromModel(file_path, quality)

        case _:
            print(f"Usage: {sys.argv[0]} <file_path> [<quality>]")
            sys.exit(1)

#TODO:
#test higher quality
#WRITE REPORT
#CONSIDER ITERATIVE MARCHING CUBES