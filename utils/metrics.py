# Adapted from https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation

import random
import numpy


def find_closest_index(L,t):
    """
    Find the index of the closest value in a list.
    
    Input:
    L -- the list
    t -- value to be found
    
    Output:
    index of the closest element
    """
    beginning = 0
    difference = abs(L[0] - t)
    best = 0
    end = len(L)
    while beginning < end:
        middle = int((end+beginning)/2)
        if abs(L[middle] - t) < difference:
            difference = abs(L[middle] - t)
            best = middle
        if t == L[middle]:
            return middle
        elif L[middle] > t:
            end = middle
        else:
            beginning = middle + 1
    return best

def ominus(a,b):
    """
    Compute the relative 3D transformation between a and b.
    
    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)
    
    Output:
    Relative 3D transformation from a to b.
    """
    return numpy.dot(numpy.linalg.inv(a),b)

def scale(a,scalar):
    """
    Scale the translational components of a 4x4 homogeneous matrix by a scale factor.
    """
    return numpy.array(
        [[a[0,0], a[0,1], a[0,2], a[0,3]*scalar],
         [a[1,0], a[1,1], a[1,2], a[1,3]*scalar],
         [a[2,0], a[2,1], a[2,2], a[2,3]*scalar],
         [a[3,0], a[3,1], a[3,2], a[3,3]]]
                       )

def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return numpy.linalg.norm(transform[0:3,3])

def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return numpy.arccos( min(1,max(-1, (numpy.trace(transform[0:3,0:3]) - 1)/2) ))

def distances_along_trajectory(traj):
    """
    Compute the translational distances along a trajectory. 
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i+1]],traj[keys[i]]) for i in range(len(keys)-1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += compute_distance(t)
        distances.append(sum)
    return distances
    
def rotations_along_trajectory(traj,scale):
    """
    Compute the angular rotations along a trajectory. 
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i+1]],traj[keys[i]]) for i in range(len(keys)-1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += compute_angle(t)*scale
        distances.append(sum)
    return distances
    

def evaluate_trajectory(traj_gt,traj_est,param_max_pairs=10000,param_fixed_delta=False,param_delta=1.00,param_delta_unit="s",param_offset=0.00,param_scale=1.00):
    """
    Compute the relative pose error between two trajectories.
    
    Input:
    traj_gt -- the first trajectory (ground truth)
    traj_est -- the second trajectory (estimated trajectory)
    param_max_pairs -- number of relative poses to be evaluated
    param_fixed_delta -- false: evaluate over all possible pairs
                         true: only evaluate over pairs with a given distance (delta)
    param_delta -- distance between the evaluated pairs
    param_delta_unit -- unit for comparison:
                        "s": seconds
                        "m": meters
                        "rad": radians
                        "deg": degrees
                        "f": frames
    param_offset -- time offset between two trajectories (to model the delay)
    param_scale -- scale to be applied to the second trajectory
    
    Output:
    list of compared poses and the resulting translation and rotation error
    """
    stamps_gt = list(traj_gt.keys())
    stamps_est = list(traj_est.keys())
    stamps_gt.sort()
    stamps_est.sort()
    
    stamps_est_return = []
    for t_est in stamps_est:
        t_gt = stamps_gt[find_closest_index(stamps_gt,t_est + param_offset)]
        t_est_return = stamps_est[find_closest_index(stamps_est,t_gt - param_offset)]
        t_gt_return = stamps_gt[find_closest_index(stamps_gt,t_est_return + param_offset)]
        if not t_est_return in stamps_est_return:
            stamps_est_return.append(t_est_return)
    if(len(stamps_est_return)<2):
        raise Exception("Number of overlap in the timestamps is too small. Did you run the evaluation on the right files?")

    if param_delta_unit=="s":
        index_est = list(traj_est.keys())
        index_est.sort()
    elif param_delta_unit=="m":
        index_est = distances_along_trajectory(traj_est)
    elif param_delta_unit=="rad":
        index_est = rotations_along_trajectory(traj_est,1)
    elif param_delta_unit=="deg":
        index_est = rotations_along_trajectory(traj_est,180/numpy.pi)
    elif param_delta_unit=="f":
        index_est = range(len(traj_est))
    else:
        raise Exception("Unknown unit for delta: '%s'"%param_delta_unit)

    if not param_fixed_delta:
        if(param_max_pairs==0 or len(traj_est)<numpy.sqrt(param_max_pairs)):
            pairs = [(i,j) for i in range(len(traj_est)) for j in range(len(traj_est))]
        else:
            pairs = [(random.randint(0,len(traj_est)-1),random.randint(0,len(traj_est)-1)) for i in range(param_max_pairs)]
    else:
        pairs = []
        for i in range(len(traj_est)):
            j = find_closest_index(index_est,index_est[i] + param_delta)
            if j!=len(traj_est)-1: 
                pairs.append((i,j))
        if(param_max_pairs!=0 and len(pairs)>param_max_pairs):
            pairs = random.sample(pairs,param_max_pairs)
    
    gt_interval = numpy.median([s-t for s,t in zip(stamps_gt[1:],stamps_gt[:-1])])
    gt_max_time_difference = 2*gt_interval
    
    result = []
    for i,j in pairs:
        stamp_est_0 = stamps_est[i]
        stamp_est_1 = stamps_est[j]

        stamp_gt_0 = stamps_gt[ find_closest_index(stamps_gt,stamp_est_0 + param_offset) ]
        stamp_gt_1 = stamps_gt[ find_closest_index(stamps_gt,stamp_est_1 + param_offset) ]
        
        if(abs(stamp_gt_0 - (stamp_est_0 + param_offset)) > gt_max_time_difference  or
           abs(stamp_gt_1 - (stamp_est_1 + param_offset)) > gt_max_time_difference):
            continue
        
        error44 = ominus(  scale(
                           ominus( traj_est[stamp_est_1], traj_est[stamp_est_0] ),param_scale),
                           ominus( traj_gt[stamp_gt_1], traj_gt[stamp_gt_0] ) )
        
        trans = compute_distance(error44)
        rot = compute_angle(error44)
        
        result.append([stamp_est_0,stamp_est_1,stamp_gt_0,stamp_gt_1,trans,rot])
        
    if len(result)<2:
        raise Exception("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory!")
        
    return result





