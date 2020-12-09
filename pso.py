import random

# DO NOT import any other modules.
# DO NOT change the prototypes of any of the functions.
# Sample test cases given
# Grading will be based on hidden tests


# Cost function to be optimised
# Takes a list of elements
# Return the total sum of squares of even-indexed elements and inverse squares of odd-indexed elements
def cost_function(X): # 0.25 Marks
    # Your code goes here
    
    sumeven=0
    sumodd=0
    l=len(X)
    for j in range(0,l):
        if (j%2):
            sumodd=sumodd+1/(X[j]**2)
        else:
            sumeven=sumeven+X[j]**2
    return sumeven+sumodd
    


# Takes length of vector as input
# Returns 4 values - initial_position, initial_velocity, best_position and best_cost
# Initialises position to a list with random values between [-10, 10] and velocity to a list with random values between [-1, 1]
# best_position is an empty list and best cost is set to -1 initially
def initialise(length): # 0.25 Marks
    # your code goes here
    ip=[]
    iv=[]
    bp=[]
    bc=-1
    for i in range(0,length):
        r=random.uniform(-10,10)
        ip.append(r)
        r=random.uniform(-1,1)
        iv.append(r)
    return ip,iv,bp,bc


# Evaluates the position vector based on the input func
# On getting a better cost, best_position is updated in-place
# Returns the better cost 
def assess(position, best_position, best_cost, func): # 0.25 Marks
    # Your code goes here
    if(func(position)<best_cost or best_cost==-1):
        best_position[:]=position
        best_cost=func(position)
    return best_cost
    


# Updates velocity in-place by the given formula for each element:
# vel = w*vel + c1*r1*(best_position-position) + c2*r2*(best_group_position-position)
# where r1 and r2 are random numbers between 0 and 1 (not same for each element of the list)
# No return value
def velocity_update(w, c1, c2, velocity, position, best_position, best_group_position): # 0.5 Marks
    novel_velocity=[]
    l=len(velocity)
    for i in range(l):
        rand1=random.random()
        rand2=random.random()
        temp=best_position[i]-position[i]
        tempp=best_group_position[i]-position[i]
        velocity[i]=w*velocity[i]+rand1*c1*temp+rand2*c2*tempp
    return None
    # Code goes here
    


# Input - position, velocity, limits(list of two elements - [min, max])
# Updates position in-place by the given formula for each element:
# pos = pos + vel
# Position element set to limit if it crosses either limit value
# No return value
def position_update(position, velocity, limits): # 0.5 Marks
    # Code goes here
    l=len(position)
    for i in range(0,l):
        position[i]=position[i]+velocity[i]
        if(position[i]<limits[0]):
            position[i]=limits[0]
            
        if(position[i]>limits[1]):
            position[i]=limits[1]
        
    return None
    


# swarm is a list of particles each of which is a list containing current_position, current_velocity, best_position and best_cost
# Initialise these using the function written above
# In every iteration for every swarm particle, evaluate the current position using the assess function (use the cost function you have defined) and update the particle's best cost if needed
# Update the best group cost and best group position based on performance of that particle
# Then for every swarm particle, first update its velocity then its position
# Return the best position and cost for the group
def optimise(vector_length, swarm_size, w, c1, c2, limits, max_iterations, initial_best_group_position=[], initial_best_group_cost=-1): # 1.25 Marks
    # Your Code goes here
    ip=[]
    iv=[]
    ibp=[]
    ibc=[]
    ss=swarm_size
    for i in range(0,ss):
        position,velocity,bp,bc=initialise(vector_length)
        iv.append(velocity)
        ibp.append(bp)
        ibc.append(bc)
        ip.append(position)
    
    mi=max_iterations
    for i in range(0,mi):
        for j in range(0,ss):
            ibc[j]=assess(ip[j],ibp[j],ibc[j],cost_function)
            if(ibc[j]<initial_best_group_cost or initial_best_group_cost==-1):
                initial_best_group_cost=ibc[j]
                initial_best_group_position=ibp[j]
                
        for j in range(0,ss):
            velocity_update(w,c1,c2,iv[j],ip[j],ibp[j],initial_best_group_position)
            position_update(ip[j], iv[j], limits)
        
    print(initial_best_group_position)
        
    return initial_best_group_position,initial_best_group_cost
        
    
    
    
    
    
