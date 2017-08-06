import numpy as np

#returns indices of parent,left_child and right_child values
def parent(i):return np.floor((i-1)/2)
def left_child(i):return i*2 +1
def right_child(i):return i*2+2

#to swap two values in array
def swap(arr,current_index,new_index):
    arr[current_index],arr[new_index]=arr[new_index],arr[current_index]
    #return arr

#Bubble up a new value to correct position in the heap

def bubble_up(arr,value,index_value):
    parent_value=arr[parent(index_value)]
    while(value > parent_value):
        arr=swap(arr,index_value,parent(index_value))
        index_value=parent(index_value)
        parent_value=arr[parent(index_value)]
    
    return arr

#inserts a new value in heap at correct position
def insert(value,arr):
    arr=np.insert(arr,len(arr),value)
    arr=bubble_up(arr,value,len(arr)-1)
    return arr

#float down a value from given index_value to a correct index value
def float_down(arr,value,index_value):
    try:
        leftchild_value=arr[left_child(index_value)]
        rightchild_value=arr[right_child(index_value)]
    except:
        pass
    while(value < leftchild_value or value < rightchild_value):
        if value < leftchild_value :
            try:
                arr=swap(arr,index_value,left_child(index_value))
                index_value=left_child(index_value)
                leftchild_value=arr[left_child(index_value)]
                rightchild_value=arr[right_child(index_value)]
            except:
                break
        elif value < rightchild_value:
            try:
                arr=swap(arr,index_value,right_child(index_value))
                index_value=right_child(index_value)
                leftchild_value=arr[left_child(index_value)]
                rightchild_value=arr[right_child(index_value)]
            except:
                break
    return arr

#check if heap exists or not
def check_heap(arr):
    num=len(arr)//2
    for i in range(num):
        try:
            if arr[left_child(i)] > arr[i] or arr[right_child(i)] > arr[i]:
                return False
        except:
            pass
        else:
            pass
    return True


#convert a subtree into heap structure
def heapify(arr,index_value):
    parent_index=index_value
    if left_child(index_value) < len(arr) and arr[left_child(index_value)] > arr[parent_index]:
        parent_index=left_child(index_value)
        #print(parent_index)
    if right_child(index_value) < len(arr) and arr[right_child(index_value)] > arr[parent_index]:
        parent_index=right_child(index_value)
        #print(parent_index)
    if parent_index!=index_value:
        swap(arr,index_value,parent_index)
        heapify(arr,parent_index)

#build max heap on the given array of numbers
def build_max_heap(arr):
    for i in range(len(arr) // 2, -1, -1):
        heapify(arr, i)
        
#sort using heap       
def heapsort(arr):
    build_max_heap(arr)    
    for i in range(len(arr)-1,-1,-1):
        swap(arr,0,i)
        build_max_heap(arr[:i])
    return arr
