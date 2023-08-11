# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 20:26:25 2021

@author: Noshin
"""
import numpy as np
#%%
#your own implementation using formula of linear algebra
def transpose_matrix(mat):
    t_matrix = [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]
    return t_matrix



def multiplication(x,xT):
    result = [[sum(a * b for a, b in zip(A_row, B_col))
                        for B_col in zip(*x)]
                                for A_row in xT]
    return result



def cofactor(mat):
    cofactor_mat = []
    Sum = 0
    for i in range(len(mat[0])):
        a = []
        for j in range(len(mat)):
            cofactor_values = [row[: j] + row[j+1:] for row in (mat[: i] + mat[i+1:])] 
                
# - - - - cofactor - - - - - - - -
            subdet_ans = (-1) ** (j + i) * np.linalg.det(cofactor_values)
            a.append(subdet_ans) #inner loop
        
        cofactor_mat.append(a)
    return cofactor_mat
    


def predict(x,weight):
    predicty=[]
    for i in range(len(x)):
        a = -1
        b = 0
        for j in range(len(x[0])):
            a += 1
            b += (x[i][j] * weight[a])
        predicty.append(b)        
    return predicty
            
def mean(List):
    m = 0
    length = len(List)
    for i in List:
        m += i
    Mean = m/length
    return Mean

    
            
    #return x     
def multiple_rgression(x,y):
    
    # - - - - - - - - - - - transpose x - - - - - - - - - - - - - - - - - - - - - - - - 
    t_matrix = transpose_matrix(x)
    
    # - - - - - - - - - - - multipication of x^T*x and x^T*y - - - - - - - - - - - - - - - - - - - - - - - - 
    mul_xT_x = multiplication(x, t_matrix)
    mul_xT_y = multiplication(y, t_matrix)
    
    # - - - - - - - - - - - determinant - - - - - - - - - - - - - - - - - - - - - - - - 
    det = np.linalg.det(mul_xT_x)
    
    # - - - - - - - - - - - cofactor matrix of x and transpose that - - - - - - - - - - - - - - - - - - - - - - - - 
    cofactor_mat = cofactor(mul_xT_x)
    t_cofactor = transpose_matrix(cofactor_mat)
    
    # - - - - - - - - - - - inverse matrix of x^T*x- - - - - - - - - - - - - - - - - - - - - - - - 
    inverse_mat = [[(t_cofactor[i][j])*(1/det) for j in range(len(t_cofactor))] for i in range(len(t_cofactor[0]))]
    
    # - - - - - - - - - - - weight = inverse matrix of x^T*x * x^T*y - - - - - - - - - - - - - - - - - - - - - - - - 
    weight = multiplication(mul_xT_y, inverse_mat)
       
    # - - - - - - - - - - - flatten the weight for necessity - - - - - - - - - - - - - - - - - - - - - - - - 
    w = [j for sub in weight for j in sub]
    
    # - - - - - - - - - - - predicted value of y - - - - - - - - - - - - - - - - - - - - - - - - 
    y_predict = predict(x,w)
    return y_predict
    #print(y_predict)
