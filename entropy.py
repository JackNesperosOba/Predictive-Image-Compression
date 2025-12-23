import numpy as np

def entropia_0(img):
    
    pixels = img.ravel()#1D
    valores, frec = np.unique(pixels, return_counts=True)
    
    probabilitats = frec / np.sum(frec)
  
    entropia = -np.sum(probabilitats * np.log2(probabilitats))

    return entropia 

def entropia_1_espacial(img, components):

    H = 0  
    H_cond_total = 0
    for x in range(components):
        pixels = img[:,:,x]
        causal = pixels[:, :-1].ravel()    # píxeles de la izquierda
        actual = pixels[:, 1:].ravel()     # píxeles actuales
        
        H_Y = entropia_0(causal)# entropia pixel causal H(Y)
        # print(f"H(Y): {H_Y}")

        # Entropía conjunta H(X,Y)
        pares = np.array(list(zip(causal, actual)))
       
        valores, frecuencia = np.unique(pares, axis=0, return_counts=True)
        
        # Probabilidades conjuntas
        p = frecuencia / np.sum(frecuencia)
        H_XY = -np.sum(p * np.log2(p))
        # print(f"H(X,Y): {H_XY}")
        
        # Entropía condicional H(X|Y) = H(X,Y) - H(Y)
        H = H_XY - H_Y # H(X,Y) >= H(Y)
        # print(f"Entropia banda {x}: {H}")
        H_cond_total += H
    
    H_cond_total = H_cond_total / (components)
        
    return H_cond_total 

def entropia_1_intercanal(img, components):

    H = 0  
    H_cond_total = 0
    for x in range(components - 1):
        causal = img[:,:, x].ravel()        # píxeles de comp actual izquierda
        actual = img[:,:, x + 1].ravel()     # píxeles de comp sig
        
        H_Y = entropia_0(causal)# entropia pixel causal H(Y)
        # print(f"H(Y): {H_Y}")

        # Entropía conjunta H(X,Y)
        pares = np.array(list(zip(causal, actual)))
       
        valores, frecuencia = np.unique(pares, axis=0, return_counts=True)
        
        # Probabilidades conjuntas
        p = frecuencia / np.sum(frecuencia)
        H_XY = -np.sum(p * np.log2(p))
        # print(f"H(X,Y): {H_XY}")
        
        # Entropía condicional H(X|Y) = H(X,Y) - H(Y)
        H = H_XY - H_Y # H(X,Y) >= H(Y)
        # print(f"Entropia banda {x}: {H}")
        H_cond_total += H
    
    N = components - 1
    if components == 1:
        N = components
    
    H_cond_total = H_cond_total / N
        
    return H_cond_total