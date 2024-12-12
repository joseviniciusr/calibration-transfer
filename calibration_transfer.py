""""
Calibration transfer python library

By MSc. José Vinícius Ribeiro (Londrina State University - Department of Physics)

----------
General parameters:
Xm = master spectrum (reference)
Xs = source spectrum (to be used in training)
Requires only numpy and sklearn (PLSRegressor, LinearRegressor, CCA, PCA)
"""
def SSC(Xm, Xs):
    """
    **Spectral Subtraction Correction**
    
    ----------
    Xm : numpy.ndarray
    Xs : numpy.ndarray
    ----------
    Returns the transferred Xs
    """
    import numpy as np
    Xs_mean = Xs.mean(axis=0)
    Xm_mean = Xm.mean(axis=0)
    delta = Xm_mean - Xs_mean
    Xs_ssc = Xs + delta
    return Xs_ssc

def DS(Xm_cal, Xs_cal, Xs_pred=None):
    """
    **Direct Standardization (DS)**

    ----------
    Xm_cal : numpy.ndarray
    Xs_cal : numpy.ndarray
    Xs_pred : numpy.ndarray, optional
    ----------
    Returns the transfer matrix (F_ds) and the Xs_pred transferred if Xs_pred=True
    """
    import numpy as np
    F_ds = np.linalg.pinv(Xs_cal) @ Xm_cal
    if Xs_pred is not None:
        Xpred_transfer = Xs_pred @ F_ds
        return F_ds, Xpred_transfer
    else:
        return F_ds
    
def PDS(Xm_cal, Xs_cal, Xs_pred, window_length=7, regType='pls', lv=1):
    """
    **Partial Direct Standardization (PDS)**

    ----------
    Xm_cal : numpy.nparray <br>
    Xs_cal : numpy.nparray <br>
    Xs_pred : numpy.nparray <br>
    window_length : int, optional <br>
    regType : str, optional ('mlr' ou 'pls')
    lv : int, LV number if regType='pls'
    Returns the Xs_pred transferred
    """

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_decomposition import PLSRegression
    
    Modelos = []
    for i in range(Xm_cal.shape[1]):
        if i < window_length:
            X2i = Xs_cal[:, :2 * window_length + 1]
# Primeira Condição: Quando i é menor que window_length, significa que estamos nas primeiras i-variáveis
# Não da para utilizar um intervalo simetrico ao redor de i        
        elif i >= window_length and i < Xs_cal.shape[1] - window_length: #elif verifica uma condição adicional caso if seja falsa    
            X2i = Xs_cal[:, i - window_length:i + window_length + 1]
# Segunda Condição: Estamos no meio do espectro, a janela será centrada em torno de i em um intervalo simetrico 
# Assim: X2i é definido como as bandas de X2 do índice i−window_length até i+window_length      
        else: #Els executa um bloco de código quando todas as condições anteriores são falsas.
            X2i = Xs_cal[:, Xs_cal.shape[1] - 2 * window_length - 1:]
#Terceira Condição: Quando i está perto do final do espectro, a janela deve incluir as últimas bandas disponíveis. 
# Também nao da para definir intervalos simetricos ao redor de i
        if regType == 'mlr':
            model = LinearRegression().fit(X2i, Xm_cal[:, i])
        elif regType == 'pls':
            n_components = lv 
            model = PLSRegression(n_components=n_components).fit(X2i, Xm_cal[:, i])
        Modelos.append(model) 
 
    # Applying training to the new dataset
    Xtransfer = np.zeros(Xs_pred.shape)
    
    for i in range(Xs_pred.shape[1]):
        if i < window_length:
            Xi = Xs_pred[:, :2 * window_length + 1]
        elif i >= window_length and i < Xs_pred.shape[1] - window_length:
            Xi = Xs_pred[:, i - window_length:i + window_length + 1]
        else:
            Xi = Xs_pred[:, Xs_pred.shape[1] - 2 * window_length - 1:]
        Xtransfer[:,i]=Modelos[i].predict(Xi).ravel()

    return Xtransfer

def SST(Xm_cal, Xs_cal, ncomp=10, Xs_pred=None):

    """
    **Spectral Space Transformation (SST)**
    ----------
    X1 : numpy.nparray
    X2 : numpy.nparray
    ncomp : int, default=10 
    Xs_pred : numpy.ndarray
    Returns the transfer matrix and Xs_pred transferred
    """
    import numpy as np
    k = Xm_cal.shape[1]
    X_comb = np.concatenate((Xm_cal, Xs_cal), axis = 1)
    U, S, Pt = np.linalg.svd(X_comb)
    ncomp = ncomp
    U = U[:,0:ncomp] #scores
    S = S[0:ncomp]
    Pt = Pt[0:ncomp,:] #loadings
    P = Pt.T
    Pm = P[0:k,:] 
    Ps = P[k:,:]
    F_sst = np.eye(k)+np.linalg.pinv(Ps.T).dot(Pm.T - Ps.T)
    if Xs_pred is not None:
        Xpred_transfer = Xs_pred @ F_sst
        return F_sst, Xpred_transfer
    else:
        return F_sst

def GLSW(Xm_cal, Xs_cal, Xs_pred=None, alfa=1e9):

    """
    **Generalized Least Squares Weighting (GLSW)**
    ----------
    Xm_cal : np.array
    Xs_cal : np.array
    Xpred : Opcional, np.array
    alfa : int, default=1e9
    -------
    returns W (weight matrix) and Xgls (transferred Xs_pred spectrum)
    """
    import numpy as np
    n = Xs_cal.shape[0]
    m = Xs_cal.shape[1]
    X_d = (Xm_cal - Xs_cal)
    C_d = (np.dot(X_d.T, X_d)/(n - 1)) + alfa * np.eye(m)
    U, S, Vt = np.linalg.svd(C_d) #svd
    V = Vt.T
    S_diag = np.diag(S)
    S_adj = np.sqrt((S_diag * m)/np.trace(S_diag))
    W = V @ np.linalg.pinv(S_adj) @ Vt
    if Xs_pred is not None:
        Xpred_transfer = Xs_pred @ W
        return W, Xpred_transfer
    else:
        return W

def CCACT(Xm_cal, Xs_cal, Xs_pred, ncomp=2):
    """
    **Canonical Correlation Analisys Calibration Transfer (CCACT)**
    Args:
        Xm_cal: np.array
        Xs_cal: np.array
        Xs_pred: np.array

    Returns:
    X_transfer: Dataset transferred
    T: Transfer matrix
    """
    from sklearn.cross_decomposition import CCA
    import numpy as np
    cca = CCA(n_components=ncomp) 
    cca.fit(Xm_cal, Xs_cal)
    L1, L2 = cca.transform(Xm_cal, Xs_cal)
    F1 = np.linalg.pinv(L2) @ L1
    F2 = np.linalg.pinv(L1) @ Xm_cal
    W1, W2 = cca.x_weights_, cca.y_weights_
    T = W2 @ F1 @ F2
    X_transfer = Xs_pred @ T
    X_transfer = np.abs(X_transfer)
    return T, X_transfer #,corr_canonical
    
def TTFA(Xm_cal, Xs_cal, ncomp=10, Xs_pred=None):

    """
    **Target Transformation Factor Analysis (TTFA)**

    Credits to MSc. João Marcos Fávaro Lopes (Londrina State University - Department of Physics)
    
    ----------
    Xm_cal : numpy.nparray
    Xs_cal : numpy.nparray
    ncomp : int, default=10
    Xs_pred : numpy.ndarray
    -------
    Returns the Xs_cal and Xs_pred transferred
    """
    import numpy as np
    from sklearn.decomposition import PCA
    pca_Xm = PCA(n_components=ncomp)
    pca_Xs = PCA(n_components=ncomp)
    Xm_scores_cal = pca_Xm.fit_transform(Xm_cal)
    Xs_scores_cal = pca_Xs.fit_transform(Xs_cal)
    T = np.linalg.lstsq(Xs_scores_cal, Xm_scores_cal, rcond=None)[0]
    Xs_transformed_cal = np.dot(Xs_scores_cal, T)
    Xs_reconstructed_cal = pca_Xm.inverse_transform(Xs_transformed_cal)
    if Xs_pred is not None:
        # Test set
        Xs_scores_pred = pca_Xs.transform(Xs_pred)
        Xs_transformed_pred = np.dot(Xs_scores_pred, T)
        Xs_reconstructed_pred = pca_Xm.inverse_transform(Xs_transformed_pred)

        return Xs_reconstructed_cal, Xs_reconstructed_pred
    else:
        return Xs_reconstructed_cal
