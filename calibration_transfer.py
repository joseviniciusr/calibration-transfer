""""
Biblioteca para realizar transferência de calibração
Xm = espectro mester (referência)
Xs = espectro slave (a ser utilizado no treinamento)
Requer apenas numpy e sklearn (PLSRegressor, LinearRegressor, CCA)
"""
def SSC(Xm, Xs):
    """
    **Spectral Subtraction Correction**
    ----------
    Xm : numpy.ndarray <br
        Espectro mestre <br>
    Xs : numpy.ndarray <br>
        Espectro slave <br>
    ---------- <br>
    Retorna o Xs transferido
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
    Xm_cal : numpy.ndarray <br
        Espectro mestre - cal<br>
    Xs_cal : numpy.ndarray <br>
        Espectro slave - cal<br>
    Xs_pred : numpy.ndarray, optional <br>
        Espectro slave independente a ser transferido <br>
    ---------- <br>
    Retorna a matriz de transferência F_ds, e o espectro Xs_pred transferido se Xs_pred=True

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
        O espectro mester - cal <br>   
    Xs_cal : numpy.nparray <br>
        O espectro slave - cal <br>
    Xs_pred : numpy.nparray <br>
        Os novos dados slaves que serão transformados <br>
    window_length : int, optional <br>
       Tamanho da janela que será utilizada para mais e para menos <br>
    regType : str, optional <br>
        tipo de regressão ('mlr' ou 'pls'). Padrão 'pls' <br>
    lv : numero de LVs se o pls for escolhido <br>
    
    Retorna: Xs_pred transferido : numpy.ndarray <br>
        O espectro de teste transferido
    """

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_decomposition import PLSRegression
    
    Modelos = []
    
    # Gerando os modelos
    for i in range(Xm_cal.shape[1]):
        if i < window_length: #If verifica uma condição e executa um bloco de código se a condição for verdadeira.
            X2i = Xs_cal[:, :2 * window_length + 1]
# Primeira Condição: Quando i é menor que window_length, significa que estamos nas primeiras i-variáveis
# Não da para utilizar um intervalo simetrico ao redor de i
# Portanto, a janela deve incluir todas as bandas até o limite que é possível.
# Assim, X2i é configurado para incluir as primeiras 2 * window_length + 1 variaveis do espectro slave. 
# Ex: window_length = 7, as primeiras 7 i-variaveis de Xm estarão relacionadas com as 2 * 7 + 1 = 15 primeiras variaveis de Xs
        
        elif i >= window_length and i < Xs_cal.shape[1] - window_length: #elif verifica uma condição adicional caso if seja falsa    
            X2i = Xs_cal[:, i - window_length:i + window_length + 1]
# Segunda Condição: Estamos no meio do espectro, a janela será centrada em torno de i em um intervalo simetrico 
# Assim: X2i é definido como as bandas de X2 do índice i−window_length até i+window_length
# Serão 2∗window_length+1 bandas 
# Ex: window_length=7, teremos a i-ésima variavel de Xm estará relacionada com banda que vai desde i-7, i, ..., 1+7 de Xs               
        
        else: #Els executa um bloco de código quando todas as condições anteriores são falsas.
            X2i = Xs_cal[:, Xs_cal.shape[1] - 2 * window_length - 1:]
#Terceira Condição: Quando i está perto do final do espectro, a janela deve incluir as últimas bandas disponíveis. 
# Também nao da para definir intervalos simetricos ao redor de i
# Então: X2i é configurado para pegar as últimas 2 * window_length + 1 variaveis de Xs e definir como bandas 
# Ex: window_length=7, a i-ésima variavel de Xm que cair aqui estará relacionada com as ultimas 15 variaveis de Xs
# a notação [X2.shape[1] - 2 * window_length - 1:] indica da variavel X2.shape[1] - 2 * window_length - 1 até as ultimas do array

# Escolhendo o tipo de regressão
# Estamos treinando um modelo de regressão com os dados X2i como variáveis de entrada e X1[:, i] como a variável de saída.
# Em cada iteração do loop, X2i contém as bandas espectrais vizinhas (janela) de X2 ao redor da banda i
# O formato de X2i é uma matriz de dimensões (n_amostras, janela_de_bandas)
        if regType == 'mlr':
            model = LinearRegression().fit(X2i, Xm_cal[:, i]) #X1[:, i][:, None] se necessario para transformar X1 em matriz coluna e nao vetor unidimensional
            #F[:, i] = model.coef_.ravel()
        elif regType == 'pls':
            n_components = lv  # O padrão é 1
            model = PLSRegression(n_components=n_components).fit(X2i, Xm_cal[:, i]) #X1[:, i][:, None]
        Modelos.append(model) 
        
        # model_i = Models[i] para acessar o modelo ajustado pra banda i
        # coef_i = model_i.coef_ Acessa os coeficientes do modelo da banda i
        # ex: model_i.predict(X2i) faz previsões para a banda i
    """
    Fluxo de cada iteração:

    1. Janela das variáveis de entrada: Uma janela ao redor da banda i de X2 (definida pelo window_length) é selecionada. 
    Essa janela tem 2 * window_length + 1 variáveis, ou seja, bandas vizinhas de X2.

    2. Variável de saída: A banda i de X1 é escolhida como a variável dependente

    3. Treinamento do modelo: O modelo de regressão é ajustado com as variáveis de entrada X2i e a variável de saída X1[:, i]

    4. Repetição: O processo é repetido para cada banda espectral i, ajustando um modelo de regressão para cada uma.
    
    5. Armazenamento do i-esimo modelo: O método Model.append(model) armazena model à lista Modelos, lado a lado.
    Com isso, cada modelo ajustado (um para cada banda i) é armazenado em uma lista

    RESUMO: 
    Criação da janela de bandas: Para a banda i, criamos X2i, uma matriz contendo uma janela de bandas vizinhas de X2.
    Treinamento do modelo: O modelo de regressão (MLR ou PLS) é ajustado para prever a banda i de X1 (espectro master) a partir das bandas vizinhas de X2 (espectro slave).
    Armazenamento do modelo: O modelo ajustado para a banda i é adicionado à lista Models com o comando Models.append(model).
    Repetição: Esse processo é repetido para cada banda espectral, criando uma coleção de modelos que mapeiam X2 para X1.
    """

# Aplicando o treinamento ao novo conjunto de dados
    Xtransfer = np.zeros(Xs_pred.shape)
    
    for i in range(Xs_pred.shape[1]):
# Aqui o loop acessa as bandas da mesma maneira que lá em cima
        if i < window_length:
            Xi = Xs_pred[:, :2 * window_length + 1]
# Primeiro itera sobre as i-ésimas variaveis < window_length 
# Nesse caso ele vai escolher as 2 * window_length + 1 primeiras variaveis até que i > window_length            
        elif i >= window_length and i < Xs_pred.shape[1] - window_length:
            Xi = Xs_pred[:, i - window_length:i + window_length + 1]
# Quando i > window_length, podemos pegar um intervalo simétrico em torno de i
# Vamos iterar de i - window_length até i + window_length            
        else:
            Xi = Xs_pred[:, Xs_pred.shape[1] - 2 * window_length - 1:]
# No ultimo caso estamos no finalzinho do espectro, as ultimas variaveis
# Também nao da para definir um intervalo simetrico em torno de i
# Resta utilizar as ultimas 2 * window_length -1 variaveis
# A iteração segue assim até a ultima variavel i            

#Em cada caso desses, a i-ésima variavel do espectro transferido é calculada pelo i-ésimo modelos
# O modelo treinado para a i-ésima variavel é eplicado na i-ésima variavel da matrix Xpred
        Xtransfer[:,i]=Modelos[i].predict(Xi).ravel()#ravel deixa o vetor gerado 1D para concatená-los certo 

    return Xtransfer #retorna o espectro Xpred transferido

def SST(Xm_cal, Xs_cal, ncomp=10, Xs_pred=None):

    """
    **Spectral Space Transformation (SST)**
    ----------
    X1 : numpy.nparray <br>
        O espectro mester - cal <br>
    X2 : numpy.nparray <br>
        O espectro slave - cal <br>
    ncomp : int, padrao=10 <br>
        numero de componentes incluidas na svd <br>
    Xs_pred : numpy.ndarray <br>
        O espectro slave de teste transferido
    Retorna : A matriz de transferencia, Xs_pred transferido    
    """
    import numpy as np
    
    k = Xm_cal.shape[1] #para identificar o numero de variaveis da matriz master

#X combinado
    X_comb = np.concatenate((Xm_cal, Xs_cal), axis = 1) #Xlab e Xinsitu devem ter os mesmos nomes nas colunas

#svd
    U, S, Pt = np.linalg.svd(X_comb)

#ncomp = numero de componentes escolhidas (PCs) após a svd
    ncomp = ncomp

    U = U[:,0:ncomp] #scores
    S = S[0:ncomp] #matriz diagonal
    Pt = Pt[0:ncomp,:] #loadings
    P = Pt.T

    Pm = P[0:k,:] #loadings da matriz master
    Ps = P[k:,:] #loadings da matriz slave

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
    Xm_cal : np.array <br>
        O espectro mester - cal<br>
    Xs_cal : np.array <br>
        O espectro slave - cal<br>
    Xpred : Opcional, np.array <br>
        O novo slave espectro a ser transferido    
    alfa : int, padrao=1e9 <br>
        coeficiente, multiplica a identidade no calculo da matriz de covariância<br>
    -------
    retorna W (matriz de pesos) e Xgls (espectro Xs_pred transferido)
    """

    import numpy as np
    
    n = Xs_cal.shape[0] #numero de amostras
    m = Xs_cal.shape[1] #numero de variaveis da matriz slave

    # matriz de diferença centrada na média
    X_d = (Xm_cal - Xs_cal)

    #matriz de covariancia
    C_d = (np.dot(X_d.T, X_d)/(n - 1)) + alfa * np.eye(m) #np.eye retorna um array 2D - matriz identidade

    #svd na matriz de covariancia
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
        Xm_cal: Conjunto de dados mestre - cal. np.array
        Xs_cal: Conjunto de dados slave - cal. np.array
        Xs_pred: Conjunto slave de dados a ser transferido. np.array

    Returns:
        X_transfer: Conjunto de dados transferido.
        T: matriz de transferência 
    """
    from sklearn.cross_decomposition import CCA
    import numpy as np

    # Cria um objeto CCA
    cca = CCA(n_components=ncomp)  # Ajustar o número de componentes conforme necessário

    # Ajusta o modelo aos dados
    cca.fit(Xm_cal, Xs_cal)

    # Obtém as projeções canônicas
    L1, L2 = cca.transform(Xm_cal, Xs_cal)

    # Calcula as matrizes F1 e F2
    F1 = np.linalg.pinv(L2) @ L1
    F2 = np.linalg.pinv(L1) @ Xm_cal

    # Calcula a matriz de transferência T
    W1, W2 = cca.x_weights_, cca.y_weights_
    T = W2 @ F1 @ F2

    # Transforma o conjunto de dados Bp
    X_transfer = Xs_pred @ T
    X_transfer = np.abs(X_transfer)
    #corr_canonical = cca.score(X1, X_transfer)

    return T, X_transfer #,corr_canonical

def TOP(X_tuple, Xs_pred=None, ncomp=1): 
    """
    **Transfer by orthogonal projection (TOP)**
    Args:
        X_tuple: np.array, uma tupla com as matrizes mester e slave de calibração (Xm, Xs)
        Xs_pred: Conjunto de dados a ser transferido. np.array
        ncomp: numero de componentes principais as quais as dados vão ser projetados e as componentes ortogonais serao extraidas
        RetornA:
        XS_transfer: Componentes ortogonais as PCs 
        F: matriz que projeta os dados ortogonalmente as PCs
        Baseado: A. Andrew and T. Fearn, “Transfer by orthogonal projection: Making near-infrared calibrations robust to between-instrument variation,” Chemom. Intell. Lab. Syst., vol. 72, no. 1, pp. 51–56, 2004, doi: 10.1016/j.chemolab.2004.02.004.

    """    
    import numpy as np
    # Xtuple = (X1, X2)
    m = len(X_tuple) #a tupla armazena os dois espectros em um mesmo arquivo
    k = X_tuple[0].shape[1] #dimensao das colunas do primeiro espectro


    R = np.zeros((m,k)) #a matriz R é inicializada como uma matriz de zeros com dimensões (m, k)

    for mi in range(m):

        R[mi,:] = X_tuple[mi].mean(axis=0) #O loop percorre cada matriz em X_tuple e calcula a média ao longo do eixo 0 (ou seja, a média das linhas para cada coluna)
                                           #O resultado é armazenado na linha correspondente da matriz R
    # Após a execução do loop, a matriz R conterá as médias das colunas de cada uma das matrizes em X_tuple
    # Cada linha de R representa a média das colunas da matriz correspondente em X_tuple

    U0,S,V0t = np.linalg.svd(R)
    #S_matrix = np.zeros((top_ncp,top_ncp))
    #S_matrix[0:top_ncp,:][:,0:top_ncp] = np.diag(S[0:top_ncp])
    #S_matrix = np.diag(S)
    V = V0t[0:ncomp].T #loadings da matriz R de acordo com o numero de componentes escolhidas (top_ncp)
    #U = U0[:,0:ncomp]

    F = np.identity(n=V.shape[0]) - V.dot(V.T) 
    #O produto matricial V.dot(V.T) resulta em uma matriz que representa a projeção nas direções definidas pelos vetores V mais significativos.
    #A matriz F é construída subtraindo da matriza projeção das componentes principais (representada por V.dot(V.T)) da matriz identidade. 
    # Isso resulta em uma matriz que captura as variações nos dados que não estão alinhadas com essas componentes principais.
    #Em termos geométricos, F pode ser interpretada como um operador que projeta os dados ortogonalmente às componentes principais.
    if Xs_pred is not None:
        Xpred_transfer = Xs_pred @ F
        return F, Xpred_transfer
    else:
        return F
    
def TTFA(Xm_cal, Xs_cal, ncomp=10, Xs_pred=None):

    """
    **Target Transformation Factor Analysis (TTFA)**
    ----------
    Xm_cal : numpy.nparray <br>
        O espectro mester de cal <br>
    Xs : numpy.nparray <br>
        O espectro slave de cal <br>
    ncomp : int, padrao=10 <br>
        numero de componentes incluidas na pca <br>
    Xs : numpy.ndarray <br>
        O espectro slave de teste
    -------
    retorna o espectro slave transferido de calibração e de predição
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
        # conjunto teste
        Xs_scores_pred = pca_Xs.transform(Xs_pred)
        Xs_transformed_pred = np.dot(Xs_scores_pred, T)
        Xs_reconstructed_pred = pca_Xm.inverse_transform(Xs_transformed_pred)

        return Xs_reconstructed_cal, Xs_reconstructed_pred
    else:
        return Xs_reconstructed_cal

    

