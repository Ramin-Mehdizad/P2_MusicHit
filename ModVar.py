

"""
===============================================================================
 Created on Feb 15, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""


#==============================================================================
# This module holds the variables that are shared between modules
#==============================================================================
import numpy as np


#==============================================================================
# initializing parameters of classifiers
#==============================================================================
# public parameters
random_state=np.random.randint(low = 0, high = 100)
tFrac=0
cvNum=0
figdpi=0
Color_List=0
resultssubpath=''
OptVals=0
Clf_Names=0

# classifiers
RF=0
KNN=0
SVM=0
LR=0


# Rand Forest parameters
RF_VarPar=0
RF_criterion=0

# KNN parameters
KNN_VarPar=0
KNN_metric=0
KNN_p=0

# SVM
SVM_kernel=0
SVM_C=0
SVM_logVarPar=0
# here SVM_VarPar is SVM_gamma_values
SVM_VarPar=0

#Logestic regression
LR_logVarPar=0
LR_VarPar=0
LR_solver=0
LR_max_iter=0

str_input=''
args=''

MainDir=''


#==============================================================================
# initializing Flags
#==============================================================================
logFlag=False
FlagPrintSplitData=False


#==============================================================================
# initializing datasets daraframes
#==============================================================================
df_trn=0
df_tst=0
df_trn_drp=0
y_trn=0
X_trn_scaled=0

df_tst_dropped=0
tst_dropped=0
tst_data_dropped_scaled=0



#==============================================================================
# initializing split data
#==============================================================================
X_trn=0
X_tst=0
y_trn=0
y_tst=0


#==============================================================================
# feature importance variables
#==============================================================================
ExtraTrees_importance_list=0


#==============================================================================
# run times
#==============================================================================
Time_List=0


#==============================================================================
# initializing Dict variables
#==============================================================================
trnErr_A=dict()
valErr_A=dict()
trnErr_B=dict()
valErr_B=dict()
SVCTrainedModelsDict_C=dict()
SVCTrainedModelsDict_Gamma=dict()


#==============================================================================
# initializing Lists
#==============================================================================
trnErrArray_A=[]
valErrArray_A=[]
trnErrArray_B=[]
valErrArray_B=[]
x_data=[]
y_data=[]
sysinfo=[]








