
"""
===============================================================================
 Created on Feb 15, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""

#==============================================================================
# importing standard classes
#==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import os.path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import argparse
import sys


#==============================================================================
# importing module codes
#==============================================================================
import ModVar as Var
import ModFunc as Func


#==============================================================================
# this function asks user input data
#==============================================================================  
def Input_Data_Message():
    
    print('')
    print('|===========================================================')
    print('|  ==> To run the code with default values, just press Enter')
    print('|  ==> Otherwise:')
    print('|  ==> Enter the parameters as following format:')
    print('|')
    print('|  -d D:/.../*.csv -t D:/.../*.csv -w D:/... -p 0 -l 0 -r 1')
    print('|')
    print('|  ==> To get help, type "-h" and press Enter')
    print('|  ==> To exit program, type "Q" and press Enter')
    print('|===========================================================')
    
    Var.str_input=input('  Enter parameters: ').strip()

    
#==============================================================================
# this function asks user input data
#==============================================================================
def Call_Parser():
    
    # create parse class
    parser1 = argparse.ArgumentParser(add_help=True,prog='Music_Hit Program',
             description='* This program analyzes Music_Hit by use of\
                 several classifiers*')
    
    # set program version
    parser1.add_argument('-v','--version',action='version',
                        version='%(prog)s 1.0')

    parser1.add_argument('-f', '--ScriptPath', action='store',
                        default=Var.MainDir,
                        dest='ScriptPath',help='Shows Script File Address')
    
    # set train Data file address
    parser1.add_argument('-d', '--TrainPath', action='store',
                        default=Var.MainDir+'\MusicHitTrainData.csv',
                        dest='TrainDataPath',help='Enter .csv file address')
    
    # set test data file address
    parser1.add_argument('-t', '--TestPath', action='store',
                        default=Var.MainDir+'\MusicHitTestData.csv',
                        dest='TestDataPath',help='Enter .csv file address')
    
    # set work directory
    parser1.add_argument('-w', '--wDirAd', action='store',
                        default=Var.MainDir,
                        dest='WorkDir',help='Enter work Directory')
    
    # whether to save plots or not
    parser1.add_argument('-p', '--SavePlot', action='store', 
                         default='1',  dest='SavePlot', choices=['0', '1'],
                         help='0: Dont Save plots     1: Save plots')
    
    # whether to create log file or not
    parser1.add_argument('-l', '--log', action='store',
                         default='1', dest='logFile', choices=['0', '1'],
                         help='0: Dont write logfile     1: write logfile')
    
    # whether to create report.docx file or not
    parser1.add_argument('-r', '--report', action='store',
                         default='1', dest='ReportFile', choices=['0', '1'],
            help='0: Dont write report file     1: write report file')
    
    
    # indicates when to exit while loop
    entry=False
    while entry==False:
        # initialize
        ParsErr=0
        FileErr=0
        makedirErr=0
        
        # --------------in this section we try to parse successfully-----------
        # function to call input data from command line    
        Func.Input_Data_Message()
        
        # user wanted to continue with default values
        if Var.str_input=='':
            Var.args=parser1.parse_args()
            # exit while loop
            entry=True
        elif Var.str_input.upper()=='Q':
            # exit script
            sys.exit()
        else:
            entry=True
            ParsErr=0
            try:
                Var.args=parser1.parse_args(Var.str_input.split(' '))
            except:
                entry=False
                ParsErr=1
        #----------------------------------------------------------------------
        
        
        #-------------After having parsed successfully, we coninue-------------
        # continue if parse was done successfully
        if ParsErr==0:  
            #check if train data base file exists
            TrainFileErr=0
            if os.path.isfile(Var.args.TrainDataPath):
                pass
            else:
                print("Train Data file address doesn't exist.")
                print('Enter a valid file address.')
                entry=False
                TrainFileErr=1
                
            # continue if train data file exists
            if TrainFileErr==0:  
                #check if test data base file exists
                TestFileErr=0
                if os.path.isfile(Var.args.TestDataPath):
                    pass
                else:
                    print("Test Data file address doesn't exist.")
                    print('Enter a valid file address.')
                    entry=False
                    TestFileErr=1
            
                # continue if test file address is correct
                if TestFileErr==0:
                    #check for work dir. if not exist, create it
                    if os.path.exists(Var.args.WorkDir):
                        pass
                    else:
                        makedirErr=0
                        try:
                            # make work dir 
                            os.mkdir(Var.args.WorkDir)
                        except OSError:  
                            print("!!!**Work Dir doesn't exist and couldn't be created**!!!")
                            print("Try another directory again")
                            entry=False
                            makedirErr=1
                        except:  
                            print("!!!**Work Dir doesn't exist and couldn't be created**!!!")
                            print("Try another directory again")
                            entry=False
                            makedirErr=1
                        
                        if makedirErr==0:   
                            os.chdir(Var.args.WorkDir)
                        
                        # figs must be saved for report file
                        if Var.args.ReportFile==1:
                            Var.args.SavePlot=1

        #----------------------------------------------------------------------


#==============================================================================
# this function creates and sets result directory
#==============================================================================   
def CreateAndSetResultsDir():
    global path
    
    # the work directory is set in arg-parse
    # get current path
    path=Var.args.WorkDir 
    
    # creating results directory and avoid to override the previous results
    i=1
    ResFldExst=True
    while ResFldExst==True:
        resultspath = path+"\\Results_Run_"+str(i)
        ResFldExst=os.path.exists(resultspath)
        i+=1
    Var.resultssubpath=resultspath
    # no we create results sub folder and set it as result path
    os.mkdir(Var.resultssubpath)
    os.chdir(Var.resultssubpath)
  

#==============================================================================
# this function inputs the parameters for Part A of problem 
#==============================================================================
def CreateSubPlotStructure_PartA(x):
    global axes,cmap,xMesh,yMesh,fig_SubPlotC
    
    nrows=len(Var.C_values)//3  if len(Var.C_values)%3==0 else len(Var.C_values)//3+1
    fig_SubPlotC,axes=plt.subplots(nrows=nrows,ncols=3,figsize=(15,5.0*nrows))
    cmap=ListedColormap(['#b30065','#178000'])
    
    xMin,xMax=x[:,0].min()-1,x[:,0].max()+1
    yMin,yMax=x[:,1].min()-1,x[:,1].max()+1
    xMesh,yMesh=np.meshgrid(np.arange(xMin,xMax,0.01),np.arange(yMin,yMax,0.01))


#==============================================================================
# this function sets the classifier parameters of problem
#==============================================================================
def ClassifierParameters():
    
    # public parameters
    Var.OptVals= {'RF':20,'KNN':13,'SVM':0.1,'LR':1}
    Var.Clf_Names=['RF','KNN','SVM','LR']
    Var.random_state=np.random.randint(low = 0, high = 100)
    Var.tFrac=0.2
    Var.cvNum=5
    Var.figdpi=100
    Color_List=['orange', 'g', 'r', 'c', 'm', 'y', 'k','yellow'
                          ,'cyan','b','g', 'r', 'c', 'm','yellow',
                          'cyan','orange', 'g',"darkorange"]
    Var.Color_List=Color_List+Color_List
    Var.Time_List=list()
    
    # Rand Forest parameters
    Var.RF_VarPar=[1,2,3,4,5,7,9,11,13,15,20,30,40,50]
    # Var.RF_VarPar=[1,2,13,15,20,30]

    # Rand Forest criterion
    Var.RF_criterion='entropy'
    
    # KNN parameters
    Var.KNN_VarPar=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25]
    # Var.KNN_VarPar=[1,2,10,11,12,13,14,15]
    Var.KNN_metric='minkowski'
    Var.KNN_p=2
    
    # SVM
    Var.SVM_kernel='rbf'
    Var.SVM_C=1
    Var.SVM_logVarPar=np.arange(-3.5, 1, 0.25)
    # Var.SVM_logVarPar=np.arange(-1, 1.5, 0.5)
    Var.SVM_VarPar=np.power(10.0, Var.SVM_logVarPar)
    
    #Logestic regression
    Var.LR_logVarPar=np.arange(-4, 2, 0.5)
    # Var.LR_logVarPar=np.arange(-1, 2, 0.5)
    Var.LR_VarPar=np.power(10.0, Var.LR_logVarPar)
    Var.LR_solver='lbfgs'
    Var.LR_max_iter=100
    
    
 #==============================================================================
# this function inputs the parameters for Part B of problem 
#==============================================================================
def CreateSubPlotStructure_PartB(x):
    global axes,cmap,xMesh,yMesh,fig_SubPlotGamma
    
    nrows=len(Var.GAMMA_values)//3  if len(Var.GAMMA_values)%3==0 else len(Var.GAMMA_values)//3+1
    fig_SubPlotGamma,axes=plt.subplots(nrows=nrows,ncols=3,figsize=(15,5.0*nrows))
    cmap=ListedColormap(['#b30065','#178000'])
    
    xMin,xMax=x[:,0].min()-1,x[:,0].max()+1
    yMin,yMax=x[:,1].min()-1,x[:,1].max()+1
    xMesh,yMesh=np.meshgrid(np.arange(xMin,xMax,0.01),np.arange(yMin,yMax,0.01))


#==============================================================================
# this function calculate feature importance using ExtraTreesClassifier 
#==============================================================================
def Feature_importance_ExtraTrees():
    
    n_estimators=[1,3,5,10,15,20,50]
    importance_list=list()
    
    Var.ExtraTrees_importance_list=list()
    
    for N in n_estimators:
        # Build a forest and compute the feature importances
        forest = ExtraTreesClassifier(n_estimators=N, 
                                      random_state=Var.random_state)
        
        forest.fit(Var.X_trn_scaled, Var.y_trn)
        
        importances = forest.feature_importances_
        importance_list.append([np.arange(Var.X_trn_scaled.shape[1]),
                                importances])
    
        
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        
        # Plot bar the feature importances of the forest
        if N==n_estimators[-1]:
            Feature_Impo_BarPlot=plt.figure()
            plt.title("Feature importance by ExtraTrees")
            plt.bar(range(Var.X_trn_scaled.shape[1]), importances[indices],
                   color='yellow', yerr=std[indices], align="center")
            # plt.xticks(range(X_trn_scaled.shape[1]), indices)
            plt.xticks(range(Var.X_trn_scaled.shape[1]), 
                       [t for t in Var.df_trn_drp.columns[indices]],
                           rotation=80)
            plt.xlim([-1, Var.X_trn_scaled.shape[1]])
            plt.ylabel('Importance Score')
        plt.show()
    
    
    Feature_Impo_Plot=plt.figure()
   
    for i in range(len(n_estimators)):
        feature_ID=importance_list[i][0]+1
        plt.plot(feature_ID, importance_list[i][1],'--',
           color=Var.Color_List[i],
           label='n={}'.format(n_estimators[i]))
    
    plt.title("Feature importances by ExtraTrees")
    plt.xticks(range( Var.X_trn_scaled.shape[1]+1))
    plt.xlim([0, Var.X_trn_scaled.shape[1]+1])
    plt.legend(loc='best')
    plt.xlabel('Feature No.')
    plt.ylabel('Importance Score')
    plt.show()
    
    if Var.saveplotFlag==True:
        Feature_Impo_BarPlot.savefig('Imp_1.jpg', 
                     dpi=Var.figdpi, facecolor='w')            
        Feature_Impo_Plot.savefig('Imp_2.jpg', 
                     dpi=Var.figdpi, facecolor='w', edgecolor='r')
        

#==============================================================================
# this function calculate feature importance using PCA 
#==============================================================================
def Feature_importance_PCA():
    
    Feat_Names = list(Var.df_trn_drp.keys())
    # PCA analysis
    n_Features=len(Var.df_trn_drp.columns)
    # number of components kept
    n_pcs= n_Features
    pca = PCA(n_components=n_pcs)
    pca.fit_transform(Var.X_trn_scaled)
    
#    X_trn_scaled=pca.fit_transform(Var.X_trn_scaled)
#    X_tst_scaled=pca.fit_transform(X_tst_scaled)
    
    
    print('pca.components_',pca.components_,end='\n\n')
    print('pca.explained_variance_',pca.explained_variance_,end='\n\n')
    var_ratio=pca.explained_variance_ratio_
    print('var_ratio   ',var_ratio,end='\n\n')
    #print('pca.singular_values_',pca.singular_values_,end='\n\n')

    # get most important feature names
    print('n_pcs',n_pcs,end='\n\n')
    percent=var_ratio[0:n_pcs].sum()
    print('percent ',percent*100,end='\n\n')
    
    # get the index of the most important feature on EACH component i.e. largest 
    #absolute value using LIST COMPREHENSION HERE
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    print('most_important',most_important,end='\n\n')
    
    # get the names
    most_important_names = [Feat_Names[most_important[i]] for i in range(n_pcs)]
    print('most_important_names',most_important_names,end='\n\n')
    
    # using LIST COMPREHENSION HERE AGAIN
    dic = {'PC{:2.0f}'.format(i+1): most_important_names[i] for i in range(n_pcs)}
    print('dic',dic,end='\n\n')
    
    # build the dataframe
    df_most_important = pd.DataFrame(sorted(dic.items()))
    
    print('df_most_important',df_most_important,end='\n\n')
   
    
    # ---------------------plot variance ratio---------------------------------
    x=np.arange(len(pca.explained_variance_))+1
    width=0.5
    
    fig_PCA_var, ax = plt.subplots()
    fig1=plt.gca()
    
    rects1=plt.bar(x, pca.explained_variance_ratio_, color='y',width = width)
    
    var_pct=list(pca.explained_variance_ratio_*100)
    
    def autolabel(rects):
        # Attach a text label above each bar in *rects*, displaying its height
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{:.0f}%'.format(height*100),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    
    plt.title('PCA (variance ratio)')
    plt.ylabel('Score')
    fig1.set_xticks(x)
    fig1.set_xticklabels(most_important_names,  rotation=80)
    plt.show()
    # -------------------------------------------------------------------------
    
 
    # -------------------------- plot variance --------------------------------
    fig_PCA_varRatio=plt.figure()
    fig1=plt.gca()
    
    rects2=plt.bar(x, pca.explained_variance_, color='c', width = width)
    
    plt.title("PCA ( variance )")
    plt.ylabel('Score')
    fig1.set_xticks(x)
    fig1.set_xticklabels(most_important_names,  rotation=80)
    plt.show()
    
    if Var.saveplotFlag==True:
        fig_PCA_var.savefig('PCA_1.jpg',
                            dpi=Var.figdpi, facecolor='w')            
        fig_PCA_varRatio.savefig('PCA_2.jpg',
                        dpi=Var.figdpi, facecolor='w', edgecolor='r')
    #--------------------------------------------------------------------------    

 
#==============================================================================
# this function reduces features using PCA
#==============================================================================
def Feature_reduction_PCA():

    # number of components kept
    n_pcs= 10
    pca = PCA(n_components=n_pcs)
    
    Var.X_trn_scaled=pca.fit_transform(Var.X_trn_scaled)
   

#==============================================================================
# this function gets computer spec
#==============================================================================
def GetSysInfo():
    import platform,socket,re,uuid,psutil
    try:
        Var.sysinfo.append(['platform',platform.system()]) 
        Var.sysinfo.append(['platform-release',platform.release()])
        Var.sysinfo.append(['platform-version',platform.version()])
        Var.sysinfo.append(['architecture',platform.machine()])
        Var.sysinfo.append(['hostname',socket.gethostname()])
        Var.sysinfo.append(['ip-address',socket.gethostbyname(socket.gethostname())])
        Var.sysinfo.append(['mac-address',':'.join(re.findall('..', '%012x' % uuid.getnode()))])
        Var.sysinfo.append(['processor',platform.processor()])
        Var.sysinfo.append(['ram',str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"])
    
    except Exception as e:
        print(e)
        
 
#==============================================================================
# this function plots train and validation errors of PartA
#==============================================================================
def PltTrnValErr_PartA():
    global fig_TrnVal_PartA
    
    # initializethe lists
    trnErrList_A=[]
    valErrList_A=[]
    
    # extracting data from Dict to List
    fig_TrnVal_PartA=plt.figure(80)
    for i,(xx,yy) in enumerate(Var.trnErr_A.items()):
        trnErrList_A.append([xx,yy])
        
    for i,(xx,yy) in enumerate(Var.valErr_A.items()):
        valErrList_A.append([xx,yy])
        
    # we need to sort because they are added to the list as soon as they finish
    # using sort() + lambda to sort list of list (sort by second index) 
    trnErrList_A.sort(key=lambda trnErrList_A:trnErrList_A[0]) 
    valErrList_A.sort(key=lambda valErrList_A:valErrList_A[0]) 
    
    Var.trnErrArray_A=np.array(trnErrList_A)
    Var.valErrArray_A=np.array(valErrList_A)

    #plotting train validation plot
    plt.plot(np.log10(Var.trnErrArray_A[:,0]),Var.trnErrArray_A[:,1],'*',color='black')
    plt.plot(np.log10(Var.valErrArray_A[:,0]),Var.valErrArray_A[:,1],'*',color='black') 
    
    plt.plot(np.log10(Var.trnErrArray_A[:,0]),Var.trnErrArray_A[:,1],'-',color='blue', label='train') 
    plt.plot(np.log10(Var.valErrArray_A[:,0]),Var.valErrArray_A[:,1],'-',color='red', label='validation')
    
    plt.legend(loc='lower right',frameon=False)
    plt.xlabel('log(C values)') 
    plt.ylabel('Accuracy') 
    
    
#==============================================================================
# this function plots train and validation errors of Part B
#==============================================================================
def PltTrnValErr_PartB():
    global fig_TrnVal_PartB
    
    # initializethe lists
    trnErrList_B=[]
    valErrList_B=[]
    
    # extracting data from Dict to List
    fig_TrnVal_PartB=plt.figure(50)
    for i,(xx,yy) in enumerate(Var.trnErr_B.items()):
        trnErrList_B.append([xx,yy])
        
    for i,(xx,yy) in enumerate(Var.valErr_B.items()):
        valErrList_B.append([xx,yy])
        
    # we need to sort because they are added to the list as soon as they finish
    # using sort() + lambda to sort list of list (sort by second index) 
    trnErrList_B.sort(key=lambda trnErrList_B:trnErrList_B[0]) 
    valErrList_B.sort(key=lambda valErrList_B:valErrList_B[0]) 
    
    Var.trnErrArray_B=np.array(trnErrList_B)
    Var.valErrArray_B=np.array(valErrList_B)

    #plotting train validation plot
    plt.plot(np.log10(Var.trnErrArray_B[:,0]),Var.trnErrArray_B[:,1],'*',color='black')
    plt.plot(np.log10(Var.valErrArray_B[:,0]),Var.valErrArray_B[:,1],'*',color='black') 
    
    plt.plot(np.log10(Var.trnErrArray_B[:,0]),Var.trnErrArray_B[:,1],'-',color='blue', label='train') 
    plt.plot(np.log10(Var.valErrArray_B[:,0]),Var.valErrArray_B[:,1],'-',color='red', label='validation')
    
    plt.legend(loc='lower right',frameon=False)
    plt.xlabel('log(Gamma values)') 
    plt.ylabel('Accuracy')     

    
#==============================================================================
# this function prinrtsparsed data
#==============================================================================
def PrintParsedData(): 
    print('') 
    print('  ========================Parsed  Data=====================')  
    print('  ', Var.args)
    print('')
    print('  Script Path                =', Var.args.ScriptPath)
    print('  .csv Train file address    =', Var.args.TrainDataPath)
    print('  .csv Test file address     =', Var.args.TestDataPath)
    print('  Work Directory             =', Var.args.WorkDir)
    # results path is ceated ans set in CreateAndSetResultsDir()
    print('  Results Path               =', os.getcwd())
    print('  Save Plots                 =', Var.args.SavePlot)
    print('  Create Log File            =', Var.args.logFile)
    print('  Create Report File         =', Var.args.ReportFile)
    print('  =========================================================')
    print('')



#==============================================================================
# this function reads and splits data
#==============================================================================        
def Read_Scale_Data(trnPath,tstPath):        
    # read data
    Var.df_trn=pd.read_csv(trnPath)
    Var.df_tst=pd.read_csv(tstPath)
    print(Var.df_trn.head())
    # drop features
    Var.df_trn_drp = Var.df_trn.drop(columns=['Artist','Track','Year','Label'])
    Var.y_trn=Var.df_trn['Label']

    # StandardScaler
    scaler = StandardScaler()
    Var.X_trn_scaled=scaler.fit_transform(Var.df_trn_drp)
    Var.n_Features=len(Var.df_trn_drp.columns)
    
    # preparing test data for prediction
    Var.df_tst_dropped = Var.df_tst.drop(columns=['Valence','Acousticness'])
    Var.tst_dropped=Var.df_tst_dropped.values
    
    scaler = StandardScaler()
    Var.tst_data_dropped_scaled=scaler.fit_transform(Var.tst_dropped)
    

#==============================================================================
# this function saves results of predcrions of test data into csv file
#==============================================================================        
def SavePredResults_CVSFile():
    Rasults=np.hstack((Var.RF.predicted_data,Var.KNN.predicted_data,
                   Var.SVM.predicted_data, Var.LR.predicted_data))
    df_results=pd.DataFrame(Rasults,columns=Var.Clf_Names)
    df_results.to_csv('Results.csv',index=False)


#==============================================================================
# this function sets flags for log and save plot
#==============================================================================        
def SetFlags():        

    Var.saveplotFlag=False if Var.args.SavePlot=='0' else True
    Var.logFlag=False if Var.args.logFile=='0' else True
    Var.docxFlag=False if Var.args.ReportFile=='0' else True
    
    if Var.docxFlag==True: Var.logFlag=True


#==============================================================================
# this function splits data
#==============================================================================        
def Split(): 
    Var.X_trn,Var.X_tst,Var.y_trn,Var.y_tst=train_test_split(
            Var.X_trn_scaled,Var.y_trn,
              test_size=Var.tFrac,random_state=Var.random_state)
        
    
#==============================================================================
# this function does the calculation of prediction and prepare plot data Part A
#==============================================================================
def visualize_PartA(p,clf,param,x,y,index):
    
    # train the model
    clf_trained=clf.fit(x,y)
    
    # we save the trained models to be used later in cross val
    Var.SVCTrainedModelsDict_C[p]=clf_trained
    
    # define which plot to be used for this svc model
    r,c=np.divmod(index,3)
    ax=axes[r,c]
    print("index,r,c",index,r,c)

    # Plot contours
    zMesh=clf.decision_function(np.c_[xMesh.ravel(),yMesh.ravel()])
    zMesh=zMesh.reshape(xMesh.shape)
    ax.contourf(xMesh,yMesh,zMesh,cmap=plt.cm.PiYG,alpha=0.6)
    
    ax.contour(xMesh,yMesh,zMesh,colors='k',levels=[-1,0,1],
               alpha=0.5,linestyles=['--','-','--'])

    # Plot data
    ax.scatter(x[:,0],x[:,1],c=y,cmap=cmap,edgecolors='k')
    ax.set_title('{0}={1}'.format(param, p))
    
    
#==============================================================================
# this function does the calculation of prediction and prepare plot data Part B
#==============================================================================
def visualize_PartB(p,clf,param,x,y,index):
    
    # train the model
    clf_trained=clf.fit(x,y)
    
    # we save the trained models to be used later in cross val
    Var.SVCTrainedModelsDict_Gamma[p]=clf_trained
    
    # define which plot to be used for this svc model
    r,c=np.divmod(index,3)
    ax=axes[r,c]
    print("index,r,c",index,r,c)

    # Plot contours
    zMesh=clf.decision_function(np.c_[xMesh.ravel(),yMesh.ravel()])
    zMesh=zMesh.reshape(xMesh.shape)
    ax.contourf(xMesh,yMesh,zMesh,cmap=plt.cm.PiYG,alpha=0.6)
    
    ax.contour(xMesh,yMesh,zMesh,colors='k',levels=[-1,0,1],
               alpha=0.5,linestyles=['--','-','--'])

    # Plot data
    ax.scatter(x[:,0],x[:,1],c=y,cmap=cmap,edgecolors='k')
    ax.set_title('{0}={1}'.format(param, p))
    











