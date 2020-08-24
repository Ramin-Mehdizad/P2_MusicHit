

"""
===============================================================================
 Created on Feb 15, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""


#==============================================================================
# This module contains all the Classes that are used in the main code
#==============================================================================


#==============================================================================
# importing standard classes
#==============================================================================
import logging
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


#==============================================================================
# importing module codes
#==============================================================================
import ModVar as Var


#==============================================================================
# this class creates machin learning classifier and controls it
#==============================================================================
class MLClassifier_Class:
  
    
    # initializing class instance parameters
    def __init__(self):
        self.X_trn=Var.X_trn
        self.X_tst=Var.X_tst
        self.y_trn=Var.y_trn
        self.y_tst=Var.y_tst
    
    
    # initialize lists
    def initialize(self):
        self.fpr_list_trn = list()
        self.tpr_list_trn = list()
        self.AUC_list_trn = list()
        
        self.fpr_list_tst = list()
        self.tpr_list_tst = list()
        self.AUC_list_tst = list()

        self.accuracy_score_testData = list()
        
        self.Model_List=dict()


    # select classifier     
    def Classifier_Run(self,MLC):
        self.initialize()
        self.MLCtype=MLC
        if self.MLCtype=='RF':
            # self.initialize()
            for i,N in enumerate(Var.RF_VarPar):
                self.clf = RandomForestClassifier(
                    criterion=Var.RF_criterion,
                     n_estimators=N ,random_state=Var.random_state)
                self.Model_List[N]=self.clf
                # do the predictions and post processing
                self.Calc()
                
        elif self.MLCtype=='KNN': 
            # self.initialize()
            for i,N in enumerate(Var.KNN_VarPar):
                self.clf = KNeighborsClassifier(
                    n_neighbors=N, metric=Var.KNN_metric,
                    p=Var.KNN_p)
                self.Model_List[N]=self.clf
                # do the predictions and post processing
                self.Calc()
                
        elif self.MLCtype=='SVM': 
            # self.initialize()
            for i,G in enumerate(Var.SVM_VarPar):
                self.clf = SVC(kernel=Var.SVM_kernel,
                               probability=True,C=Var.SVM_C,
                                random_state=Var.random_state,
                                gamma=G)
                self.Model_List[G]=self.clf
                # do the predictions and post processing
                self.Calc()
                
        elif self.MLCtype=='LR': 
            # self.initialize()
            for i,C in enumerate(Var.LR_VarPar):
                self.clf = LogisticRegression(C=C,solver=Var.LR_solver,
                            max_iter=Var.LR_max_iter)
                self.Model_List[C]=self.clf
                # do the predictions and post processing
                self.Calc()        
    
    
    # this function trains classifier and calculates fpr, tpr and AUC
    def Calc(self):
        self.clf.fit(self.X_trn, self.y_trn)
        self.y_pred_trn = self.clf.predict_proba(self.X_trn)
        self.y_pred_tst = self.clf.predict_proba(self.X_tst)
        
        fpr_trn, tpr_trn, thresholds_trn = roc_curve( 
            self.y_trn, self.y_pred_trn[:,1])   
        roc_auc = auc(fpr_trn, tpr_trn)
        print(roc_auc)
        
        self.fpr_list_trn.append(fpr_trn)
        self.tpr_list_trn.append(tpr_trn)
        self.AUC_list_trn.append(roc_auc)
        
        fpr_tst, tpr_tst, thresholds_tst = roc_curve( 
                        self.y_tst, self.y_pred_tst[:,1])   
        
        roc_auc = auc(fpr_tst, tpr_tst)
        self.fpr_list_tst.append(fpr_tst)
        self.tpr_list_tst.append(tpr_tst)
        self.AUC_list_tst.append(roc_auc)
        
        y_predicted=self.clf.predict(self.X_tst)
        self.accuracy_score_testData.append(
            accuracy_score(self.y_tst, y_predicted, normalize=True))
    
    
    # in this method, we calculate accuracy score     
    def AccScor(self):
        if self.MLCtype=='RF':
            clf=RandomForestClassifier()
            parname="n_estimators"
            parrange=Var.RF_VarPar
        elif self.MLCtype=='KNN':
            clf=KNeighborsClassifier()
            parname="n_neighbors"
            parrange=Var.KNN_VarPar
        elif self.MLCtype=='SVM':
            clf=SVC()
            parname="gamma"
            parrange=Var.SVM_VarPar
        elif self.MLCtype=='LR':
            clf=LogisticRegression()
            parname="C"
            parrange=Var.LR_VarPar
            
        train_scores, test_scores = validation_curve(clf, 
                    self.X_trn, self.y_trn, param_name=parname, 
                    param_range=parrange, scoring="accuracy",
                    cv=Var.cvNum)
        
        self.train_scores_mean = np.mean(train_scores, axis=1)
        self.test_scores_mean = np.mean(test_scores, axis=1)
    
    
    # this function plots ROC curves for train data
    def PltRocTrn(self,variable_list):
        # Plot of a ROC curve for train data
        for i in range(len(variable_list)):
            if i==0:
                self.fig_ROC_trn=plt.figure()
                fig=plt.gca()
                plt.plot([0, 1], [0, 1], color='navy', linestyle='--',lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                ttl='(' + self.MLCtype + ') ROC of Train Data'
                plt.title(ttl)
                
            plt.plot(self.fpr_list_trn[i], self.tpr_list_trn[i],
                     color=Var.Color_List[i], lw=1, 
                     label='n={}, (AUC ={:.2f} )'.format(variable_list[i],
                              self.AUC_list_trn[i] ))
            lgd1 =plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0),
            ncol=1, fancybox=True, shadow=True)
        plt.show()
    
    
    # this function plots ROC curves for test data
    def PltRocTst(self,variable_list):
        # Plot of a ROC curve for test data
        for i in range(len(variable_list)):
            if i==0:
                self.fig_ROC_tst=plt.figure()
                fig=plt.gca()
                plt.plot([0, 1], [0, 1], color='navy', linestyle='--',lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                ttl='(' + self.MLCtype + ') ROC of Test Data'
                plt.title(ttl)
            
            plt.plot(self.fpr_list_tst[i], self.tpr_list_tst[i],
                     color=Var.Color_List[i], lw=1, 
                     label='n={}, (AUC ={:.2f} )'.format(variable_list[i],
                              self.AUC_list_tst[i] ))
            lgd1 =plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0),
            ncol=1, fancybox=True, shadow=True)
        plt.show()
        
    
    # this function plots cross validation 
    def PltCrsVal(self,variable_list):
        # Plot Accuracy train and val
        self.fig_CrsVal=plt.figure()
        fig=plt.gca()
        plt.xlabel('n')
        plt.ylabel('Accuracy')
        plt.title('(' + self.MLCtype + ') Accuracy')
        if self.MLCtype in ['RF','KNN']:
            plt.plot(variable_list, self.train_scores_mean, '-',
                     color='darkorange', lw=1, label='Train score')
            plt.plot(variable_list, self.test_scores_mean,
                     label="Test score", color="navy", lw=1)
            plt.legend(loc="best")
            plt.show()
        else:
            plt.plot(np.log10(variable_list), self.train_scores_mean, '-',
                     color='darkorange', lw=1, label='Train score')
            plt.plot(np.log10(variable_list), self.test_scores_mean,
                     label="Test score", color="navy", lw=1)
            plt.legend(loc="best")
            plt.show()


    # this function plots accuracy of test data 
    def PltAccTst(self,variable_list):
        # Plot Accuracy of test data
        # print('accuracy_score_testData',accuracy_score_testData)
        self.fig_ACC_tst=plt.figure()
        fig=plt.gca()
        plt.xlabel('n')
        plt.ylabel('Accuracy')
        plt.title('(' + self.MLCtype + ') Accuracy of test data ')
        if self.MLCtype in ['RF','KNN']:
            plt.plot(variable_list, self.accuracy_score_testData, 
                     '-', color='b', lw=2)
            plt.show()
        else:
            plt.plot(np.log10(variable_list), self.accuracy_score_testData, 
                     '-', color='b', lw=2)
            plt.show()


    def MainPred(self,N):
        # main prediction
        clf = self.Model_List[N]
        clf.fit(self.X_trn, self.y_trn)
        self.predicted_data=np.array((clf.predict(
            Var.tst_data_dropped_scaled))).reshape(-1,1)

    
    # this function saves plots
    def Save_plots(self):
        name=self.MLCtype +'_1.jpg'
        self.fig_ROC_trn.savefig(name, dpi=Var.figdpi,
                                facecolor='w',bbox_inches='tight')
        name=self.MLCtype +'_2.jpg'              
        self.fig_ROC_tst.savefig(name, dpi= Var.figdpi,
                            facecolor='w',bbox_inches='tight')
        name=self.MLCtype +'_3.jpg'
        self.fig_CrsVal.savefig(name, dpi= Var.figdpi,
                           facecolor='w',bbox_inches='tight') 
        name=self.MLCtype +'_4.jpg'
        self.fig_ACC_tst.savefig(name, dpi= Var.figdpi,
                           facecolor='w',bbox_inches='tight') 
        
        
        
#==============================================================================
# defining class logging events and results into *.log file
#        
# Note:
#     All the methods and logging data are dreatedin the methods of this class.
#     Then the logging action is done in the main code
#==============================================================================
class LogClass():
   
    # initializing class instance parameters
    def __init__(self):
        self.logger=logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.formatter=logging.Formatter('%(message)s')
        self.filehandler=logging.FileHandler(
                            Var.resultssubpath+'\\LogFile.log')
        self.filehandler.setFormatter(self.formatter)
        self.logger.addHandler(self.filehandler)
        self.splitterLen=84
        self.splitterChar='*'
        self.EndSep=self.splitterChar*self.splitterLen
    
    
    # this method logs joining of all visualise processes
    def Analysis_Execution_Start_Header(self,n):    
        title=' MONITORING MAIN CODE RUN '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('')
    
    
    # this method logs joining of all visualise processes
    def AllVisJoined(self,n):    
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('All visual threads joined') 
    
    
    # this method logs joining of accuracy processes   
    def AllAccJoined(self,n):    
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('All accuracy threads joined')
        
        
    # this method logs saving created data into .csv file
    def DataSavdToCSV(self,n):  
        self.LogFrmt(n)
        self.logger.info("Raw data saved into csv file")
    
    
    # this method logs saving of results and figures    
    def FigResSaved(self,n):
        self.LogFrmt(n)
        self.logger.info('Figures and results successfully saved.')
        self.logger.info(self.EndSep)


    # this method performs the format of logging for each log action    
    def LogFrmt(self,n):
        if n=='M':
            self.formatter=logging.Formatter(' %(message)s')
            self.filehandler.setFormatter(self.formatter)
        elif n=='LM':
            self.formatter=logging.Formatter('%(levelname)s: %(message)s')
            self.filehandler.setFormatter(self.formatter)
        elif n=='TLM':
            self.formatter=logging.Formatter('%(acstime)s: %(levelname)s: %(message)s')
            self.filehandler.setFormatter(self.formatter)


    # this method logs ParsedData
    def ParsedData(self,n):
        self.LogFrmt(n)
        title=' Data Entered by User '
        sp=self.splitterChar*(round((self.splitterLen-len(title))/2))
        self.logger.info(sp+title+sp)
        self.logger.info('')
        self.logger.info('  Script Path                =', Var.args.ScriptPath)
        self.logger.info('  .csv Train file address    =', Var.args.TrainDataPath)
        self.logger.info('  .csv Test file address     =', Var.args.TestDataPath)
        self.logger.info('  Work Directory             =', Var.args.WorkDir)
        # results path is ceated ans set in CreateAndSetResultsDir()
        self.logger.info('  Results Path               =', os.getcwd())
        self.logger.info('  Save Plots                 =', Var.args.SavePlot)
        self.logger.info('  Create Log File            =', Var.args.logFile)
        self.logger.info('  Create Report File         =', Var.args.ReportFile)
        self.logger.info(self.EndSep)
    
    
    # this method logs start of the main
    def ProgStart(self,n):
        self.LogFrmt(n)
        self.logger.info('')
        title=' Main Program Started '
        sp=self.splitterChar*(round((self.splitterLen-len(title))/2))
        self.logger.info(sp+title+sp)
        self.logger.info('')
        self.logger.info('')
     
        
    # this method logs     
    def ProbDef(self,n):
        
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('')
        title=' Problem Definition '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('The problem has 4 parts:\n')  
        self.logger.info('    Part A: Random Forest various n_estimators')
        self.logger.info('    Part B: KNN with various n_neighbors')
        self.logger.info('    Part C: SVM with various Gamma values')
        self.logger.info('    Part D: Logistic Regression with various C values')
        
        # public settings
        self.logger.info('')
        self.logger.info('Public settings:')
        self.logger.info('')
        self.logger.info('Train\Val percentage: {} '.format(Var.tFrac))
        self.logger.info('Cross Val KFold: {} '.format(Var.cvNum))
        self.logger.info('Figure dpi: {} '.format(Var.figdpi))
        
        # random forest settings
        self.logger.info('')
        self.logger.info('random forest settings:')
        self.logger.info('Number of estimators: {} '.format(Var.RF_VarPar))
        self.logger.info('Criterion: {} '.format(Var.RF_criterion))
        
        # KNN settings
        self.logger.info('')
        self.logger.info('KNN settings:')
        self.logger.info('Number of neighbors: {} '.format(Var.KNN_VarPar))
        self.logger.info('Distance metric: {} '.format(Var.KNN_metric))
        self.logger.info('metric p: {} '.format(Var.KNN_p))

        # SVM settings
        self.logger.info('')
        self.logger.info('SVM settings:')
        self.logger.info('kerbel: {} '.format(Var.SVM_kernel))
        self.logger.info('C: {} '.format(Var.SVM_C))
        self.logger.info('log (Gamma): {} '.format(Var.SVM_logVarPar))

        # Logestic regression settings
        self.logger.info('')
        self.logger.info('Log Reg settings:')
        self.logger.info('log(C): {} '.format(Var.LR_logVarPar))
        self.logger.info('solver: {} '.format(Var.LR_solver))
        self.logger.info('max_iter: {} '.format(Var.LR_VarPar))
        
        self.logger.info(self.EndSep)
        self.logger.info('')
        self.logger.info('')


    # this method logs the system on which the analysis if performed  
    def SysSpec(self,sysinfo,n):
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('')
        title=' COMPUTER SPEC '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('Data analsys is done on the system with following spec:\n')  
        for i,[a1,a2] in enumerate(Var.sysinfo):
            DataStartChar=30
            len1=len(Var.sysinfo[i][0])
            Arrow='-'*(DataStartChar-len1)+'> '
            self.logger.info(Var.sysinfo[i][0]+Arrow+Var.sysinfo[i][1])
        self.logger.info(self.EndSep)


    # this method logs plotting the input data set    
    def XYPlotted(self,n): 
        self.LogFrmt(n)
        self.logger.info("Raw data plotted successfully")
        self.logger.info('')
        
        
        
    # ------------the following messages logged in claculation loop------------    
    # this method logs start of calculations for each classifier    
    def Curr_Classifier_Calc_Started(self,n,i): 
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info("Calculations of classifier {0} started".format(i))
        
    # this method logs finishing of accuracy score of each classifier    
    def Curr_Classifier_AccScor_Done(self,n,i): 
        self.LogFrmt(n)
        self.logger.info("Accuracy Score calculations of classifier {0} done".format(i))
        
    # this method logs finishing of roc plot of train data of each classifier    
    def Curr_Classifier_PltRocTrn_Done(self,n,i): 
        self.LogFrmt(n)
        self.logger.info("ROC Plot of train data of classifier {0} done".format(i))
        
    # this method logs finishing of roc plot of test data of each classifier    
    def Curr_Classifier_PltRocTst_Done(self,n,i): 
        self.LogFrmt(n)
        self.logger.info("ROC Plot of test data of classifier {0} done".format(i))
        
    # this method logs finishing of cross val plot of each classifier    
    def Curr_Classifier_PltCrsVal_Done(self,n,i): 
        self.LogFrmt(n)
        self.logger.info("Cross Val Plot of classifier {0} done".format(i))
        
    # this method logs finishing of accuracy plot of test data of each classifier    
    def Curr_Classifier_PltAccTst_Done(self,n,i): 
        self.LogFrmt(n)
        self.logger.info("Accuracy plot of Test data of classifier {0} done".format(i))
        
    # this method logs saving plots of each classifier    
    def Curr_Classifier_Plots_Saved(self,n,i): 
        self.LogFrmt(n)
        self.logger.info("Plots of classifier {0} done".format(i))
        
    # this method logs finishing of calculations for each classifier    
    def Curr_Classifier_Calc_Ended(self,n,i): 
        self.LogFrmt(n)
        self.logger.info("Calculations of classifier {0} finished".format(i))
        
    # Auto Docx report file creation message    
    def Docx_Report_Error(self,n,msgID): 
        self.LogFrmt(n)
        self.logger.info('')
        if msgID==1:
            self.logger.info("Auto Docx Report file created successfully")
        else:
            self.logger.info("Couldn't create Auto Docx Report File")
    #--------------------------------------------------------------------------   
    

















