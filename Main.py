
"""
===============================================================================
 Created on Feb 15, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""


#==============================================================================
# deleting variables before starting main code
#==============================================================================
try:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
    print('variables deleted')
except:
    print('couldn" delete variables on ipython')
        

#==============================================================================
# importing module codes
#==============================================================================
import ModClass as Clss
import ModVar as Var
import ModFunc as Func
import ModRepDocx as RepDocx


#==============================================================================
# importing standard classes
#==============================================================================
from sklearn.svm import SVC
import time
import os
import threading
import argparse


#==============================================================================
# main code starts here
#==============================================================================
if __name__=='__main__':
    
    # script path
    Var.MainDir=os.path.abspath(__file__)
    Var.MainDir=Var.MainDir[0:len(Var.MainDir)-len('Main.py')]
    print('MainDir  is:  ', Var.MainDir)
    
    # call input data from user by means of parsing
    Func.Call_Parser()
    
    # starting timer of total run
    TotRun_Start=time.perf_counter()
    
    # setting flags for log file, save plots and report file
    Func.SetFlags()
    
    # create and set results directory
    Func.CreateAndSetResultsDir()
    
    # print parsed data            
    Func.PrintParsedData()
    
    # create log object
    if Var.logFlag: 
        My_Log=Clss.LogClass()
        # log the data of previous lines
        My_Log.ProgStart('LM')
        My_Log.ParsedData('M')
    
    # logging system information
    Func.GetSysInfo()
    if Var.logFlag: My_Log.SysSpec(Var.sysinfo,'M')
    
    # problem input data
    Func.ClassifierParameters()
    
    # logging problem definition
    if Var.logFlag: My_Log.ProbDef('M')
    
    # read datasets
    Func.Read_Scale_Data(Var.args.TrainDataPath, Var.args.TestDataPath)
    
    # feature importance by ExtraTreesClassifier
    Func.Feature_importance_ExtraTrees()
    
    # feature importance by PCA
    Func.Feature_importance_PCA()
    
    # feature reduction by PCA
    Func.Feature_reduction_PCA()
    
    # split data
    Func.Split()
    
    # log main calculations for each classifier
    if Var.logFlag: My_Log.Analysis_Execution_Start_Header('M')
    
    # this loop does the calculations of each classifier
    for i , N in enumerate(Var.Clf_Names):
        
        t="t1=time.perf_counter()"
        exec(t)
        if Var.logFlag: My_Log.Curr_Classifier_Calc_Started('M',i)
        
        t="Var."+ N +"=Clss.MLClassifier_Class()"
        exec(t)
        
        t="Var."+ N +".Classifier_Run('"+ N +"')"
        eval(t)
        
        t="Var."+ N +".AccScor()"
        eval(t)
        if Var.logFlag: My_Log.Curr_Classifier_AccScor_Done('M',i)
        
        t="Var."+ N +".PltRocTrn(Var." + N +"_VarPar)"
        eval(t)
        if Var.logFlag: My_Log.Curr_Classifier_PltRocTrn_Done('M',i)
        
        t="Var."+ N +".PltRocTst(Var." + N +"_VarPar)"
        eval(t)
        if Var.logFlag: My_Log.Curr_Classifier_PltRocTst_Done('M',i)
        
        t="Var."+ N +".PltCrsVal(Var." + N +"_VarPar)"
        eval(t)
        if Var.logFlag: My_Log.Curr_Classifier_PltCrsVal_Done('M',i)
        
        t="Var."+ N +".PltAccTst(Var." + N +"_VarPar)"
        eval(t)
        if Var.logFlag: My_Log.Curr_Classifier_PltAccTst_Done('M',i)
        
        t="Var."+ N +".MainPred(Var.OptVals['" + N +"'])"
        eval(t)

        t="Var."+ N +".Save_plots()"
        eval(t)
        if Var.logFlag: My_Log.Curr_Classifier_Plots_Saved('M',i)
        
        t="dt=time.perf_counter()-t1"
        exec(t)
        
        t="Var.Time_List.append(round(dt*100)/100)"
        eval(t)
        if Var.logFlag: My_Log.Curr_Classifier_Calc_Ended('M',i)
    

    # save results of predcrions of test data into csv file
    Func.SavePredResults_CVSFile()
    
    dt=time.perf_counter()-TotRun_Start
    Var.Time_List.append(round(dt*100)/100)
    
    print(Var.Time_List)
    

    # check docx package
    Import_Err=0
    try:
        import docx
        from docx import Document
        from docx.shared import Inches
        from docx.shared import Pt
        from docx.shared import RGBColor
        from docx.enum.style import WD_STYLE_TYPE
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.oxml import ns
        from docx.shared import Cm, Mm
        if Var.logFlag: My_Log.Docx_Report_Error('M',1)
    except:
        print('Error importing docx or related components')
        Import_Err=1
        if Var.logFlag: My_Log.Docx_Report_Error('M',0)
        
        
    # create report file
    if Import_Err==0:
        # RepDocx.Rep()
        if Var.docxFlag:
            try:
                RepDocx.Rep()
            except:
                print("Report file could't be created for some reason")

#------------------------------------------------------------------------------








