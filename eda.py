import pandas as pd
from sklearn.model_selection import train_test_split
import pprint

import matplotlib.pyplot as plt

# data = pd.read_csv('data/survey_results_public.csv')

# training_data, testing_data = train_test_split(data, test_size=0.2)

# training_data.to_csv('data/survey_results_public_training.csv')
# testing_data.to_csv('data/survey_results_public_testing.csv')

data = pd.read_csv('data/survey_results_public_training.csv')
print(data.info(verbose=True))
# print(data.head())

# Developers who use artificial intelligence as part of their development process make more money
# More experienced developers use less artificial intelligence
# Do programmers who code outside of work make more money?
# Which country’s programmers code most outside of work?
# Which country’s programmers use AI the most?

column = 'CodingActivities'
print(data[column].dropna())
# print(data[data[column] == 'Less than 1 year'])
# 44 where YearsCodePro == 'More than 50 years' >> set this to 51???
# 2279 where YearsCodePro == 'Less than 1 year' >> prob just set this to 0

# Step 1. Find the columns I care about (which ones will help me answer my questions)
# 

# For the number of years professionally coding, instead of doing a month measurement, they measure years and put everything else as "Less than 1 year". Is there a good way to "turn this text into a number" so I can use this column as a numeric variable?

# Data columns (total 115 columns):
#  #    Column                          Dtype  
# ---   ------                          -----  
#  0    Unnamed: 0                      int64  
#  1    ResponseId                      int64  
#  2    MainBranch                      object 
#  3    Age                             object >> Categorical
#  4    Employment                      object >> Categorical
#  5    RemoteWork                      object 
#  6    Check                           object >> Drop this column
#  7    CodingActivities                object >> Multivalued List
#  8    EdLevel                         object 
#  9    LearnCode                       object 
#  10   LearnCodeOnline                 object 
#  11   TechDoc                         object 
#  12   YearsCode                       object 
#  13   YearsCodePro                    object >> CouldBeNumeric
#  14   DevType                         object >> Categorical !!!
#  15   OrgSize                         object 
#  16   PurchaseInfluence               object 
#  17   BuyNewTool                      object 
#  18   BuildvsBuy                      object 
#  19   TechEndorse                     object 
#  20   Country                         object >> Categorical !!!
#  21   Currency                        object 
#  22   CompTotal                       float64
#  23   LanguageHaveWorkedWith          object >> Multivalued List
#  24   LanguageWantToWorkWith          object >> Multivalued List
#  25   LanguageAdmired                 object >> Multivalued List
#  26   DatabaseHaveWorkedWith          object >> Multivalued List
#  27   DatabaseWantToWorkWith          object >> Multivalued List
#  28   DatabaseAdmired                 object >> Multivalued List
#  29   PlatformHaveWorkedWith          object >> Multivalued List
#  30   PlatformWantToWorkWith          object >> Multivalued List
#  31   PlatformAdmired                 object >> Multivalued List
#  32   WebframeHaveWorkedWith          object >> Multivalued List
#  33   WebframeWantToWorkWith          object >> Multivalued List
#  34   WebframeAdmired                 object >> Multivalued List
#  35   EmbeddedHaveWorkedWith          object >> Multivalued List
#  36   EmbeddedWantToWorkWith          object >> Multivalued List
#  37   EmbeddedAdmired                 object >> Multivalued List
#  38   MiscTechHaveWorkedWith          object >> Multivalued List
#  39   MiscTechWantToWorkWith          object >> Multivalued List
#  40   MiscTechAdmired                 object >> Multivalued List
#  41   ToolsTechHaveWorkedWith         object >> Multivalued List
#  42   ToolsTechWantToWorkWith         object >> Multivalued List
#  43   ToolsTechAdmired                object >> Multivalued List
#  44   NEWCollabToolsHaveWorkedWith    object >> Multivalued List
#  45   NEWCollabToolsWantToWorkWith    object >> Multivalued List
#  46   NEWCollabToolsAdmired           object >> Multivalued List
#  47   OpSysPersonal use               object >> Categorical
#  48   OpSysProfessional use           object >> Categorical
#  49   OfficeStackAsyncHaveWorkedWith  object >> Multivalued List
#  50   OfficeStackAsyncWantToWorkWith  object >> Multivalued List
#  51   OfficeStackAsyncAdmired         object >> Multivalued List
#  52   OfficeStackSyncHaveWorkedWith   object >> Multivalued List
#  53   OfficeStackSyncWantToWorkWith   object >> Multivalued List
#  54   OfficeStackSyncAdmired          object >> Multivalued List
#  55   AISearchDevHaveWorkedWith       object >> Multivalued List
#  56   AISearchDevWantToWorkWith       object >> Multivalued List
#  57   AISearchDevAdmired              object >> Multivalued List
#  58   NEWSOSites                      object >> Multivalued List
#  59   SOVisitFreq                     object 
#  60   SOAccount                       object 
#  61   SOPartFreq                      object 
#  62   SOHow                           object 
#  63   SOComm                          object 
#  64   AISelect                        object >> ['Yes', 'No, but I plan to soon', 'No, and I don't plan to']
#  65   AISent                          object >> ['Very Favorable', 'Favorable', 'Indifferent', 'Unsure', 'Unfavorable', 'Very Unfavorable']
#  66   AIBen                           object >> ['Increase productivity', 'Speed Up Learning', 'Greater Efficiency', 'Improve Accuracy in Coding', 'Make Workload More Manageable', 'Improve Collaboration']
#  67   AIAcc                           object >> ['Highly Trust', 'Somewhat Trust', 'Neither Trust nor Distrust', 'Somewhat Distrust', 'Highly Distrust']
#  68   AIComplex                       object >> ['Very well at handling complex tasks', 'Good, but not great at handling complex tasks', 'Neither good or bad at handling complex tasks', 'Bad at handling complex tasks', 'Very poor at handling complex tasks']
#  69   AIToolCurrently Using           object >> Categorical
#  70   AIToolInterested in Using       object >> Categorical
#  71   AIToolNot interested in Using   object 
#  72   AINextMuch more integrated      object >> Put in all the pre-defined answer choices
#  73   AINextNo change                 object >> Remove them from all five categories
#  74   AINextMore integrated           object >> Take the 'other' output
#  75   AINextLess integrated           object 
#  76   AINextMuch less integrated      object >> Turns out, I don't have the text useres entered into the 'other' field
#  77   AIThreat                        object 
#  78   AIEthics                        object 
#  79   AIChallenges                    object 
#  80   TBranch                         object 
#  81   ICorPM                          object 
#  82   WorkExp                         float64 >> Numeric!
#  83   Knowledge_1                     object 
#  84   Knowledge_2                     object 
#  85   Knowledge_3                     object 
#  86   Knowledge_4                     object 
#  87   Knowledge_5                     object 
#  88   Knowledge_6                     object 
#  89   Knowledge_7                     object 
#  90   Knowledge_8                     object 
#  91   Knowledge_9                     object 
#  92   Frequency_1                     object 
#  93   Frequency_2                     object 
#  94   Frequency_3                     object 
#  95   TimeSearching                   object 
#  96   TimeAnswering                   object 
#  97   Frustration                     object 
#  98   ProfessionalTech                object 
#  99   ProfessionalCloud               object 
#  100  ProfessionalQuestion            object 
#  101  Industry                        object 
#  102  JobSatPoints_1                  float64
#  103  JobSatPoints_4                  float64
#  104  JobSatPoints_5                  float64
#  105  JobSatPoints_6                  float64
#  106  JobSatPoints_7                  float64
#  107  JobSatPoints_8                  float64
#  108  JobSatPoints_9                  float64
#  109  JobSatPoints_10                 float64
#  110  JobSatPoints_11                 float64
#  111  SurveyLength                    object 
#  112  SurveyEase                      object 
#  113  ConvertedCompYearly             float64
#  114  JobSat                          float64

# Developers who use artificial intelligence as part of their development process make more money

# salary = data['ConvertedCompYearly']
ai_data = data[['ConvertedCompYearly', 'AISelect']].dropna()
use_ai = ai_data[ai_data['AISelect'] == 'Yes']
no_ai = ai_data[ai_data['AISelect'] != 'Yes']
# print(use_ai)
# print(no_ai)
# use_ai.plot()

# More experienced developers use less artificial intelligence

# Do programmers who code outside of work make more money?
# Which country’s programmers code most outside of work?
# Which country’s programmers use AI the most?