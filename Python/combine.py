import pandas as pd
import numpy as np

sub1 =  'submissions/sub_mult_cl_1.csv'
sub2 =  'submissions/sub_mult_cl_2.csv'
sub3 =  'submissions/sub_mult_cl_3.csv'
sub4 =  'submissions/sub_mult_cl_4.csv'
sub5 =  'submissions/sub_mult_cl_5.csv'
sub6 =  'submissions/sub_mult_self_1.csv'
sub7 =  'submissions/sub_mult_self_2.csv'
sub8 =  'submissions/sub_mult_self_3.csv'
sub9 =  'submissions/sub_mult_self_4.csv'
sub10 =  'submissions/sub_mult_self_5.csv'

s1 = pd.read_csv(sub1)
s2 = pd.read_csv(sub2)
s3 = pd.read_csv(sub3)
s4 = pd.read_csv(sub4)
s5 = pd.read_csv(sub5)      
s6 = pd.read_csv(sub6)
s7 = pd.read_csv(sub7)
s8 = pd.read_csv(sub8)
s9 = pd.read_csv(sub9)
s10 = pd.read_csv(sub10)      


s_all = s1.Prediction + s2.Prediction + s3.Prediction + s4.Prediction + s5.Prediction + s6.Prediction + s7.Prediction + s8.Prediction + s9.Prediction + s10.Prediction

s_sub_again = np.array(s_all > 1.0)
s_sub_again = s_sub_again.astype(int)

submission = pd.DataFrame({'Id': s1.Id, 'Prediction': s_sub_again})
submission.to_csv('submissions/sub_multi_combi.csv', index=False)
