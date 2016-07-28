import pandas as pd
import numpy as np

sub1 =  'submissions/sub_mult_1.csv'
sub2 =  'submissions/sub_mult_2.csv'
sub3 =  'submissions/sub_mult_3.csv'
sub4 =  'submissions/sub_mult_4.csv'
sub5 =  'submissions/sub_mult_5.csv'

s1 = pd.read_csv(sub1)
s2 = pd.read_csv(sub2)
s3 = pd.read_csv(sub3)
s4 = pd.read_csv(sub4)
s5 = pd.read_csv(sub5)      


s_all = s1.Prediction + s2.Prediction + s3.Prediction + s4.Prediction + s5.Prediction

s_sub_again = np.array(s_all > 2.5)
s_sub_again = s_sub_again.astype(int)

submission = pd.DataFrame({'Id': s1.Id, 'Prediction': s_sub_again})
submission.to_csv('submissions/sub_multi_svm.csv', index=False)
