import pandas as pd
import numpy as np

#sub1 =  '/home/pold/Downloads/3_subs/sub_multi_combi.csv'
sub1 = '/home/pold/Downloads/sub_multi_3_subs.csv'
#sub2 =  '/home/pold/Downloads/3_subs/sub_multi_self.csv'
sub2 = 'submissions/sub_50_todo.csv'
#sub3 =  '/home/pold/Downloads/3_subs/sub_multi_svm.csv'
sub3 = 'submissions/sub_xgb_todo.csv'

s1 = pd.read_csv(sub1)
s2 = pd.read_csv(sub2)
s3 = pd.read_csv(sub3)

s_all = s1.Prediction + s2.Prediction + s3.Prediction

s_sub_again = np.array(s_all > 1.5)
s_sub_again = s_sub_again.astype(int)

submission = pd.DataFrame({'Id': s1.Id, 'Prediction': s_sub_again})
submission.to_csv('submissions/final_sub.csv', index=False)
