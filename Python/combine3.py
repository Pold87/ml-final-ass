import pandas as pd
import numpy as np

sub1 =  'submissions/sub_xgb_1.csv'
s1 = pd.read_csv(sub1)
s_all = s1.Prediction

for i in range(2, 26):
    sub_tmp =  'submissions/sub_xgb_%d.csv' % i
    s_tmp = pd.read_csv(sub_tmp)
    s_all += s_tmp.Prediction

#for i in range(1, 26):
#    sub_tmp =  'submissions/sub_mult_25_%d.csv' % i
#    s_tmp = pd.read_csv(sub_tmp)
#    s_all += s_tmp.Prediction

print s_all
    
s_sub_again = np.array(s_all >= 13.0)
s_sub_again = s_sub_again.astype(int)

print np.sum(s_sub_again)

submission = pd.DataFrame({'Id': s1.Id, 'Prediction': s_sub_again})
submission.to_csv('submissions/sub_xgb_todo.csv', index=False)
