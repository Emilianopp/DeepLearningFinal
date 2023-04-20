from tqdm import tqdm
from IPython.display import display
import pandas as pd 
import matplotlib.pyplot as plt
import ast
import pandas as pd
import argparse 
parser = argparse.ArgumentParser('Interface for the link prediction task')
parser.add_argument('--dataset_dir', type=str,default = './data/')
args = parser.parse_args()

out = pd.read_csv(f'{args.data_dir}/edge_citation.csv',index_col = [0])
mapper = {0:'Artificial Intellegence', 1 :'Data Mining' , 2 :'Data Engineering/Management'}
out.src_label = out.src_label.apply(mapper.get)
out.dest_label = out.dest_label.apply(mapper.get)
print(out.groupby(['year','src_label']).count().source.to_latex())
grouped = out.groupby(['src_label','year','dest_label']).count()
grouped
grouped.loc['Artificial Intellegence'].plot('source')
grouped.plot()
grouped.loc['Artificial Intellegence']
grouped.loc['Artificial Intellegence',]
label_map = {x: i for i,x in enumerate(out.src_label.unique())}
out.src_label = out.src_label.apply(label_map.get)
out.dest_label = out.dest_label.apply(label_map.get)
out.groupby('src_label').count().sort_values('target',ascending=False)
out.to_csv(f'{args.data_dir}/edge_citation.csv')
out.year.unique()

df = pd.read_csv(f'{args.data_dir}/edge_citation.csv')
df.src_label = df.src_label
df.dest_label = df.dest_label
dst_df =df[['target','year','dest_label']]
dst_df = dst_df.rename(columns = {'target':'source' , 'dest_label' : 'src_label'})
src_df = df[['source','year','src_label']]

df = pd.concat([dst_df,src_df],axis=0)

label_df = []
timestamps =sorted(df.year.unique())[-7:-1]
for ts in tqdm(timestamps):
    exploded = df[df.year == ts].explode('src_label')
    exploded = exploded.drop(exploded[ exploded['src_label']== ''].index)
    label_counts = exploded['src_label'].value_counts()


    label_counts = exploded.groupby('source')['src_label'].value_counts()



    user_to_label = {}
    for user in label_counts.index.levels[0]:
        user_to_label[user] = label_counts[user].idxmax()
    data = []
    for user, label in user_to_label.items():
        data.append({'user': user, 'label': label,'year':ts})
    df_new = pd.DataFrame(data)
    label_df.append(df_new)
    print(f'{len(df_new)=} at year {ts}')
    

label_stack = pd.concat(label_df,axis = 0 )

import pandas as pd 
df = pd.read_csv(f'{args.data_dir}/edge_citation.csv')

subset_df_list = []
label_stack_adjusted = []

for ts in label_stack.year.unique():
    target_ids = set(label_stack[label_stack.year == ts].user) 

    cur_df = df[(df.year == ts) &(df.source.isin(target_ids)) & (df.target.isin(target_ids))]
    cur_df_set = set(cur_df.source) | set(cur_df.target)
  
    subset_df_list.append(df[(df.year == ts) &(df.source.isin(target_ids)) & (df.target.isin(target_ids))])
    label_stack_adjusted.append(label_stack[(label_stack.year == ts) & (label_stack.user.isin(cur_df_set))])

sub_df = pd.concat(subset_df_list)
label_stack= pd.concat(label_stack_adjusted)
node_set = set(label_stack.user)
node_map = {x:i for i,x in enumerate(node_set)}
label_stack_mapped = label_stack.copy()
label_stack_mapped.user = label_stack.user.apply(node_map.get)
label_stack_mapped.to_csv(f'{args.dataset_dir}/labels.csv',index = False)

