import pandas as pd
import numpy as np

'''
TODO:
    1. fix "deployment size" column output (current NaN)
    2. begin plotting graphs
'''

def output(DF):
    # Output first 10 formatted rows to txt file
    num = DF.columns
    max_widths = {col: max(len(str(x)) for x in DF[col]) for col in num}
    max_widths = {k: max(len(k), v) + 2 for k, v in max_widths.items()}

    with open('formatted_output.txt', 'w') as f:
        header = "".join(f"{col:{max_widths[col]}}" for col in num)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for index, row in DF.head(10).iterrows():
            f.write("".join(f"{str(row[col]):{max_widths[col]}}" for col in num) + "\n")

def main():

    data_path = 'vmtable.csv'
    headers=['vmid','subscriptionid','deploymentid','vmcreated', 'vmdeleted', 'maxcpu', 'avgcpu', 'p95maxcpu', 'vmcategory', 'vmcorecount', 'vmmemory']
    trace_dataframe = pd.read_csv(data_path, header=None, index_col=False,names=headers,delimiter=',')

    deployment_data_path = 'deployments.txt'
    deployment_headers=['deploymentid','deploymentsize']
    deployment_trace_dataframe = pd.read_csv(deployment_data_path, header=None, index_col=False,names=deployment_headers,delimiter='\t')

    # Compute VM Lifetime based on VM Created and VM Deleted timestamps and transform to Hour
    trace_dataframe['lifetime'] = np.maximum((trace_dataframe['vmdeleted'] - trace_dataframe['vmcreated']),300)/ 3600
    trace_dataframe['corehour'] = trace_dataframe['lifetime'] * trace_dataframe['vmcorecount']

    # Truncating deploymentid, vmid, subscriptionid to first 5 characters
    trace_dataframe['deploymentid'] = trace_dataframe['deploymentid'].apply(lambda x: x[:10])
    trace_dataframe['vmid'] = trace_dataframe['vmid'].apply(lambda x: x[:5])
    trace_dataframe['subscriptionid'] = trace_dataframe['subscriptionid'].apply(lambda x: x[:10])

    # Merge the two dataframes, rounding all decimals to 4 places for formatting
    combinedDF = pd.merge(trace_dataframe, deployment_trace_dataframe,on='deploymentid',how='left')
    combinedDF = combinedDF.round(4)
    output(combinedDF)

    return 0

if __name__ == "__main__":
    main()
