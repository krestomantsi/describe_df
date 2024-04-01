# Describe 
A cli utility for plotting the statistics of each column of a local csv file (min,max,mean,std,most frequent value). Basically if describe from pandas was a cli tool. The column has to be of float dtype. Results are dumped both in stdout and saved locally into a describe_result.csv. 

# How to use
~/.cargo/bin/describe_df -f <file_name> 

# Installation 

## From source
go into the describe folder and run 
cargo install --path . --locked

# Goals
- Have the describe result csv file name to not be just describe_result.csv.
- Better stdout visualization (pretty table instead of pretty print of Vec<DescribeResult>)
- A faster most frequent value algorithm (low priority)
