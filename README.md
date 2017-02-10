# AWS_model_files

The aim of this folder is to automate the creation of the instances on Amazon Web Services EC2. 
For each instances, the scripts will install Jupyter Notebook and allow the users to access it in the browser.
It will also install python 3. for the users who need it.
Furthermore, the scripts also allow the users to run several models in each instances in order to make a distributed system.
It would require some adjustments regarding the model the users want to use and also how they want to get the results into the master which would be the user computer probably.
For example, it is possible to use github to get all the results and compare them to get directly the model with the best parameters.

Be aware that you need at least to install the following packages :
  numpy,
  boto3,
  pexpect
  
