This is the code corresponding to the paper: "**PCKT: Problem Composition in Knowledge Tracking**".

Here are the code notes:

1. Before running the code, you need to first download the dataset and put it into the "data" folder in the corresponding dataset folder. 

   Take dataset "Assist09" as an example:

   First, you need to first create a folder named "Assist09".

   Then, create another "data" folder inside the "Assist09" folder. 

   Finally, rename the downloaded dataset to "Assist09_original.csv" and put it into the "data" folder.

   You can download the dataset here:

   **Assist09**: https://drive.google.com/file/d/1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE/view

   **Assist12**: https://drive.google.com/file/d/1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY/view

2. The code runs in the following order, please be sure to run in order, otherwise an error will be reported: 

   ProcessData_09.py---->TrainEmbedding_09.py---->JointEmbedding_09.py---->TrainModel_09.py

   or

   ProcessData_12.py---->TrainEmbedding_12.py---->JointEmbedding_12.py---->TrainModel_12.py

3. You can use the following command to install the package: 

   ```shell
   pip lnstall tensorflow==1.15.0 pandas numpy scipy sklearn
   ```

   The version of python we are using is 3.7.0

4. If you have any questions, please feel free to contact us at 761744650@qq.com.